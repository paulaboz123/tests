"""
llm.py - every model call goes through here: chat, JSON output, image description, embeddings.

Two backends behind the same functions, chosen per job in .env (CHAT_/VISION_/EMBED_BACKEND):
- bifrost: the company gateway in front of Amazon Bedrock. It holds the AWS credentials; we
  only send its URL and our key. It offers two APIs and we can use either:
    OpenAI-style  <base>/v1/chat/completions, /v1/embeddings, /v1/models
    Anthropic     <base>/anthropic/v1/messages (the route Claude Code uses)
  GATEWAY_API=auto tries the first and switches to the second if the gateway doesn't have it.
  Plain HTTP via httpx, no OpenAI SDK (client's request).
- ollama: local models through Ollama's native API.
Scripts never talk HTTP themselves, so switching backend = changing .env, not code.

Safety and bookkeeping:
- Retries with backoff when the server is busy or unreachable (429, 5xx, timeouts), and
  readable errors for a wrong key (401/403) or an unknown model / wrong URL (404).
- Ollama silently drops the BEGINNING of a prompt longer than its context window; we
  refuse such prompts instead.
- One row per call in cost_log.csv: tokens, seconds, estimated USD.
- A spending lock for gateway calls: per day (DAILY_BUDGET_USD, never above $100) and per
  command (RUN_BUDGET_USD).

Smoke test:  uv run llm.py
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
import re
import ssl
import threading
import time
from datetime import date, datetime
from pathlib import Path

import httpx

import config

log = logging.getLogger("llm")

COST_FIELDS = ["time", "step", "backend", "model", "in_tokens", "in_tokens_est",
               "out_tokens", "seconds", "est_cost_usd"]
TRANSIENT_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class LLMError(RuntimeError):
    """Anything that went wrong talking to a model server, with a readable message."""


class BudgetExceeded(LLMError):
    """Today's estimated gateway spend reached DAILY_BUDGET_USD."""


# --- HTTP ---------------------------------------------------------------------------------
_client: httpx.Client | None = None


def _http() -> httpx.Client:
    """One shared connection pool. Honours HTTPS_PROXY and the optional company CA bundle."""
    global _client
    if _client is None:
        verify: ssl.SSLContext | bool = True
        if config.CA_BUNDLE:
            verify = ssl.create_default_context(cafile=config.CA_BUNDLE)
        _client = httpx.Client(timeout=httpx.Timeout(config.LLM_TIMEOUT_S, connect=20),
                               verify=verify, trust_env=True)
    return _client


def _gateway_url(path: str, api: str = "openai") -> str:
    """Full gateway address for an API path.
    openai:    '/chat/completions' -> '<base>/v1/chat/completions'
    anthropic: '/messages'         -> '<base>/anthropic/v1/messages'"""
    if not config.BIFROST_BASE_URL:
        raise LLMError("BIFROST_BASE_URL is empty in .env. Paste the gateway address from the platform "
                       "team's e-mail, or set the *_BACKEND settings to ollama to run locally.")
    prefix = "/anthropic/v1" if api == "anthropic" else "/v1"
    return config.BIFROST_BASE_URL + prefix + path


def _gateway_headers(api: str = "openai") -> dict[str, str]:
    """Headers that carry the key. 'auto' sends it both as Authorization: Bearer and as x-bf-vk
    (Bifrost's virtual-key header), so it works whichever one the gateway checks."""
    key = config.BIFROST_API_KEY
    if not key:
        raise LLMError("BIFROST_API_KEY is empty in .env. Paste the virtual key from the e-mail.")
    mode = config.BIFROST_AUTH_HEADER
    if mode in ("x-bf-vk", "x-api-key"):
        headers = {mode: key}
    elif mode == "authorization":
        headers = {"Authorization": f"Bearer {key}"}
    else:
        headers = {"Authorization": f"Bearer {key}", "x-bf-vk": key}
    if api == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _explain(status: int, url: str, body: str) -> str:
    """Readable message for an HTTP error that retrying will not fix."""
    if status in (401, 403):
        return (f"The gateway rejected the request (HTTP {status}). Check BIFROST_API_KEY in .env, "
                f"and ask the platform team whether your key may use this model. Server said: {body}")
    if status == 404:
        return (f"Not found (HTTP 404) at {url}. Either BIFROST_BASE_URL is wrong, or the model id is "
                f"unknown to the gateway (list them: uv run doctor.py --models). Server said: {body}")
    return f"The model server refused the request (HTTP {status}) at {url}: {body}"


def _request(url: str, payload: dict | None = None, headers: dict | None = None,
             attempts: int = 4) -> dict:
    """GET (payload None) or POST JSON, with retries for temporary problems only."""
    hint = ("Is Ollama running? Check with: ollama list" if url.startswith(config.OLLAMA_URL)
            else "Can this laptop reach the gateway (VPN, proxy)?")
    err = ""
    for attempt in range(1, attempts + 1):
        wait = 2.0 ** attempt
        try:
            r = (_http().post(url, json=payload, headers=headers) if payload is not None
                 else _http().get(url, headers=headers))
        except httpx.TransportError as e:   # connection refused, timeout, proxy, TLS ...
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise LLMError(f"TLS certificate check failed for {url}. The company network "
                               "probably inspects TLS: set CA_BUNDLE in .env to the company "
                               "certificate bundle (.pem).") from e
            err = f"{type(e).__name__}: {e}"
        else:
            if r.status_code < 400:
                try:
                    return r.json()
                except ValueError:   # e.g. an HTML login page or a web UI instead of the API
                    raise LLMError(
                        f"{url} answered HTTP {r.status_code} but not with JSON "
                        f"(content-type: {r.headers.get('content-type', '?')}). The address is probably "
                        f"not the API, or a login page / proxy answered. Start of the answer: "
                        f"{' '.join(r.text[:200].split())!r}") from None
            body = r.text[:500]
            if r.status_code not in TRANSIENT_STATUS:
                raise LLMError(_explain(r.status_code, url, body))
            err = f"HTTP {r.status_code}: {body[:200]}"
            retry_after = r.headers.get("retry-after", "")
            if retry_after.replace(".", "", 1).isdigit():
                wait = float(retry_after)
        if attempt == attempts:
            raise LLMError(f"No usable answer from {url} after {attempts} attempts ({err}). {hint}")
        log.warning("Model server problem (%s), retrying in %.0fs", err, min(wait, 60))
        time.sleep(min(wait, 60))
    raise AssertionError("unreachable")


# --- Costs and budget -----------------------------------------------------------------------
_spent_today: float | None = None
_spent_session = 0.0
_cost_lock = threading.Lock()   # course.py makes chapter calls in parallel
_calls: list[dict] = []          # every priced call of this process, for the run card of each result


def meter_mark() -> int:
    """A bookmark in this process's list of model calls (see calls_since)."""
    with _cost_lock:
        return len(_calls)


def calls_since(mark: int) -> list[dict]:
    """The model calls made since meter_mark() returned `mark`: step, backend, model, tokens,
    seconds and estimated cost of each (runcard.py adds them up per result)."""
    with _cost_lock:
        return [dict(c) for c in _calls[mark:]]


def approx_tokens(text: str) -> int:
    """Rough token count without a tokenizer: ~3.5 characters per token for English."""
    return int(len(text) / 3.5) + 1


def price(model: str, backend: str) -> tuple[float, float]:
    """Estimated USD per 1M tokens (input, output) for a model."""
    if backend == "ollama":
        return 0.0, 0.0
    if model == config.CHAT_MODEL and (config.PRICE_IN_PER_MTOK or config.PRICE_OUT_PER_MTOK):
        return config.PRICE_IN_PER_MTOK, config.PRICE_OUT_PER_MTOK
    name = model.lower()
    factor = config.REGIONAL_PREMIUM if re.search(r"(^|[/.])(eu|us|apac|jp|au|ca)\.", name) else 1.0
    for part, p_in, p_out in config.PRICES:
        if part in name:
            return p_in * factor, p_out * factor
    return 5.0 * factor, 25.0 * factor   # unknown model: assume a high price, never zero


def spent_today() -> float:
    """Estimated USD spent today, from cost_log.csv."""
    if not config.COST_LOG.exists():
        return 0.0
    today = date.today().isoformat()
    total = 0.0
    with open(config.COST_LOG, encoding="utf-8-sig", newline="") as f:   # -sig: opened and saved in Excel
        for row in csv.reader(f):
            if row and row[0].strip().startswith(today):
                try:
                    total += float(row[-1] or 0)   # the cost is the last column in every layout
                except ValueError:
                    continue
    return total


def spent_this_run() -> float:
    """Estimated USD spent by this process so far (e.g. for one video)."""
    return _spent_session


def _check_budget() -> None:
    """Stop before a gateway call if today's or this command's estimated spend is at its limit."""
    global _spent_today
    if _spent_today is None:
        _spent_today = spent_today()
    if _spent_today >= config.DAILY_BUDGET_USD:
        raise BudgetExceeded(
            f"Spending lock: today's estimated spend ${_spent_today:.2f} reached DAILY_BUDGET_USD="
            f"${config.DAILY_BUDGET_USD:.2f} (hard maximum ${config.HARD_DAILY_LIMIT_USD:.0f}). "
            f"No more gateway calls today. Raise DAILY_BUDGET_USD in .env only if this is expected.")
    if _spent_session >= config.RUN_BUDGET_USD:
        raise BudgetExceeded(
            f"Spending lock: this command already spent ~${_spent_session:.2f}, the limit per command "
            f"(RUN_BUDGET_USD) is ${config.RUN_BUDGET_USD:.2f}. Something may be looping; check the log.")


def log_fixed_cost(step: str, model: str, cost: float, secs: float, backend: str = "bifrost",
                   characters: int = 0) -> None:
    """Count a call with a fixed price (a generated picture, a cloud voice billed per character)
    in the log, the spending lock and the run card."""
    global _spent_today, _spent_session
    with _cost_lock:
        _spent_session += cost
        if _spent_today is not None:
            _spent_today += cost
        new = not config.COST_LOG.exists()
        _calls.append({"step": step, "backend": backend, "model": model, "in_tokens": 0, "out_tokens": 0,
                       "characters": characters, "seconds": round(secs, 1), "cost_usd": round(cost, 6)})
        with open(config.COST_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COST_FIELDS)
            w.writerow([datetime.now().isoformat(timespec="seconds"), step, backend, model, characters, 0, 0,
                        round(secs, 1), round(cost, 6)])


def _log_cost(step: str, backend: str, model: str, tin: int, tin_est: int, tout: int,
              secs: float, cache_write: int = 0, cache_read: int = 0) -> None:
    """Append one row to cost_log.csv (creates the file with a header on first use). Prompt-cache
    tokens are priced as Claude bills them: written x1.25, read x0.1 of the input price."""
    global _spent_today, _spent_session
    p_in, p_out = price(model, backend)
    cost = ((tin or (0 if cache_write or cache_read else tin_est)) / 1e6 * p_in + tout / 1e6 * p_out
            + cache_write / 1e6 * p_in * 1.25 + cache_read / 1e6 * p_in * 0.1)
    with _cost_lock:
        _spent_session += cost
        if _spent_today is not None:
            _spent_today += cost
        new = not config.COST_LOG.exists()
        _calls.append({"step": step, "backend": backend, "model": model,
                       "in_tokens": (tin or tin_est) + cache_write + cache_read, "out_tokens": tout,
                       "cache_read_tokens": cache_read, "seconds": round(secs, 1), "cost_usd": round(cost, 6)})
        with open(config.COST_LOG, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(COST_FIELDS)
            w.writerow([datetime.now().isoformat(timespec="seconds"), step, backend, model,
                        tin, tin_est, tout, round(secs, 1), round(cost, 6)])


# --- Chat -----------------------------------------------------------------------------------
_structured_ok = True   # set to False if the gateway refuses response_format


def _gateway_text(data: dict) -> tuple[str, bool]:
    """Reply text from an OpenAI-style response, and whether it was cut off by max_tokens."""
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, list):   # some gateways return content parts
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    if not content and msg.get("tool_calls"):   # structured output delivered as a tool call
        content = msg["tool_calls"][0].get("function", {}).get("arguments", "")
    return content or "", choice.get("finish_reason") == "length"


CACHE_MIN_CHARS = 4500   # ~1,100 tokens: Claude caches only prompt parts of at least 1,024 tokens
_cache_ok = True          # set to False if the gateway refuses cache_control


def _cached(text: str) -> list[dict] | str:
    """A long prompt part as a block marked for caching (re-read later at ~10% of the price)."""
    if not (config.PROMPT_CACHE and _cache_ok) or len(text) < CACHE_MIN_CHARS:
        return text
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _to_anthropic(messages: list[dict], model: str, max_tokens: int, temperature: float) -> dict:
    """OpenAI-style messages -> Anthropic Messages request (system prompt separate, images as
    base64 blocks). The system prompt and the first user message (instructions and material)
    are marked for prompt caching when they are long: follow-up calls with the same start
    (self-review, corrections) read them from the cache."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    out = []
    for m in messages:
        if m["role"] == "system":
            continue
        content = m["content"]
        if isinstance(content, list):
            blocks = []
            for part in content:
                if part.get("type") == "text":
                    blocks.append({"type": "text", "text": part["text"]})
                elif part.get("type") == "image_url":
                    url = part["image_url"]["url"]            # data:image/jpeg;base64,....
                    media_type, data = url[5:].split(";base64,", 1)
                    blocks.append({"type": "image",
                                   "source": {"type": "base64", "media_type": media_type, "data": data}})
            content = blocks
        if isinstance(content, str) and not out and m["role"] == "user":
            content = _cached(content)
        out.append({"role": m["role"], "content": content})
    payload = {"model": model, "max_tokens": max_tokens, "temperature": temperature, "messages": out}
    if system:
        payload["system"] = _cached(system)
    return payload


_route: str | None = None   # gateway API that worked: "openai" or "anthropic"


def gateway_route() -> str:
    """Which gateway API is in use ('openai', 'anthropic', or 'not used yet')."""
    return _route or "not used yet"


def _gateway_chat(messages: list[dict], model: str, max_tokens: int, temperature: float,
                  schema: dict | None) -> tuple[str, bool, int, int]:
    """One chat call through the gateway. Returns (text, cut off?, input tokens, output tokens).
    With GATEWAY_API=auto it uses the OpenAI-style route, and switches to the Anthropic route
    for the rest of the run if the gateway doesn't offer the first one."""
    global _structured_ok, _route
    _check_budget()
    # Prompt caching needs the Anthropic route, so with caching on, auto tries that one first
    # (Claude Code uses the same route with the same key) and falls back to the OpenAI-style one.
    preferred = "anthropic" if config.GATEWAY_API == "anthropic" or (
        config.GATEWAY_API == "auto" and config.PROMPT_CACHE) else "openai"
    api = _route or preferred
    if api == "anthropic" and _route is None and config.GATEWAY_API == "auto":
        try:
            result = _anthropic_chat(messages, model, max_tokens, temperature)
        except LLMError as e:
            if not ("HTTP 404" in str(e) or "HTTP 405" in str(e) or "not with JSON" in str(e)):
                raise
            log.info("Anthropic route not available (%s), using the OpenAI-style route", str(e)[:120])
            _route = "openai"
            api = "openai"
        else:
            _route = "anthropic"
            return result

    if api == "openai":
        payload = {"model": model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens}
        if schema and _structured_ok:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "output", "schema": schema}}
        try:
            data = _request(_gateway_url("/chat/completions"), payload, _gateway_headers())
        except LLMError as e:
            msg = str(e)
            if "response_format" in payload and "HTTP 400" in msg:
                log.warning("Gateway refused response_format, asking for JSON in the prompt instead")
                _structured_ok = False
                payload.pop("response_format")
                data = _request(_gateway_url("/chat/completions"), payload, _gateway_headers())
            elif config.GATEWAY_API == "auto" and _route is None and preferred == "openai" and (
                    "HTTP 404" in msg or "HTTP 405" in msg or "not with JSON" in msg):
                log.info("OpenAI-style route not available (%s), trying the Anthropic route", msg[:120])
                try:
                    result = _anthropic_chat(messages, model, max_tokens, temperature)
                except LLMError as e2:
                    raise LLMError(f"Both gateway routes failed.\n  OpenAI-style: {msg}\n  Anthropic: {e2}") from e2
                _route = "anthropic"
                return result
            else:
                raise
        _route = "openai"
        text, cut_off = _gateway_text(data)
        usage = data.get("usage") or {}
        return text, cut_off, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)

    result = _anthropic_chat(messages, model, max_tokens, temperature)
    _route = "anthropic"
    return result


_last_cache: dict = {}   # cache tokens of the last Anthropic-route call (for the cost log)


def _anthropic_chat(messages: list[dict], model: str, max_tokens: int,
                    temperature: float) -> tuple[str, bool, int, int]:
    """Chat through the gateway's Anthropic route. Returns (text, cut off?, tokens in, out);
    prompt-cache tokens (written / read) are kept in _last_cache for the cost log."""
    global _cache_ok
    payload = _to_anthropic(messages, model, max_tokens, temperature)
    try:
        data = _request(_gateway_url("/messages", "anthropic"), payload, _gateway_headers("anthropic"))
    except LLMError as e:
        if "cache" in str(e).lower() and _cache_ok and "HTTP 4" in str(e):
            log.warning("Gateway refused prompt caching; continuing without it (%s)", str(e)[:120])
            _cache_ok = False
            payload = _to_anthropic(messages, model, max_tokens, temperature)
            data = _request(_gateway_url("/messages", "anthropic"), payload, _gateway_headers("anthropic"))
        else:
            raise
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage") or {}
    _last_cache.clear()
    _last_cache.update(write=int(usage.get("cache_creation_input_tokens") or 0),
                       read=int(usage.get("cache_read_input_tokens") or 0))
    return (text, data.get("stop_reason") == "max_tokens",
            usage.get("input_tokens", 0), usage.get("output_tokens", 0))


def _allowed(model: str) -> None:
    """Company rule: no models of Chinese origin, whatever the backend (safety.py)."""
    import safety
    why = safety.model_not_allowed(model)
    if why:
        raise LLMError(why + ". Choose another model in .env.")


def chat(prompt: str | list[dict], *, system: str = "", schema: dict | None = None,
         step: str = "chat", model: str | None = None, temperature: float = 0.3,
         max_tokens: int | None = None) -> str:
    """
    Send a conversation to the chat model and return its reply as text.

    prompt  - a user message (str) or a full message list [{"role": ..., "content": ...}]
    system  - optional system prompt (e.g. the body of a SKILL.md)
    schema  - optional JSON schema; the model must answer with matching JSON
    step    - label for cost_log.csv, so costs can be split per pipeline step
    """
    backend = config.CHAT_BACKEND
    model = model or config.CHAT_MODEL
    _allowed(model)
    max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS
    if system:   # every prompt carries material from files: it is data, never instructions
        import safety
        system = f"{system}\n\n{safety.DATA_RULE}"
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
    if schema and backend == "bifrost":   # also in the prompt, in case response_format is ignored
        system = (system + "\n\n" if system else "") + (
            "Answer with one JSON object that matches this JSON schema, and nothing else:\n"
            + json.dumps(schema))
    if system:
        messages = [{"role": "system", "content": system}] + messages
    tin_est = approx_tokens("".join(str(m["content"]) for m in messages))

    t0 = time.perf_counter()
    if backend == "ollama":
        if tin_est + max_tokens > config.NUM_CTX:
            raise LLMError(f"Prompt is ~{tin_est} tokens; with {max_tokens} for the answer it "
                           f"exceeds NUM_CTX={config.NUM_CTX}. Send less material (MAX_MATERIAL_WORDS) "
                           f"or raise NUM_CTX in .env.")
        payload = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
                   "options": {"num_ctx": config.NUM_CTX, "num_predict": max_tokens,
                               "temperature": temperature, "seed": 42}}
        if schema:
            payload["format"] = schema
        data = _request(config.OLLAMA_URL + "/api/chat", payload)
        text = data["message"]["content"]
        tin, tout = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        cut_off = data.get("done_reason") == "length"
    elif backend == "bifrost":
        if not model:
            raise LLMError("CHAT_MODEL is empty in .env. Paste the Sonnet model id from the e-mail.")
        _last_cache.clear()
        text, cut_off, tin, tout = _gateway_chat(messages, model, max_tokens, temperature, schema)
    else:
        raise LLMError(f"Unknown CHAT_BACKEND={backend!r} in .env, use one of {config.BACKENDS}")

    secs = time.perf_counter() - t0
    cw, cr = _last_cache.get("write", 0), _last_cache.get("read", 0)
    _log_cost(step, backend, model, tin, tin_est, tout, secs, cw, cr)
    log.info("%s: %s in / %s out tokens%s, %.0fs (%s)", step, tin or f"~{tin_est}", tout,
             f" (cache: {cw} written, {cr} read)" if cw or cr else "", secs, model)
    if cut_off:
        log.warning("%s: answer hit the %s-token output limit and is probably cut off "
                    "(raise MAX_OUTPUT_TOKENS in .env)", step, max_tokens)
    return text


def parse_json(text: str) -> dict:
    """JSON object from a model reply, tolerating ```json fences or text around it."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1] if "\n" in s else s
        s = s.rsplit("```", 1)[0]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise LLMError(f"Model reply contains no JSON object. Start of reply: {text[:200]!r}")
    try:
        return json.loads(s[start:end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Model did not return valid JSON: {e}. Start of reply: {text[:200]!r}") from e


def chat_json(prompt: str | list[dict], schema: dict, **kwargs) -> dict:
    """chat() that returns parsed JSON. Raises LLMError if the reply is not valid JSON."""
    return parse_json(chat(prompt, schema=schema, **kwargs))


# --- Images ---------------------------------------------------------------------------------
def describe_image(image_path: Path | str, prompt: str, *, model: str | None = None,
                   step: str = "describe_image", max_side: int = 1280) -> str:
    """Ask the vision model about an image. The image is shrunk first: slide text stays
    readable at 1280 px, and both local time and gateway cost grow with pixels."""
    from PIL import Image

    backend = config.VISION_BACKEND
    model = model or config.VISION_MODEL
    _allowed(model)
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        width, height = im.size
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    tin_est = approx_tokens(prompt) + width * height // 750   # Claude: ~1 token per 750 pixels

    t0 = time.perf_counter()
    if backend == "ollama":
        data = _request(config.OLLAMA_URL + "/api/chat", {
            "model": model, "stream": False, "keep_alive": "30m",
            "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            "options": {"num_ctx": 8192, "num_predict": 1024, "temperature": 0.1, "seed": 42},
        })
        text = data["message"]["content"]
        tin, tout = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
    elif backend == "bifrost":
        if not model:
            raise LLMError("VISION_MODEL is empty in .env (it defaults to CHAT_MODEL).")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
        text, _, tin, tout = _gateway_chat(messages, model, 1024, 0.1, None)
    else:
        raise LLMError(f"Unknown VISION_BACKEND={backend!r} in .env, use one of {config.BACKENDS}")
    secs = time.perf_counter() - t0
    _log_cost(step, backend, model, tin, tin_est, tout, secs)
    log.info("%s: %s in / %s out tokens, %.0fs (%s)", step, tin or f"~{tin_est}", tout, secs, model)
    return text.strip()


# --- Embeddings -------------------------------------------------------------------------------
_embed_one_by_one = False   # set when the gateway/model takes only one text per request


def _gateway_embed(texts: list[str]) -> tuple[list[list[float]], int]:
    """One embeddings request to the gateway. Returns (vectors in input order, input tokens)."""
    data = _request(_gateway_url("/embeddings"), {"model": config.EMBED_MODEL, "input": texts},
                    _gateway_headers())
    rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in rows], (data.get("usage") or {}).get("prompt_tokens", 0)


def embed(texts: list[str], *, step: str = "embed", batch_size: int = 32) -> list[list[float]]:
    """Turn texts into embedding vectors, in batches. Returns one vector per input text."""
    global _embed_one_by_one
    backend = config.EMBED_BACKEND
    _allowed(config.EMBED_MODEL)
    vectors: list[list[float]] = []
    tin_total, t0 = 0, time.perf_counter()
    if backend == "bifrost":
        _check_budget()
    elif backend != "ollama":
        raise LLMError(f"Unknown EMBED_BACKEND={backend!r} in .env, use one of {config.BACKENDS}")

    i = 0
    while i < len(texts):
        size = 1 if (backend == "bifrost" and _embed_one_by_one) else batch_size
        batch = texts[i:i + size]
        if backend == "ollama":
            data = _request(config.OLLAMA_URL + "/api/embed",
                            {"model": config.EMBED_MODEL, "input": batch, "keep_alive": "30m"})
            got, tin = data["embeddings"], data.get("prompt_eval_count", 0)
        else:
            try:
                got, tin = _gateway_embed(batch)
            except LLMError as e:
                if len(batch) == 1 or "HTTP 400" not in str(e):
                    raise
                got, tin = [], 0
            if len(got) != len(batch):   # e.g. Titan embeds one text per call
                if len(batch) == 1:
                    raise LLMError(f"Gateway returned {len(got)} embeddings for 1 text")
                log.info("Embedding model takes one text per request, switching to one by one")
                _embed_one_by_one = True
                continue
        vectors.extend(got)
        tin_total += tin
        i += len(batch)
        log.debug("embedded %s/%s", i, len(texts))
    if len(vectors) != len(texts):
        raise LLMError(f"Asked for {len(texts)} embeddings, got {len(vectors)}")
    _log_cost(step, backend, config.EMBED_MODEL, tin_total, sum(map(approx_tokens, texts)), 0,
              time.perf_counter() - t0)
    return vectors


# --- Model lists (doctor.py) ------------------------------------------------------------------
def list_models(backend: str) -> list[str]:
    """Model names the backend offers (the gateway may not offer a list; then LLMError)."""
    if backend == "ollama":
        data = _request(config.OLLAMA_URL + "/api/tags", attempts=1)
        return [m["name"] for m in data.get("models", [])]
    errors = []
    for api in ("openai", "anthropic"):
        try:
            data = _request(_gateway_url("/models", api), headers=_gateway_headers(api), attempts=2)
            return sorted(m.get("id", "") for m in data.get("data", []))
        except LLMError as e:
            errors.append(str(e)[:150])
    raise LLMError("No model list from the gateway: " + " | ".join(errors))


# Claude models on Amazon Bedrock that doctor.py --models tries when the gateway gives no list.
# Only the part after the provider prefix; the prefix is taken from CHAT_MODEL
# (e.g. "bedrock/eu.anthropic."). An id that doesn't exist simply shows as unknown.
KNOWN_CLAUDE = (
    "claude-haiku-4-5-20251001-v1:0", "claude-sonnet-4-5-20250929-v1:0", "claude-opus-4-5-20251101-v1:0",
    "claude-sonnet-4-6", "claude-opus-4-6-v1", "claude-opus-4-7", "claude-opus-4-1-20250805-v1:0",
    "claude-sonnet-4-20250514-v1:0", "claude-3-7-sonnet-20250219-v1:0", "claude-3-5-sonnet-20240620-v1:0",
    "claude-3-haiku-20240307-v1:0",
)


def candidate_models(extra: list[str] | None = None) -> dict[str, list[str]]:
    """Model ids worth trying, each with the .env settings that use it: the configured ones,
    the Claude ids from the platform team's e-mail block, the known Claude models with the same
    prefix as CHAT_MODEL, and any extra ids."""
    found: dict[str, list[str]] = {}

    def add(model: str, where: str = "", always: bool = False) -> None:
        model = (model or "").strip()
        if model and (always or "/" in model or model.startswith(("eu.", "us.", "global.", "anthropic."))):
            found.setdefault(model, [])
            if where and where not in found[model]:
                found[model].append(where)

    for name in ("CHAT_MODEL", "VISION_MODEL", "REVIEW_MODEL", "RERANK_MODEL", "JUDGE_MODEL"):
        add(getattr(config, name, ""), name)
    for name in ("OPUS", "SONNET", "HAIKU"):
        add(config._get(f"ANTHROPIC_DEFAULT_{name}_MODEL"), f"ANTHROPIC_DEFAULT_{name}_MODEL")
    for part in config._get("MODEL_CHOICES").split(","):
        add(part.partition("=")[2], "MODEL_CHOICES")
    base = config.CHAT_MODEL or config._get("ANTHROPIC_DEFAULT_SONNET_MODEL")
    prefix = base[:base.find("claude-")] if "claude-" in base else "bedrock/eu.anthropic."
    for name in KNOWN_CLAUDE:
        add(prefix + name)
    for model in extra or []:
        add(model, "--try", always=True)
    return found


def try_model(model: str) -> tuple[str, str]:
    """One tiny call (a few tokens, a fraction of a cent) to see whether the key may use a model.
    Returns (status, detail): 'ok', 'no access', 'unknown id', 'not allowed' or 'error'."""
    try:
        _allowed(model)
    except LLMError as e:
        return "not allowed", str(e)[:120]
    messages = [{"role": "user", "content": "Reply with the single word OK."}]
    routes = ([_route] if _route else ["anthropic"] if config.GATEWAY_API == "anthropic"
              else ["openai"] if config.GATEWAY_API == "openai" else ["openai", "anthropic"])
    msg = ""
    for api in routes:
        t0 = time.perf_counter()
        try:
            if api == "anthropic":
                data = _request(_gateway_url("/messages", "anthropic"),
                                {"model": model, "max_tokens": 5, "messages": messages},
                                _gateway_headers("anthropic"), attempts=1)
                usage = data.get("usage") or {}
                tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            else:
                data = _request(_gateway_url("/chat/completions"),
                                {"model": model, "max_tokens": 5, "messages": messages},
                                _gateway_headers(), attempts=1)
                usage = data.get("usage") or {}
                tin, tout = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        except LLMError as e:
            msg = str(e)
            continue
        _log_cost("doctor", "bifrost", model, tin, 10, tout, time.perf_counter() - t0)
        return "ok", "answers"
    code = re.search(r"HTTP (\d{3})", msg)
    said = msg.split("Server said:")[-1].strip() if "Server said:" in msg else msg
    said = " ".join(said.split())[:110]
    if code and code.group(1) in ("401", "403"):
        return "no access", f"HTTP {code.group(1)}: {said}"
    if code and code.group(1) in ("400", "404"):
        return "unknown id", f"HTTP {code.group(1)}: {said}"
    return "error", said


def _smoke_test() -> None:
    """Call each configured model once: plain chat, JSON output, embeddings."""
    print(f"chat: {config.CHAT_BACKEND} {config.CHAT_MODEL} | vision: {config.VISION_BACKEND} "
          f"{config.VISION_MODEL} | embed: {config.EMBED_BACKEND} {config.EMBED_MODEL}\n")
    print("1) Plain chat:")
    print(chat("Explain in 2 sentences what a Teams meeting transcript is.", step="smoke_test",
               max_tokens=200))
    print("\n2) Structured JSON:")
    schema = {"type": "object",
              "properties": {"term": {"type": "string"}, "definition": {"type": "string"}},
              "required": ["term", "definition"]}
    print(chat_json("Define the term 'VTT file'.", schema=schema, step="smoke_test", max_tokens=300))
    print("\n3) Embeddings:")
    vecs = embed(["budget planning", "quarterly forecast"], step="smoke_test")
    print(f"{len(vecs)} vectors of length {len(vecs[0])}")
    print(f"\nOK. Estimated cost of this test: ${spent_this_run():.4f}. Calls logged in {config.COST_LOG.name}")


if __name__ == "__main__":
    config.setup_logging()
    try:
        _smoke_test()
    except LLMError as err:
        log.error("%s", err)
        raise SystemExit(1) from None
