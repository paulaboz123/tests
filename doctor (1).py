"""
doctor.py - checks that everything the pipeline needs is installed and reachable.

Run it first, and again whenever something "is not recognized" or "can't connect":
  uv run doctor.py            # all checks (a gateway check costs a fraction of a cent)
  uv run doctor.py --models   # also: which Claude models your key may use (tries each once)
  uv run doctor.py --models --try <id> <id>   # ... plus these ids
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys

import config
import llm
import media


def show_access(extra: list[str]) -> None:
    """Try each candidate model with one tiny call and print which ones the key may use."""
    found = llm.candidate_models(extra)
    print(f"\nModels your key may use ({len(found)} tried, one tiny call each, under a cent in total):")
    width = max(map(len, found), default=0)
    ok = []
    for model, used_in in found.items():
        status, detail = llm.try_model(model)
        where = f"  <- {', '.join(used_in)}" if used_in else ""
        if status == "ok":
            ok.append(model)
            print(f"  [OK]          {model:<{width}}{where}")
        else:
            print(f"  [{status.upper():<11}] {model:<{width}}{where}\n                {detail}")
    def newest(name: str) -> str:
        """The id from the e-mail block if it works, else the newest working one of that family."""
        def key(m: str) -> tuple:
            v = re.search(rf"{name}-(\d+)(?:-(\d)(?=\D|$))?", m.lower())
            return (bool(found[m]), (int(v.group(1)), int(v.group(2) or 0)) if v else (0, 0))
        return max((m for m in ok if name in m.lower()), key=key, default="")

    pick = {name: newest(name) for name in ("haiku", "opus")}
    print("\n  For answers and their evaluation (.env):")
    print(f"    RERANK_MODEL={pick['haiku']}" + ("" if pick["haiku"] else "    (no Haiku: leave empty = CHAT_MODEL)"))
    print(f"    JUDGE_MODEL={pick['opus']}" + ("" if pick["opus"] else "    (no Opus: leave empty = CHAT_MODEL)"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", action="store_true",
                    help="list the model ids and try each Claude model once: which ones your key may use")
    ap.add_argument("--try", dest="try_ids", nargs="+", default=[], metavar="ID",
                    help="with --models: also try these model ids")
    args = ap.parse_args()
    args.models = args.models or bool(args.try_ids)
    problems = 0

    def report(ok: bool, label: str, fix: str = "") -> bool:
        """Print one check line; count it as a problem if it failed. Returns ok."""
        nonlocal problems
        print(f"[{'OK' if ok else 'FAIL'}] {label}")
        if not ok:
            print(f"       fix: {fix}")
            problems += 1
        return ok

    report(sys.version_info >= (3, 11), f"Python {sys.version.split()[0]}", "install Python 3.12")
    for name, exe in [("ffmpeg", config.FFMPEG), ("ffprobe", config.FFPROBE)]:
        report(shutil.which(exe) is not None, name, "winget install Gyan.FFmpeg, then close and reopen VS Code")
    try:
        langs = media.tesseract_languages()
        missing = [lang for lang in config.OCR_LANG.split("+") if lang not in langs]
        report(not missing, "tesseract", f"language pack missing: {missing}")
    except media.MediaError as e:
        report(False, f"tesseract ({config.TESSERACT})",
               f"{str(e)[:200]}\n       Installed already? Find tesseract.exe and put its full path in .env: "
               f"TESSERACT=C:\\...\\tesseract.exe")

    jobs = [("chat", config.CHAT_BACKEND, config.CHAT_MODEL),
            ("vision", config.VISION_BACKEND, config.VISION_MODEL),
            ("embed", config.EMBED_BACKEND, config.EMBED_MODEL)]
    print("\nModels: " + " | ".join(f"{j}: {b} {m}" for j, b, m in jobs))
    for job, backend, _ in jobs:
        report(backend in config.BACKENDS, f"{job} backend '{backend}'", f"use one of {config.BACKENDS}")

    # --- Gateway ---
    if "bifrost" in {b for _, b, _ in jobs}:
        if report(bool(config.BIFROST_BASE_URL and config.BIFROST_API_KEY),
                  "BIFROST_BASE_URL and BIFROST_API_KEY set in .env",
                  "ask the platform team for the gateway URL and a key for local development"):
            try:
                models = llm.list_models("bifrost")
                report(True, f"gateway reachable: {len(models)} models listed")
                if args.models:
                    print("       " + "\n       ".join(models or ["(empty list)"]))
            except llm.LLMError as e:   # some gateways don't offer a list; not a problem
                print(f"[INFO] gateway gives no model list ({str(e)[:120]})."
                      + (" Trying the models one by one below." if args.models else
                         " To see which models your key may use: uv run doctor.py --models"))
            if config.CHAT_BACKEND == "bifrost":
                if report(bool(config.CHAT_MODEL), "CHAT_MODEL set",
                          "paste the Sonnet model id from the e-mail into CHAT_MODEL in .env"):
                    try:
                        reply = llm.chat("Reply with the single word OK.", step="doctor", max_tokens=10)
                        report(True, f"chat model answers: {reply.strip()[:30]!r} "
                                     f"(gateway route: {llm.gateway_route()})")
                    except llm.LLMError as e:
                        report(False, f"chat model {config.CHAT_MODEL}", str(e))
            if args.models:
                show_access(args.try_ids)
            if config.EMBED_BACKEND == "bifrost":
                try:
                    dim = len(llm.embed(["test"], step="doctor")[0])
                    report(True, f"embedding model answers ({dim} dimensions)")
                except llm.LLMError as e:
                    report(False, f"embedding model {config.EMBED_MODEL}",
                           f"{str(e)[:200]}\n       Search can run locally instead: EMBED_BACKEND=ollama, "
                           f"EMBED_MODEL=nomic-embed-text (ollama pull nomic-embed-text)")

    # --- Local models ---
    local = [(j, m) for j, b, m in jobs if b == "ollama"]
    if local or args.models:
        try:
            have = llm.list_models("ollama")
            report(True, f"Ollama at {config.OLLAMA_URL}")
            for job, model in local:
                full = model if ":" in model else model + ":latest"
                if job == "vision" and full not in have:   # optional step
                    print(f"[WARN] vision model {model} not pulled (only needed for pipeline.py --vision)")
                    continue
                report(full in have, f"{job} model {model}", f"ollama pull {model}")
            if args.models:
                print("       " + "\n       ".join(have or ["(no local models)"]))
        except llm.LLMError:
            if local:
                report(False, f"Ollama at {config.OLLAMA_URL}", "start the Ollama app, or: ollama serve")

    # --- Voice ---
    try:
        import tts
        ok, detail = tts.check()
        report(ok, f"voice ({config.TTS_ENGINE}): {detail}",
               "uv run tts.py --download   (or TTS_ENGINE=system in .env)")
    except Exception as e:
        report(False, f"voice ({config.TTS_ENGINE})", str(e))

    # --- Data ---
    if config.INPUT_DIR.exists():
        files = [p for p in config.INPUT_DIR.rglob("*") if p.is_file()]
        recordings = sum(p.suffix.lower() in config.MEDIA_EXT for p in files)
        transcripts = sum(p.suffix.lower() in config.TRANSCRIPT_EXT for p in files)
        report(recordings > 0, f"{config.INPUT_DIR}: {recordings} recordings, {transcripts} transcripts",
               "copy the recordings (.mp4/.m4a) and transcripts (.vtt/.docx) there")
    else:
        report(False, f"{config.INPUT_DIR} exists", f"create the folder {config.INPUT_DIR}")
    free_gb = shutil.disk_usage(config.ROOT).free / 1e9
    report(free_gb > 5, f"{free_gb:.0f} GB free disk", "free up space")
    # --- Slides ---
    try:
        import slides_html
        ok, detail = slides_html.available()
        print(f"[{'OK' if ok else 'INFO'}] designer slides (browser): {detail}"
              + ("" if ok else "  -> uv add playwright   (the simple slides are used until then)"))
    except Exception as e:   # noqa: BLE001
        print(f"[INFO] designer slides: {e}")
    print(f"[INFO] style: {config.STYLE}; pictures: " + (f"AI ({config.IMAGE_MODEL})" if config.IMAGE_MODEL
                                                          else "built-in illustrations (IMAGE_MODEL not set)"))
    # --- Options to compare (benchmark.py): extra voices and video engines ---
    import remotion_render
    import tts
    for engine in ("elevenlabs",):
        saved = config.TTS_ENGINE
        config.TTS_ENGINE = engine
        ok, detail = tts.check()
        config.TTS_ENGINE = saved
        print(f"[{'OK' if ok else 'INFO'}] voice to compare, {engine}: {detail}")
    if config.RENDERER.startswith("remotion"):   # a fresh copy of the project: install its packages once
        remotion_render.ensure_installed()
    ok, detail = remotion_render.available()
    print(f"[{'OK' if ok else 'INFO'}] video engine to compare, remotion: {detail}"
          + (f" (draws with {remotion_render.cli()[0]})" if ok else ""))
    print(f"\nSpending lock: today ~${llm.spent_today():.2f} of ${config.DAILY_BUDGET_USD:.0f}/day "
          f"(hard maximum ${config.HARD_DAILY_LIMIT_USD:.0f}), at most ${config.RUN_BUDGET_USD:.0f} per command")

    print("\nAll good." if problems == 0 else f"\n{problems} problem(s) to fix first.")
    steps = config.ROOT / "NEXT_STEPS.md"
    if steps.exists():
        todo = [x[6:].strip() for x in steps.read_text(encoding="utf-8").splitlines() if x.startswith("- [ ]")]
        if todo:
            print(f"\nReminder: {len(todo)} step(s) left to the demo (NEXT_STEPS.md). Next: {todo[0]}")
    return problems


if __name__ == "__main__":
    sys.exit(main())
