"""
A stand-in for the Bifrost gateway, used only by the tests: same routes and answer formats,
canned answers, no real model and no cost. Behaviour can be switched per test via `state`.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KEY = "test-key"
MODELS = ["bedrock/eu.anthropic.claude-sonnet-5-test", "bedrock/amazon.titan-embed-text-v2:0", "bedrock/image-test",
          "bedrock/eu.anthropic.claude-opus-5-5-test"]


def vector(text: str, dim: int = 128) -> list[float]:
    """Bag-of-words vector: texts sharing words get similar vectors (good enough for tests)."""
    v = [0.0] * dim
    for w in re.findall(r"[a-z]{3,}", text.lower()):
        v[int(hashlib.md5(w[:6].encode()).hexdigest(), 16) % dim] += 1
    return v


def storyboard_answer(prompt: str) -> dict:
    """A valid storyboard that uses the labels found in the prompt."""
    s = sorted(set(re.findall(r"\[(S\d+)\]", prompt)), key=lambda x: int(x[1:])) or ["S1"]
    p = sorted(set(re.findall(r"\[(P\d+)\]", prompt)), key=lambda x: int(x[1:]))
    target = int(re.search(r"about (\d+) spoken words", prompt).group(1))
    per = max(12, target // 8)   # ~8 scenes with narration: close to the target
    say = ("A join combines two tables on a shared key so you see related data together "
           "and avoid copying values by hand between sheets. ") * 20
    words = say.split()
    scenes = [{"visual": "title", "heading": "Joining tables", "narration": " ".join(words[:per])}]
    if p:
        scenes.append({"visual": "slide", "slide": p[0], "narration": " ".join(words[per:2 * per]), "sources": s[:1]})
    scenes += [
        {"visual": "concept", "heading": "Join key", "items": ["A column both tables share"],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "comparison", "heading": "Inner vs left", "columns": [
            {"heading": "Inner", "items": ["Only matches"]}, {"heading": "Left", "items": ["All of the first table"]}],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "table", "heading": "Which join", "table_header": ["Need", "Join"],
         "table_rows": [["Only matches", "Inner"], ["Keep all rows", "Left"]],
         "narration": " ".join(words[:per]), "animation": "highlight", "cues": ["not in the text", "nor this"],
         "sources": s[:1]},
        {"visual": "steps", "heading": "How to join", "items": ["Pick the key", "Clean the key", "Join"],
         "narration": "Joining takes three steps. Pick the key column that both tables share. Clean the key "
                      "values so they match. Run the join and check the row counts.",
         "animation": "build", "cues": ["Pick the key column", "Clean the key values", "Run the join and"],
         "sources": s[:1]},
        {"visual": "cycle", "heading": "Check and repeat", "items": ["Join", "Count rows", "Fix keys"],
         "narration": "Joining is a loop. Join the tables. Count the rows before and after. Fix the keys that "
                      "did not match, then join again.",
         "animation": "highlight", "cues": ["Join the tables", "Count the rows", "Fix the keys"],
         "sources": s[:1]},
        {"visual": "cards", "heading": "Three things make a join work", "subheading": "Get these right first",
         "elements": [{"title": "A shared key", "detail": "Both tables need the **same column**", "icon": "key-round"},
                      {"title": "Clean values", "detail": "Trim spaces and match the data types", "icon": "not-an-icon"},
                      {"title": "A check", "detail": "Count rows before and after", "icon": ""}],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "example", "heading": "Customer orders", "items": ["Situation: orders and customers tables",
                                                                      "What to do: left join on customer ID",
                                                                      "Result: every order with its customer"],
         "narration": " ".join(words[:per]), "sources": s[:1]},
        {"visual": "question", "heading": "Which join keeps every row of the first table?",
         "options": ["Inner join", "Left join", "No join"], "correct": 2,
         "narration": "Which join keeps every row of the first table? Pause and think.",
         "answer_narration": "A left join keeps every row of the first table; an inner join keeps only matches.",
         "sources": s[:1]},
        {"visual": "bullets", "heading": "Key takeaways", "items": ["Use a clean key", "Check the result"],
         "narration": " ".join(words[:per]), "sources": s[-1:]},
    ]
    sb = {"title": "Joining tables", "goal": "Join two tables on a shared key.",
          "objectives": ["Choose inner or left join", "Check a join result"], "scenes": scenes}
    if "strict instructional designer" in prompt:   # the self-review pass
        sb["changes"] = ["Made the example concrete"]
    return sb


def vision_answer(text: str, state: dict) -> str:
    """Answers about an image: the picture check, the slide review, else a slide description."""
    if "You check a picture" in text:
        if state.get("reject_images"):
            return "APPROPRIATE: yes\nMATCHES: no\nCLEAN: no\nWELL_FORMED: yes\nPROBLEM: text in the picture"
        return "APPROPRIATE: yes\nMATCHES: yes\nCLEAN: yes\nWELL_FORMED: yes\nPROBLEM: none"
    if "You review one slide" in text:
        bad = state.get("bad_slides")
        return ("READABLE: yes\nLAYOUT: no\nPICTURE: none\nAPPROPRIATE: yes\nMATCHES_VOICE: yes\nPROBLEM: cards overlap"
                if bad else "READABLE: yes\nLAYOUT: yes\nPICTURE: none\nAPPROPRIATE: yes\nMATCHES_VOICE: yes\nPROBLEM: none")
    return "TYPE: list\nSTRUCTURE: a -> b\nKEY POINTS: test"


def _text_parts(content) -> str:
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else ""


def qa_answer(text: str) -> str | None:
    """Canned answers for the question-answering prompts (answer.py, retrieval.py, qa_eval.py)."""
    if "You rank excerpts" in text:   # relevant = shares a content word with the question
        q = set(re.findall(r"[a-z]{4,}", re.search(r"Question: (.*)", text).group(1).lower())) - {"what", "does", "with"}
        cands = re.findall(r"\[C(\d+)\] (.*)", text)
        return json.dumps({"scores": [{"id": f"C{n}", "score": 3 if q & set(re.findall(r"[a-z]{4,}", line.lower())) else 0,
                                       "why": "test"} for n, line in cands]})
    if "You answer a learner's question using ONLY" in text:
        found = re.findall(r"\[(S\d+)\][^\n]*\n(?:On screen:[^\n]*\n)?([^\n]+)", text)
        if not found:
            return json.dumps({"not_in_material": True, "sentences": [], "not_covered": "all"})
        label, said = found[0]
        words = said.split(":", 1)[-1].split()
        sentences = [{"text": "Joining data puts two tables together using a shared key column.", "sources": [label],
                      "evidence": " ".join(words[:7])}]
        if len(found) > 1:
            sentences.append({"text": "Clean the key first, and 73 percent of teams forget this.", "sources": [found[1][0], "S99"],
                              "evidence": "this quote is not in the excerpt at all"})
        return json.dumps({"not_in_material": False, "sentences": sentences, "not_covered": ""})
    if "You rewrite an answer" in text:
        data = json.loads(text.split("(JSON):", 1)[1].strip().split("\n\nAnswer with one JSON")[0])
        return json.dumps({"not_in_material": False, "sentences": [{**s, "text": "In other words: " + s["text"]}
                                                                    for s in data["sentences"]]})
    if "You check every claim" in text:
        sents = re.findall(r"SENTENCE (\d+): ([^\n]*)", text)
        return json.dumps({"claims": [{"sentence": int(n), "claim": t, "verdict": "unsupported" if "73 percent" in t else "supported",
                                       "reason": "not in the excerpt" if "73 percent" in t else ""} for n, t in sents]})
    if "You grade an answer" in text:
        ans = text.split("Answer:", 1)[1].split("Excerpts found:", 1)[0].lower()
        facts = re.findall(r"^- (.+)$", text.split("Key facts:", 1)[1], re.M) if "Key facts:" in text else []
        return json.dumps({"relevance": 5, "completeness": 4, "correctness": 4, "clarity": 5,
                           "facts": [{"fact": f, "found": "yes" if f.split()[0].lower() in ans else "no"} for f in facts],
                           "comment": "test"})
    if "You write test questions" in text and "From\nthe list of sources" in text:
        titles = re.findall(r"^- ([^:]+):", text, re.M)
        return json.dumps({"general": [{"question": f"What is {t.lower()} about?", "source": t} for t in titles],
                           "out_of_scope": ["What is coffee?", "How do I bake sourdough bread?", "Who won the 1998 World Cup?"]})
    if "You write test questions" in text:
        ex = re.findall(r"\[E(\d+)\][^\n]*\n([^\n]+)", text)
        return json.dumps({"items": [{"excerpt": f"E{n}", "question": f"What should I know about {said.split()[2] if len(said.split()) > 2 else 'it'}?",
                                      "expected_answer": said, "key_facts": [said.split(":", 1)[-1].split()[1]]} for n, said in ex]})
    return None


def answer(text: str) -> str:
    """Canned model answer for the kind of request found in the prompt."""
    qa = qa_answer(text)
    if qa is not None:
        return qa
    if "You turn a learner's request" in text:
        ids = re.findall(r"- id (\S+):", text)
        return json.dumps({"format": "video", "topic": "joining data", "recordings": ids[:1] + ["made-up-id"],
                           "minutes": 1, "level": "wizard", "wishes": "use a comparison table"})
    if "You design a short video course" in text:
        refs = re.findall(r"Recording id (\S+):", text)
        ch = lambda t, a, b: {"title": t, "purpose": f"Learn {t.lower()}.", "key_points": ["keys", "joins"],  # noqa: E731
                              "weight": 2, "parts": [{"recording": refs[0], "start": a, "end": b},
                                                     {"recording": "nope", "start": "00:00", "end": "09:00"}]}
        return json.dumps({"coverage": "partial", "missing": "pivot tables", "title": "Joining data well",
                           "chapters": [ch("Join basics", "00:00", "01:50"), ch("Checking joins", "02:00", "03:50"),
                                        {"title": "", "parts": []}]})
    if "You write the frame of a video course" in text:
        q = {"question": "What does a left join keep?", "options": ["Only matches", "All rows of the first table",
                                                                    "Nothing"], "correct": 2,
             "answer_narration": "A left join keeps every row of the first table."}
        return json.dumps({"goal": "Join tables safely.", "objectives": ["Pick a join", "Clean keys", "Check results"],
                           "intro_narration": "Joining tables is everyday work. Here is how to do it right.",
                           "quiz": [q, {**q, "correct": 7}, q, q],
                           "summary_items": ["Clean the key", "Pick the join", "Check the rows"],
                           "summary_narration": "You now know how to join tables and check the result. " * 3})
    if "You map the topics" in text:
        ref = re.search(r"Recording id: (\S+)", text).group(1)
        return json.dumps({"summary": "How to join tables.", "topics": [
            {"title": "Join keys", "covers": "What a key is.", "start": "00:00", "end": "01:50", "depth": "solid",
             "level": "beginner", "minutes": 9, "visual_idea": "concept card",
             "request": f'Make a 2-minute "video" on join keys from recording {ref}'},
            {"title": "Checking a join", "covers": "Count rows.", "start": "03:00", "end": "03:50",
             "depth": "huge", "level": "wizard", "minutes": 1, "visual_idea": "steps",
             "request": f"Make a 1-minute video on checking joins from recording {ref}"}]})
    if "You plan short video lessons across" in text:
        ids = re.findall(r"Recording id (\S+):", text)
        item = {"title": "Joins end to end", "covers": "Both.", "why_here": "Basics first.",
                "recordings": ids + ["nope"], "minutes": 3, "level": "beginner", "request": "Make a video on joins"}
        return json.dumps({"shared_topics": [item], "learning_path": [item, {**item, "title": "Advanced"}]})
    if "You review a narrated video lesson" in text or "You review a written lesson" in text:
        return json.dumps({"faithful": 4, "level_fit": 5, "clarity": 4, "issues": []})
    if "spoken words" in text:
        return "```json\n" + json.dumps(storyboard_answer(text)) + "\n```"
    if "You turn the narration of a training video into a natural conversation" in text:
        scenes = re.findall(r"SCENE (\d+)[^\n]*\nNARRATION: ([^\n]*)", text)
        out = []
        for k, narration in scenes:
            lines = [{"speaker": "host", "text": "So what happens here?"}, {"speaker": "expert", "text": narration}]
            if k == "2":   # a made-up fact: must be caught, the scene keeps its narration
                lines.append({"speaker": "host", "text": "And 73 percent of teams do that wrong."})
            out.append({"scene": int(k), "lines": lines})
        return json.dumps({"scenes": out})
    if "self-study lesson" in text:
        labels = sorted(set(re.findall(r"\[(S\d+)\]", text)), key=lambda x: int(x[1:])) or ["S1"]
        target = int(re.search(r"about (\d+) words in total", text).group(1))
        body = " ".join(["Tables are joined on a shared key."] * 200)
        return json.dumps({"title": "Joining tables", "goal": "Understand joins.",
                           "overview": "A join combines two tables on a **shared key**; clean the key first.",
                           "sections": [{"heading": "Joins", "body": " ".join(body.split()[:target - 60]),
                                         "points": ["**Inner join** keeps only matches", "**Left join** keeps all rows"],
                                         "table": {"columns": ["Join", "Keeps"],
                                                   "rows": [["Inner", "matching rows"], ["Left", "every row of the first table"]]},
                                         "example": "Customer ID joins orders to customers.",
                                         "remember": "Count rows before and after the join; 4512 rows were lost.",
                                         "sources": labels[:1]},
                                        {"heading": "Checks <b>", "body": "Count the rows.", "sources": labels[:1]}],
                           "takeaways": ["Key", "Clean", "Check"], "summary": "Clean, join, check.",
                           "questions": [{"question": "What does a left join keep?", "answer": "Every row of the first table."}],
                           "glossary": [{"term": "Key", "meaning": "The shared column."}]})
    if "JSON" in text:
        return json.dumps({"term": "VTT file", "definition": "Subtitles with timestamps."})
    return "OK"


class Gateway:
    """Runs the fake gateway on a free local port in a background thread."""

    def __init__(self) -> None:
        self.state = {"openai_route": True, "anthropic_route": True, "reject_response_format": False, "calls": []}
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # keep test output clean
                pass

            def _send(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def _authorized(self) -> bool:
                ok = self.headers.get("x-bf-vk") == KEY or self.headers.get("Authorization") == f"Bearer {KEY}"
                if not ok:
                    self._send({"error": {"message": "invalid virtual key"}}, 401)
                return ok

            def do_GET(self):
                if self._authorized():
                    if self.path == "/v1/models" and gateway.state["openai_route"]:
                        return self._send({"data": [{"id": m} for m in MODELS]})
                    self._send({"error": "not found"}, 404)

            def do_POST(self):
                if not self._authorized():
                    return
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                gateway.state["calls"].append(self.path)
                gateway.state.setdefault("models", []).append(body.get("model"))
                if body.get("model") in gateway.state.get("denied", ()):   # like Bedrock without model access
                    return self._send({"error": {"message": "AccessDeniedException: no access to this model"}}, 403)
                if body.get("model") not in MODELS:
                    return self._send({"error": {"message": "unknown model"}}, 404)
                if self.path == "/v1/images/generations":   # a 2x2 PNG
                    png = ("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAFklEQVR4nGP8z8DAwMDAxMDAwMDAAAANHQEDasKb6QAAAABJRU5ErkJggg==")
                    return self._send({"data": [{"b64_json": png}]})
                if self.path == "/v1/embeddings":
                    texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
                    texts = texts[:1]   # like Titan: one text per request
                    return self._send({"data": [{"index": 0, "embedding": vector(texts[0])}],
                                       "usage": {"prompt_tokens": len(texts[0]) // 4}})
                if self.path == "/anthropic/v1/messages" and gateway.state["anthropic_route"]:
                    if gateway.state.get("reject_cache") and "cache_control" in json.dumps(body):
                        return self._send({"error": {"message": "cache_control is not supported"}}, 400)
                    plain = lambda c: c if isinstance(c, str) else _text_parts(c)   # noqa: E731
                    text = plain(body.get("system", "")) + " ".join(plain(m["content"]) for m in body["messages"])
                    last = body["messages"][-1]["content"]
                    has_image = isinstance(last, list) and any(p.get("type") == "image" for p in last)
                    out = vision_answer(_text_parts(last), gateway.state) if has_image else answer(text)
                    usage = {"input_tokens": 100, "output_tokens": len(out) // 4}
                    marked = [b for part in [body.get("system", "")] + [m["content"] for m in body["messages"]]
                              if isinstance(part, list) for b in part if b.get("cache_control")]
                    if marked:   # like Claude: the first call writes the cache, the next ones read it
                        key = hash(json.dumps(marked, sort_keys=True))
                        n = sum(len(b["text"]) // 4 for b in marked)
                        seen = gateway.state.setdefault("cached", set())
                        usage["cache_read_input_tokens" if key in seen else "cache_creation_input_tokens"] = n
                        seen.add(key)
                    return self._send({"content": [{"type": "text", "text": out}], "stop_reason": "end_turn",
                                       "usage": usage})
                if self.path == "/v1/chat/completions" and gateway.state["openai_route"]:
                    if gateway.state["reject_response_format"] and "response_format" in body:
                        return self._send({"error": {"message": "response_format not supported"}}, 400)
                    last = body["messages"][-1]["content"]
                    if isinstance(last, list):
                        out = vision_answer(_text_parts(last), gateway.state)
                    else:
                        out = answer(" ".join(m["content"] for m in body["messages"] if isinstance(m["content"], str)))
                    return self._send({"choices": [{"message": {"role": "assistant", "content": out},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 100, "completion_tokens": len(out) // 4}})
                self._send({"error": "route not found"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True   # never wait for a hanging request at the end
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop the server without ever blocking the test run (Windows can't interrupt a wait)."""
        stopper = threading.Thread(target=self.server.shutdown, daemon=True)
        stopper.start()
        stopper.join(timeout=5)
        self.server.server_close()
