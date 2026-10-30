"""
describe_slides.py - optional step: a local vision model describes what each slide SHOWS.

Why: OCR reads the words but loses the structure. On these slides the structure is the
knowledge: steps of a process from left to right, an architecture with layers and arrows,
a table, a grid of categories. A vision model can say "Business problem -> Establish data
governance -> Collect data -> ..." where OCR returns the same words in a random order.

The description is stored as slide["visual"]. Search and lessons use it automatically
(store.all_segments adds it to the slide text). Later, the slide video can redraw these
structures as clean diagrams from it.

Cost: with the gateway (VISION_BACKEND=bifrost, Claude) a few seconds and about $0.01 per
slide. Locally (VISION_BACKEND=ollama) roughly 1-3 minutes per slide with a 3B model; then
run it over lunch. Either way it runs once per slide and the result is kept.

Usage:
  uv run describe_slides.py --video <video_id> --limit 3   # try a few slides first
  uv run describe_slides.py                                # all slides not described yet
  uv run describe_slides.py --force                        # redo all
Then rebuild the search index:  uv run index.py build
"""
from __future__ import annotations

import argparse
import logging
import sys

import config
import llm
import store

log = logging.getLogger("describe_slides")

PROMPT = """This is one frame from a recorded internal training: either a presentation slide, or a
screen recording of someone working in a program (Excel, Alteryx, a browser, another tool).
Describe what it shows so that someone who cannot see it understands it.

Answer in exactly this format:
TYPE: one of process, architecture, list, table, comparison, chart, text, title, decorative, screen
STRUCTURE: the content in its real order and relations. For a process, list the steps in
order with "->". For an architecture, name each group and what is inside it, then each
arrow as "A -> B" (with its label if it has one). For a table, give each row as
"column: value". For a list, give the items in order. For a chart, say what is compared
and the trend, but do not invent numbers you cannot read.
For a screen of a program: name the program and what is open (file, sheet, window, dialog),
then what the person is doing: the selected cell and the formula in the formula bar, the
menu or ribbon path, the options set in a dialog. In Alteryx: the tools on the canvas in
workflow order with "->", and the configuration shown for the selected tool. Do not copy
data tables cell by cell: name the columns and say what the data is.
KEY POINTS: 1-3 short sentences on what the slide or screen is meant to teach.

Rules: use only what is visible. Ignore logos, page numbers, footers, the presenter's video
and any meeting software around the content. If it is only a title or decoration, say so
and keep STRUCTURE empty."""


def describe_video(video_id: str, force: bool, limit: int | None) -> int:
    """Describe each slide of one recording that has no description yet (all of them with
    force). Saves after every slide. Returns how many slides were sent to the model."""
    slides = store.load_slides(video_id)
    todo = [s for s in slides if s.get("is_slide") and (force or not s.get("visual"))]
    if limit:
        todo = todo[:limit]
    for n, sl in enumerate(todo, 1):
        image = store.video_dir(video_id) / sl["file"]
        log.info("%s [%s/%s] %s", video_id, n, len(todo), sl["file"])
        try:
            sl["visual"] = llm.describe_image(image, PROMPT, step="describe_slide")
        except llm.BudgetExceeded:
            raise
        except llm.LLMError as e:
            log.error("%s: %s", sl["id"], e)
            continue
        store.save_slides(video_id, slides)   # save after each slide: a crash loses one at most
    return len(todo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="Excel id (e.g. 22), video_id or a word from the title (default: all)")
    ap.add_argument("--limit", type=int, help="describe at most N slides per video")
    ap.add_argument("--force", action="store_true", help="redo slides already described")
    args = ap.parse_args()
    config.setup_logging()
    if args.video:
        try:   # Excel id (22), video_id, or a word from the title / file name
            ids = store.resolve_recordings(args.video)
        except ValueError as e:
            log.error("%s", e)
            return 1
    else:
        ids = [v for v in store.processed_video_ids() if store.has_slides(v)]
    if not ids:
        log.error("No video has its slides read yet. First: uv run pipeline.py --mode all")
        return 1
    total = sum(describe_video(v, args.force, args.limit) for v in ids)
    print(f"\n{'video':44} {'screens':>7} {'text':>5} {'described':>9}  note")
    for v in ids:   # what was found per video, and why nothing was sent
        slides = store.load_slides(v)
        text = [s for s in slides if s.get("is_slide")]
        done = sum(bool(s.get("visual")) for s in text)
        note = ("slides not read (audio only, or the slides step failed): uv run slides.py --video " + v
                if not store.has_slides(v) else
                "no screens stayed still for 6 s" if not slides else
                "no readable text on the screens (Tesseract: uv run doctor.py)" if not text else
                "all described" if done == len(text) else f"{len(text) - done} left (run again)")
        title = store.load_meta(v).get("title", v) if store.has_segments(v) else v
        print(f"{title[:44]:44} {len(slides):>7} {len(text):>5} {done:>9}  {note}")
    log.info("Described %s slides. Rebuild the index: uv run index.py build", total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
