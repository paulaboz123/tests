# Video lesson PoC

A person writes what they want to learn (topic, length, level, format) and gets a lesson made
**only from the training material**: a **document**, a **slide deck**, an **animated narrated
video**, a **video with two talking characters**, a **cartoon presenter**, or **audio**. Every
result is checked (truth, safety, layout, sound) and gets a verdict (OK / CHECK / FAIL), a
cost and a run card (what it was made from and with).

## Quick start

| What | Command |
|---|---|
| **Lesson Studio** (click-through page: topic suggestions, choose output / length / level / look / model / voice, estimated cost, result with verdict and cost, report per model and output, how to ask) | `uv run app.py` → opens http://localhost:8765 |
| **Does everything work here?** (one small result of every kind, with the real model) | `uv run smoke.py` (~15 min, ~$1) or `uv run smoke.py --quick` (~4 min) → `data/output/smoke_report.md` |
| One lesson from the terminal | `uv run ask.py "A 2-minute video where two characters talk about the daily scrum"` |
| **The same request with every tool, side by side** (slides and animation per tool, audio per narrator, cost of each) | `uv run benchmark.py "a 1-minute video presentation on data connections"` → `comparison.html` |
| What results cost, side by side | `uv run compare.py` |
| What each budget level gives and costs | `uv run tiers.py` |
| **What is spent and left** (today's spend, ElevenLabs characters left, Remotion); also printed after every run and shown at the top of Lesson Studio | `uv run usage.py` |
| **Pro look in Remotion**: motion-design scenes (glass cards, counters, growing charts, push transitions) and 3D presenters that talk with the voice, wave, point at the screen | `RENDERER=remotion-pro` in `.env` (once: `cd remotion`, `npm.cmd install`), or `benchmark.py --renderers "html,remotion-pro"` |
| **Real-looking presenter** saying our narration (ElevenLabs lip sync: Creatify Aurora) over a video you already have | `uv run ai_clips.py voice "data\output\<video folder>"`, make the clip in ElevenLabs, then `uv run ai_clips.py presenter ... --clip x.mp4` |
| **Real-looking clips** (people, places, nature: Google Veo made in ElevenLabs Image & Video) as intro, cutaways and outro of a video you already have | `uv run ai_clips.py prompts "data\output\<video folder>"`, make the clips in the tool, then `uv run ai_clips.py add ...` |
| Setup check | `uv run doctor.py` |

| Setting | Models | Needs | Quality / speed |
|---|---|---|---|
| **Gateway** (`.env.bifrost.example`) | Claude Sonnet on Amazon Bedrock (EU) via Bifrost; search stays local | URL + key + model id from the platform team | good / fast; ~$0.10-0.20 per short video |
| **Local** (`.env.local.example`) | Llama 3.1 (Meta), Gemma 3 (Google), nomic-embed-text, all in Ollama | nothing | OK / slow (minutes per lesson) |

Cost per video: the material sent to the model grows with the video length
(`MIN_MATERIAL_WORDS`, `MATERIAL_WORDS_PER_MINUTE`), and with `PROMPT_CACHE=yes` (default) the
self-review re-reads it from the cache at ~10% of the price. The run card shows the real cost.

Voice (Kokoro) and video (ffmpeg, browser engine) run on the laptop. No models of Chinese origin.

```
data/input  (videos, transcripts, documents, quizzes, images ...)
   │  inventory.py / documents.py / pipeline.py   read, clean, index (noise and German left out)
   ▼
ask.py / app.py "request"
   1 understand   format, topic, length, level, style (refuses off-rule requests)
   2 material     the most relevant excerpts of the whole knowledge base
   3 write        Claude writes the document / storyboard / conversation from them, then reviews it
   4 make         slides + motion + characters + voice → video, or the document / deck / audio
   5 check        numbers vs sources, AI review, safety, personal data, layout, sound → verdict
   6 run card     input, solution, cost, time
```

---

## Part A: run everything locally (start here)

### A1. One-time setup

1. **Copy the files** from the zip into your project folder (overwrite everything). Keep your
   `data/`, `models/`, `.env`, `pyproject.toml` and `uv.lock`.
2. **Programs** (once): `ffmpeg` and Tesseract OCR on PATH (or their paths in `.env`), Ollama only
   for the local route. Check with `ffmpeg -version`.
3. **Python packages** (one command; already installed ones are skipped):
   ```
   uv add httpx python-dotenv python-docx python-pptx numpy pillow pydantic openpyxl pypdf kokoro-onnx playwright
   uv add --dev pytest ruff
   ```
   Optional: `uv add faster-whisper` (transcripts for recordings without one),
   `uv add pyttsx3` (the Windows voice as a fallback).
4. **Browser for the designer slides** (once): `uv run playwright install chromium`.
   If that download is blocked, the installed Microsoft Edge is used automatically.
5. **Settings:** copy `.env.bifrost.example` (Claude through the gateway) or `.env.local.example`
   (local models), rename the copy to `.env`, fill in the key. Keep an existing `.env`, and add the
   new lines from the example (`STYLE`, `CHARACTERS`, `HOST_VOICE`, `EXPERT_VOICE`, `GSAP`).
6. **Voice** (two files, ~120 MB, once):
   ```
   uv run tts.py --download
   uv run tts.py "Joining data means combining two tables on a shared key."
   ```
   Listen to `data/output/tts_test_kokoro.wav`. If GitHub is blocked, download `kokoro-v1.0.int8.onnx` and
   `voices-v1.0.bin` in the browser from
   https://github.com/thewh1teagle/kokoro-onnx/releases/tag/model-files-v1.0 into `models/`.
7. **Check:**
   ```
   uv run doctor.py
   uv run python -m pytest tests -q
   ```
   `doctor.py`: every line `[OK]` (`[WARN]` for optional parts is fine). The tests run offline
   and free with stand-in models (a few minutes); expect all passed. Run them after every code change.

### A2. Prepare the recordings

7. Put the files in `data/input/` (sub-folders are fine): the slide videos (`.mp4`), their
   transcripts (`.vtt`/`.docx`, same name as the video), and `description.xlsx` (columns `id`, `name`, `description`).
8. ```
   uv run inventory.py
   ```
   A table appears. Every video should have an id, a transcript and `OK`. Open
   `data/manifest.csv` and set `include` to `yes` for the videos to use (screen demos without slides: `no`).
9. **Slide area**, for one video (the `video_id` is in the manifest, e.g. `13-introduction-to-joining-data-d25c`):
   ```
   uv run pipeline.py --mode easy
   uv run slides.py --video <video_id> --preview
   ```
   Open `data/processed/<video_id>/crop_preview.jpg`. The red box is what is read as the slide.
   - presenter or Teams panel **next to** the slide → set `SLIDE_CROP` in `.env`, then
     `uv run pipeline.py --mode easy --force`
   - presenter camera **on top of** the slide → set `SLIDE_MASK` (blue box); it is hidden in the video
   - look at `data/processed/<video_id>/slides/`: each slide should appear once, complete
10. **Search check:**
    ```
    uv run index.py search "how to join two tables" -k 5
    ```
    The right recording should be on top.

### A3. Make lessons and videos

11. **Plan a video first** (cheap check), then make it:
    ```
    uv run make_video.py --recording 13 --minutes 2 --plan-only
    ```
    Read `data/output/<folder>/script.md`. If it's fine:
    ```
    uv run make_video.py --rerender "data/output/<folder>"
    ```
    Watch `lesson_video.mp4`, listen to `lesson_audio.mp3`.
12. **Topic across all recordings, in one go:**
    ```
    uv run make_video.py "how to join two tables" --minutes 2
    ```
13. **Text lesson:**
    ```
    uv run lesson.py "how to join two tables" --minutes 5 --level beginner
    ```

### Lesson Studio: the page to show others

`uv run app.py` opens http://localhost:8765 (only this laptop can open it; stop with Ctrl+C).

| Tab | What it does |
|---|---|
| **Make a lesson** | 1 topic (type it or click a suggestion from the material), 2 output (document, slides, video with narrator, two characters, cartoon presenter, audio, or Auto = from your words), 3 options: length, level, look, **writing model**, **voice**, AI check. Shows what you will get and the **estimated cost** before you start, the progress while it is made, then the result: player / document / slides, verdict, review scores, files, what it was made with, cost per step, sources |
| **Results & costs** | every result; spent today (vs the daily limit) and in total; **report per writing model and per output**: count, average cost and time, how true to the sources, how many usable; CSV download |
| **How to ask** | what to write to get each output (with *Try it*), what a good request has, what is refused, the verdicts, model and voice prices |

- Clicked options win over the words of the request. Auto = taken from the words.
- Models on the page: CHAT_MODEL (default) plus `MODEL_CHOICES` / `LOCAL_MODEL` in `.env`. The choice is for that one lesson only.
- Voices on the page: Kokoro, plus ElevenLabs when its key is set up (and the Windows voice).
- Topic suggestions: the source titles; click *Find the topics in the material* once (about $0.02 per source) for concrete topics.
- The estimate is the median cost per minute of earlier results of the same output and model; before there are any, a rough figure from the model's price.

### Output types: the request decides

`ask.py` reads what the person wants and makes one of these:

| Ask for... | You get | Made by |
|---|---|---|
| a document, lecture notes, an explanation to read | `lesson.html` (open in a browser, Ctrl+P = PDF) + `lesson.md`: short answer first, sections with **bold** key terms, bullets, tables, examples, "Remember" boxes, takeaways, summary, glossary, self-check questions | `lesson.py` |
| a simple presentation, slides to present myself | `slides.pdf`, `slides.pptx` (with `uv add python-pptx`), `notes.md` speaker notes | `deck.py` |
| an animated presentation with a narrator, a video | `lesson_video.mp4` + `lesson_audio.mp3` (style clean / illustrated / presenter) | `make_video.py`, 4+ min: `course.py` |
| two people talking, a conversation, an interview | the same video, but a host asks and an expert explains, two voices, two people beside the slide | `make_video.py --style dialogue` |
| something to listen to | `lesson_audio.mp3` | as video |

```
uv run ask.py "Napisz mi dokument o rolach w Scrumie, z tabelą i podsumowaniem"
uv run ask.py "A simple 6-slide presentation on sprint planning that I can present myself"
uv run ask.py "A 3-minute video where two people talk about the daily scrum"
```
Voices of the two people: `HOST_VOICE`, `EXPERT_VOICE` in `.env` (Kokoro voices, `uv run tts.py --voices`).

Characters (presenter and dialogue styles): `CHARACTERS=cartoon` (default) = a bear (expert / presenter)
and a fox (host) whose mouths move with their own voice, who blink, nod and point at the screen;
`CHARACTERS=people` = the drawn people. A request that asks for a cartoon / fairy tale / animals gets
the cartoon presenter. Per video: `--characters cartoon|people`. The characters are drawn in code
(`cartoon.py`), so they look the same in every scene and nothing random can appear.

Every result has a **run card** (`run_card.md` / `run_card.json` in its folder): the request, which
sources were used, which models and tools made it (writing, review, search, voice, slides, characters,
pictures), the cost of every model call by step and model, the time, and the quality verdict.
To compare two setups (free vs paid, Sonnet vs Opus, two voices): run the same request twice with one
setting changed in `.env`, then `uv run compare.py --topic "<topic>"`.

### Answers to questions, and how good they are (answer.py, qa_eval.py)

A question in, a text answer out, from the knowledge base only, with the source of every sentence:
```
uv run answer.py "What is coffee?"
uv run answer.py "Who orders the Product Backlog?" --length medium
```
1. **Find** (`retrieval.py`): a meaning search (embeddings) and a word search (BM25: exact names,
   terms, acronyms, numbers), fused and weighted by `SOURCE_PRIORITY`; then the model reads the ~30
   best candidates and keeps those that really answer the question (`RETRIEVAL=hybrid+rerank`).
2. **Write**: Claude answers from those excerpts only, most important first, in its own words;
   every sentence cites its excerpts and gives an evidence quote. Not covered = "not covered", no answer.
3. **Check in code**: real sources, evidence quote found in the source, numbers in the sources, share
   copied word for word (above `COPY_LIMIT` rewritten once).
4. **answer.md**: the answer with [1] [2], and per source the recording or document, the place
   (`03:10-04:05 · slide "Three roles"`, page, section) and the evidence quote. `answer.json` has everything.

**Measuring quality** on a test set (`eval/qa_set.csv`, edit in Excel; `eval/qa_set_example.csv` shows the columns):
```
uv run qa_eval.py --draft 20     # a first set from your material: specific, general and out-of-scope questions
uv run qa_eval.py                # answer and score every question -> data/output/<time>-qa-eval/qa_report.html
uv run qa_eval.py --compare      # the same set with meaning / hybrid / hybrid+rerank: before and after
uv run qa_eval.py --set "Video_Accuracy_Test_Plan.xlsx"   # the client's test plan as it is
```
The client's plan is read as it is (Test ID, Category, Prompt, Source Video (ID+name), Expected fact;
rows with the same Test ID = one question with several videos and facts; "4 - Tools @SE" = catalog
number 4 or that name). The results also go into a copy of the plan, `<name>_results.xlsx`: the
answer, its sources (video · time), per row whether the expected fact is in the answer and the
expected video was found, claims not in the sources, PASS / CHECK and why.
| Area | Metrics |
|---|---|
| Finding the information | retrieval hit rate, MRR, answer cites the expected source |
| No hallucination | faithfulness (claim by claim, judge model), hallucination rate, evidence found word for word, numbers not in the sources, invented or missing sources (code) |
| Correct and complete | key-fact recall, correctness vs expected answer, relevance, completeness, clarity (judge model, 1-5) |
| Knows what it doesn't know | declines out-of-scope questions, false refusals |
| Own words | share copied word for word |
| Cost and time | per answer |

The report explains how each metric is measured and why, lists the failed questions with the
reason (e.g. the unsupported claim), every answer with its sources, and the previous run to compare.
Set `JUDGE_MODEL` to another model than the writer (e.g. Opus when Sonnet writes) so answers are
not graded by the model that wrote them; `RERANK_MODEL` can be Haiku (cheap).

### Comparing tools: one request, every tool at its best (benchmark.py)

One request, one script (written once, so the content is the same everywhere), every narrator and
every video tool, as slides and as an animation:
```
uv run benchmark.py "a 1-minute video presentation on data connections" --minutes 1
uv run benchmark.py --from "data/output/<an existing video folder>"            # reuse a script
uv run benchmark.py --report "data/output/<comparison folder>"                 # rebuild the page
```
Default: narrators Kokoro (local, free) and ElevenLabs; tools "local, open source" (designer slides
with motion, drawn characters) and Remotion Pro (motion design, 3D presenters); types slides and
animation = 8 videos. Open `comparison.html` in the comparison folder: a table per type (tool x
narrator, each video with its length, voice cost, time and checks), the audio of each narrator
(the same narration, other voices), the AI video section (ElevenLabs Veo clips or a lip-synced
presenter, once added with `ai_clips.py` and the page rebuilt with `--report`) and the cost of every
version. Rate them in `benchmark.csv`. Each video uses its narrator's own timing, so picture and
voice always fit; versions with the same narrator are equally long. Other combinations:
`--voices kokoro,elevenlabs,system --renderers html,remotion,remotion-ai,remotion-pro --styles
clean,illustrated,presenter,dialogue`. Versions that can't run (no key, no Node.js) are listed with
the reason; a version that breaks is listed as failed and the others still run. Each narrator's voice
is made once and reused by every tool (no double cloud cost).

**Compare configurations, not only tools.** A narrator is `engine[:voice][@model][+plain]`, and the
Remotion Pro presenters are a list too:
```
uv run benchmark.py --from "data/output/<folder>" --styles presenter --renderers remotion-pro ^
  --voices kokoro:af_heart,kokoro:bf_emma,elevenlabs:lily,elevenlabs:lily+plain,elevenlabs:george@flash ^
  --presenters robot,cat,person-3d,person-2d:micah,person-2d:notionists
```
| Part | Options |
|---|---|
| `kokoro:<voice>` | `af_heart`, `af_bella`, `af_nicole`, `bf_emma`, `am_michael`, `bm_george` … (`uv run tts.py --voices`) |
| `elevenlabs:<voice>` | `lily`, `matilda`, `jessica`, `alice`, `sarah`, `laura`, `george`, `brian`, `daniel`, `will`, `eric` … or a voice id |
| `@<model>` (ElevenLabs) | `v2` multilingual (best for narration, default), `flash`, `turbo` (half the credits), `v3` (most expressive; no context) |
| `+plain` (ElevenLabs) | the old way: every sentence spoken on its own (to hear what the context does) |
| `--presenters` | `robot`, `cat`, `person-3d`, `person-2d`, `person-2d:micah` / `:avataaars` / `:notionists`, `none` |

| `--audio-only` | only the narration of each narrator, no video: 8 voices in a few minutes |
| `--vary KEY=a,b` | any `.env` setting in several versions (repeatable; every combination): `THEME_ACCENT=#0b6e79,#7c3aed`, `GSAP=yes,no`, `VIDEO_WIDTH=1280,1920`, `REMOTION_FPS=25,30`, `ELEVENLABS_STABILITY=0.4,0.6` |
| `--characters` | the drawn bear and fox of the built-in engines (`cartoon`, `people`); default none, so their animation versions are left out |

The ElevenLabs characters are counted per narrator (5 narrators of a 1-minute lesson ≈ 5,000).
A version where Remotion failed is shown as "Not this tool" (it was made with the designer slides),
and `remotion_error.txt` in its folder has the full message.
Each variant's `run_card.json` keeps the exact settings; `comparison.html` shows them next to the video.

**ElevenLabs sound.** A lesson is spoken in short parts so the slides follow the voice. Each part is
sent with what comes before and after it (request stitching: the ids of the parts just spoken, else
their text, and the next sentence), with steady settings (`ELEVENLABS_STABILITY=0.6`, speaker boost)
and a fixed seed, and the silence at its ends is cut: one flowing narration instead of separate
sentences. The audio comes at 44.1 kHz as in ElevenLabs' own web app (`ELEVENLABS_FORMAT=mp3_44100_128`,
every plan), and the whole narration gets one steady loudness (one gain, no level control that
"pumps"). The cost on the run card is what ElevenLabs says it billed (its `character-cost`).

**Remotion on Windows.** It is started with `node` and Remotion's own script (not `npx.cmd`), with full
paths; missing packages are installed automatically (`npm install`, once). If a render fails, the
full message is in `remotion_error.txt` in the video's folder. Faster renders: `REMOTION_FPS=25`
(about 17% faster), `REMOTION_CONCURRENCY` up to the number of cores, `PRO_PRESENTER=person-2d`.

| Option | Setup | Cost |
|---|---|---|
| Kokoro voice | (default) | free, local |
| ElevenLabs voice | `ELEVENLABS_API_KEY` in `.env` (free trial: ~10 min a month; `uv run tts.py --engine elevenlabs --voices`) | `ELEVENLABS_USD_PER_1K_CHARS` (paid-plan rate) per 1,000 characters |
| Built-in video engine | (default) | free |
| Remotion | Node.js 18+, then `cd remotion` + `npm install` (once) | free for evaluation; company licence for production |

Test one voice alone: `uv run tts.py --engine elevenlabs "A short test sentence."` (voice: `ELEVENLABS_VOICE`).

### Budget tiers: what each level gives and costs

`uv run tiers.py` → `data/output/tiers.md` / `tiers.csv`: monthly cost and cost per video for each
budget tier (0 free at its best, 1 basic paid, 2a realistic AI presenter, 2b own branded cartoon
character, 2c manual business cartoons, 3 enterprise) at 20 / 100 / 500 videos a month. Prices and
volumes are in `tiers.json` (with sources and the date they were checked; edit to try others).
Once real videos exist, their run cards replace the assumed AI cost and characters per minute.

**AI presenter from another tool** (Synthesia, D-ID, Colossyan ... free plans have no API):
```
uv run avatar.py export "data/output/<video folder>"            # avatar_script.txt to paste into the tool
uv run avatar.py make "data/output/<video folder>" --clip avatar.mp4 --tool synthesia
```
→ `lesson_video_avatar.mp4`: our slides timed to the clip, the presenter beside them (a plain
green background is cut out), the presenter's own voice. Its run card names the tool.
Free plans give about 1 minute per clip (HeyGen free: 3 videos of 1 minute, with a watermark):
`avatar_script.txt` shows the minutes after each scene; paste only the scenes that fit and add
`--scenes N` to `make`, so the slides are timed to those scenes only.

**Real-looking clips** (Google Veo). Our engines and Remotion draw with code, so they
never look like filmed reality, however good the animation. Real-looking people, places, waves or
falling leaves need a generative video model. Bedrock EU has none, and the free plans of
ElevenLabs, Higgsfield and Runway have no video; ElevenLabs Starter ($6/month, 30,000 credits,
needs approval) has Image & Video with Google Veo. Clips are 8 s, cannot show exact text, and
the same person looks different in each clip, so they go around our lesson, not instead of it:
```
uv run ai_clips.py prompts "data\output\<video folder>"          # clip_prompts.txt: 3 generic prompts to paste
uv run ai_clips.py add "data\output\<video folder>" --clip intro.mp4 --clip middle.mp4 --clip end.mp4 --tool elevenlabs --model veo-3.1-fast --cost-usd 1.50
```
→ `lesson_video_clips.mp4` (+ `.srt`): the first clip before the lesson, the last after it, the
others in a pause after a full sentence (nobody speaks over them). `--into <file>.mp4`
puts them around another video of the folder. Only the title and scene headings go to Claude for the
prompts, and prompts with names, numbers or personal data are replaced by built-in ones: they are
pasted into an outside tool. Never Sora (no OpenAI in this project), Kling, Seedance, Hailuo, Wan or PixVerse (refused). The run
card shows the tool, model, prompts and cost (`--cost-usd`: the credits used, takes thrown away
included, as dollars of the plan); it is not counted in the daily Bedrock limit.

**A real-looking presenter who says our narration** (lip sync, e.g. Creatify Aurora in ElevenLabs):
```
uv run ai_clips.py voice "data\output\<video folder>" --seconds 8       # presenter_voice.mp3, ends after a sentence
uv run ai_clips.py presenter "data\output\<video folder>" --clip talking.mp4 --cost-usd 0.40
```
In ElevenLabs Image & Video: a lip-sync model, a portrait (AI-made or licensed, never a colleague's
photo without consent) and `presenter_voice.mp3`. → `lesson_video_presenter.mp4`: the presenter in a
rounded window over the lesson for those seconds, lips in time with the lesson's own voice.

**ElevenLabs video from the code** (`eleven_video.py`): the same models by API, but only on the Pro
plan ($99/month); lower plans get "paid_plan_required" (said in plain words), and the free plan has
no video at all, not even in the web app. `uv run eleven_video.py check` tells what the key can do
without using credits; `clip <folder> --prompt 1` makes a Veo clip from `clip_prompts.txt` and
`presenter <folder> --image portrait.jpg` a talking presenter; the credits used are read from the
account and become the cost on the run card.

**Remotion Pro** (`RENDERER=remotion-pro`): the same storyboard, voice and timing drawn as a
motion-design video (`remotion/src/pro`): a living dark backdrop, glass cards that flip in as the voice
reaches them, flows that draw themselves, counters and bar charts that grow, a quiz that reveals the
answer, push transitions between scenes, and presenters that talk in time with the voice, wave, turn
and point at the screen and are happy about the right answer: none in slides (clean, illustrated),
one in the presenter style, two in conversations. Choose who in `.env`:

| `PRO_PRESENTER` | Who | Licence (`remotion/assets/README.md`) | Render time* |
|---|---|---|---|
| `robot` (default) | 3D glossy toy robot | CC0 | ~5x |
| `cat` | 3D toon cat | CC BY 4.0: credit "Toon Cat FREE by Omabuarts Studio" in the video description | ~5x |
| `person-3d` | 3D woman (presenter, expert) and man (host) in suits | CC0 | ~5x |
| `person-2d` | 2D illustrated person in a round portrait; `PRO_AVATAR=micah` (colourful), `avataaars` (cartoon) or `notionists` (line art, with hand gestures) | CC BY 4.0 / free use / CC0 | ~1.2x |
| `none` | the scenes only | | 1x |

\* compared with no presenter, measured here without a graphics chip. **Faster videos:** `person-2d`;
`VIDEO_WIDTH=1280` (720p, about 1.4x faster); `REMOTION_CONCURRENCY=4` (all cores of a 4-core laptop;
more memory). 3D is drawn by the browser the way it chooses (fast, uses the graphics chip if there is
one); if that fails the render is retried with `swangle` (always works, several times slower).
After updating: `cd remotion`, `npm.cmd install` (new: the 2D drawing packages). Code-drawn, so
still an animation, not a real person; for real people use the presenter route above.

Every result is checked automatically (`quality.json`, verdict OK / CHECK / FAIL): numbers must be in
the sources, a model review scores faithfulness (skip with `--no-review`), inappropriate words and
personal data fail. Requests that try to steer the AI, ask for offensive content or for facts from
outside the material are refused before anything is made.
The conversation may not add facts: every scene is checked (numbers, length, wording) and a
scene that fails is read by one voice (a warning in the log and `script.md` shows who says what).

Motion: slides animate with GSAP (bundled, free licence; `GSAP=no` in `.env` = the simpler
built-in motion). A professional animated presenter (Lottie, e.g. from LottieFiles) can replace
the drawn one: `uv run tools/add_lottie.py --talk talk.json --point point.json --licence "Lottie Simple License"`
(checked first: no internet pictures, no text, size); `--remove` goes back to the drawing.

To fix a sentence or on-screen word in a video: edit `storyboard.json` in the output folder
(`narration`, `heading`, `items`) and run `--rerender`. No model call.

---

## Part B: switch to the gateway (Claude on Bedrock)

You need the 3 values from the platform team's e-mail: the **address** (`ANTHROPIC_BASE_URL`),
the **virtual key**, and the **Sonnet 5 model id (EU)**. Search keeps running locally
(Claude cannot make embeddings), so Ollama with `nomic-embed-text` stays installed.

1. Copy `.env.bifrost.example` and rename the copy to `.env` (the old `.env` is replaced;
   copy your `SLIDE_CROP` / `SLIDE_MASK` values over first).
2. Fill in 3 lines, values exactly as in the e-mail:
   ```
   BIFROST_BASE_URL=<the address; the /anthropic ending can stay>
   BIFROST_API_KEY=<the virtual key>
   CHAT_MODEL=<the Sonnet 5 id, e.g. bedrock/eu.anthropic.claude-sonnet-...>
   ```
3. Check:
   ```
   uv run doctor.py --models
   ```
   Look for `[OK] chat model answers: 'OK'`. A model list may or may not appear; either is fine.
   If you get HTTP 401: set `BIFROST_AUTH_HEADER=x-bf-vk` and try again.
4. Prepare again with Claude reading the slides (segments and slides are kept):
   ```
   uv run pipeline.py --mode easy --vision
   ```
5. Same commands as in A3:
   ```
   uv run make_video.py --recording 13 --minutes 2 --plan-only
   uv run make_video.py --rerender "data/output/<folder>"
   ```
   Compare `script.md` with the local one: that comparison is useful for the demo.

**Other models:** for one run add `--model <id>` (e.g. the Opus 5 id from the e-mail); to switch
for good, change `CHAT_MODEL`. Use EU ids only (the platform team asked for that).

### Models and cost

| Job | Model | Cost (estimate) |
|---|---|---|
| Storyboard, text lesson, reading slides | Claude Sonnet 5 (EU) | ~$0.03-0.06 per 2-min video; ~$0.006 per slide, once |
| Search | nomic-embed-text, local | free |
| Comparison now and then | Claude Opus 5 | ~2.5x Sonnet |

### Spending lock

| Setting | Default | What happens |
|---|---|---|
| `DAILY_BUDGET_USD` | 20 | no more gateway calls today once reached |
| `RUN_BUDGET_USD` | 5 | one command stops once it has spent this much (protects against loops) |
| hard maximum | 100 | fixed in the code: a higher `DAILY_BUDGET_USD` is cut down to 100 |

Costs are estimated from token counts with list prices (+10% for EU ids, and the higher price
when a version is unknown), logged per call in `cost_log.csv`. `uv run doctor.py` shows today's
total. Only this pipeline is counted: Claude Code on the same key counts toward the platform's
own limit too.

---

## Part C: check quality

**Quality checks:** `uv run quality.py --all [--review]`, or
`make_video.py ... --check` for one video. Status per video:

| Status | Meaning |
|---|---|
| OK | nothing found |
| CHECK | worth a look: length off, reads a slide aloud, text too long on screen, words to listen to, review score 3 or lower |
| FAIL | broken (missing or out-of-sync video) or the review found invented content |
| PLAN | storyboard only, not rendered yet |

**Pronunciation:** if the voice says a term wrongly, add a row to `pronunciation.csv`
(`KPI,K P I`), test with `uv run tts.py "The KPI ..."`, then `--rerender`. Subtitles keep the
original spelling. The quality check lists unusual words to listen to.

**NotebookLM route:** `uv run export_notebooklm.py [--combined]` writes the prepared recordings
as `.md` files in `data/output/notebooklm/` (transcript with timestamps, slide texts, no speaker
names). Upload them to NotebookLM as sources (it does not take MP4). Upload only what you are
allowed to: NotebookLM is a Google service outside the platform.

---

## Commands at a glance

| Command | What it does |
|---|---|
| `uv run app.py [--port 8765]` | Lesson Studio: make a lesson by clicking, see results and costs per model and output, how to ask |
| `uv run smoke.py [--quick] [--only document,video]` | one small result of every kind with the real model: what works, time, cost, verdict |
| **`uv run ask.py "what you want, how long, which level"`** | **the main entry: a lesson from the whole knowledge base** (document, simple slides, narrated video, two-person conversation or audio, as asked; 4-15 min videos: a course with chapters) |
| `uv run documents.py [--ai-triage] [--render-html]` | what was found in `data/input` besides recordings, and why each file is used or left out |
| `uv run topics.py` | what each recording covers, with ready requests → `data/output/topics/topics.md` |
| `uv run course.py "request" --minutes 8` / `--redo-chapter N <folder>` | a course directly / remake one chapter |
| `uv run evaluate.py --draft`, then `uv run evaluate.py` | quality as numbers on a fixed set of requests, compared with the last run |
| `uv run doctor.py [--models] [--try <id>]` | checks the setup (and tries each Claude model once: which ones your key may use) |
| `uv run inventory.py` | finds recordings, pairs transcripts and Excel rows → `data/manifest.csv` |
| `uv run catalog.py [--no-analyse]` | description.xlsx vs data/input → `data/output/description_checked.xlsx`: video/audio/transcript present or MISSING, transcript source, slides, topics; recordings not in the Excel added with a 1-3 sentence description (after `pipeline.py --mode all --vision`) |
| `uv run pipeline.py --mode easy [--vision] [--force]` | prepares all included recordings |
| `uv run slides.py --video <id> --preview` | shows the slide area (crop) and the hidden area (mask) |
| `uv run index.py search "..."` | tests search |
| `uv run make_video.py "topic" --minutes 2` | video from all recordings |
| `uv run make_video.py --recording 13 --minutes 2` | video from one recording (Excel id) |
| `uv run make_video.py ... --plan-only` / `--rerender <folder>` | plan only / voice + video from an (edited) plan |
| `uv run lesson.py "topic" --minutes 5 --level beginner` | document (`lesson.html` + `lesson.md`) |
| `uv run deck.py "topic" --minutes 4 [--style clean]` / `--rerender <folder>` | simple presentation: `slides.pdf`, `slides.pptx`, speaker notes |
| `uv run make_video.py "topic" --minutes 3 --style dialogue [--characters cartoon\|people]` | video in which a host and an expert talk |
| `uv run quality.py <folder>` / `--all [--review]` | checks of any result: video, slides or document |
| `uv run answer.py "question"` / `uv run retrieval.py "question"` | a text answer with the source, place and evidence of every sentence / only what would be used |
| `uv run qa_eval.py --draft 20` / `qa_eval.py` / `--compare` | test questions from the material / answer quality report (retrieval, hallucination, correctness, declines, copying) / retrieval modes compared |
| `uv run benchmark.py "topic" --voices kokoro:bf_emma,elevenlabs:lily@flash --renderers html,remotion-pro --presenters cat,person-2d` | the same script through different voices, models, video engines and presenters → `comparison.html`, `benchmark.csv` |
| `uv run tiers.py [--minutes 5]` | cost per budget tier at 20/100/500 videos a month → `tiers.md` |
| `uv run usage.py` | spent today vs the daily limit; ElevenLabs characters used and left this month (and when they reset); Remotion has no usage limit |
| `RENDERER=remotion-ai` in `.env`, or `benchmark.py --renderers "html,remotion-ai"` | Remotion where Claude writes each scene's animation code (checked, compiled; a scene that fails uses the template) |
| `uv run avatar.py export <folder>` / `make <folder> --clip x.mp4 --tool synthesia [--scenes N]` | AI presenter from another tool beside our slides (`--scenes`: a free plan's 1-minute clip holds the first N scenes) |
| `RENDERER=remotion-pro` / `benchmark.py --renderers "html,remotion-pro"` | motion-design video with presenters (talk, wave, point, happy on the answer); `PRO_PRESENTER=robot`, `cat`, `person-3d`, `person-2d` (+ `PRO_AVATAR`) or `none` |
| `uv run eleven_video.py check` / `clip <folder> --prompt 1` / `presenter <folder> --image p.jpg` | ElevenLabs video by API (Pro plan): Veo clip, talking presenter; credits read from the account → cost |
| `uv run ai_clips.py voice <folder> --seconds 8` / `presenter <folder> --clip talking.mp4` | narration excerpt for a lip-sync model / the talking presenter in a window over the lesson |
| `uv run ai_clips.py prompts <folder>` / `add <folder> --clip a.mp4 --clip b.mp4 --tool elevenlabs --model veo-3.1-fast --cost-usd 1.5` | real-looking AI clips (Google Veo, made by hand in the tool) as intro, cutaways and outro of a finished video → `lesson_video_clips.mp4`; cost on the run card |
| `uv run compare.py [--topic "..."] [--kind video]` | all results side by side from their run cards: model, voice, cost, time, verdict → `data/output/comparison.csv` |
| `uv run tools/add_lottie.py --talk x.json [--point y.json] --licence "..."` / `--remove` | professional animated presenter for the presenter style |
| `uv run quality.py --all [--review] [--visual]` | checks every video (OK / CHECK / FAIL / PLAN): true numbers, voice matches screen, layout, audio, safety; `--visual` = the vision model reviews every slide |
| `uv run export_notebooklm.py` | `.md` sources for NotebookLM |
| `uv run python -m pytest tests` | automatic tests (offline, free, ~1-2 min) |
| `uv run --with ruff ruff check .` | code style and common bugs |
| `uv run tts.py "text"` | voice test |
| `uv run llm.py` | model smoke test |

Levels: `beginner`, `intermediate`, `expert`.

## Output of a video

Folder `data/output/<time>-<topic>-video-<n>m/`:

| File | What |
|---|---|
| `lesson_video.mp4` | the video lesson (subtitles inside as a track) |
| `lesson_video.srt` | subtitles |
| `lesson_audio.mp3` | audio lesson |
| `script.md` | narration + on-screen text per scene, with sources (recording + time) |
| `storyboard.json` | the same as data; edit it and `--rerender` |
| `info.json` | length, timings, estimated cost, warnings |
| `quiz.md`, `quiz.json` | the check-yourself questions with answers |
| `chapters.txt` | courses: table of contents (the chapters are also marks in the mp4) |
| `audio/voice_*.wav` | the voice per part of a scene; unchanged texts are reused on `--rerender` |
| `scenes/scene_NN.png` | each scene in its final state (for the report and a quick look) |

Look: `STYLE` in `.env` or in the request ("simple", "with pictures", "with a presenter"):
`clean` designer slides only, `illustrated` + pictures of people at work where they fit the
slide, `presenter` a person stands next to every slide. Needs `uv add playwright` once (it
uses the installed Edge if Playwright's Chromium can't be downloaded); without it the simple
slides are used. `uv run illustrations.py` shows all pictures; `uv run images.py` tests AI pictures.

Animation: elements appear or are highlighted when the voice reaches them (`build` /
`highlight` in `storyboard.json`, with `cues` = where in the narration each element starts);
`script.md` shows the cut points as ▸. `ANIMATION=no` in `.env` renders still slides.

`storyboard.json` also records how it was made (`settings`: model, prompt and code version,
switches), shown at the end of `script.md`.

## When something fails

| Message | Fix |
|---|---|
| `Is Ollama running?` | start the Ollama app |
| `ollama pull ...` in doctor | the model isn't downloaded yet: run that command |
| `Prompt is ~N tokens ... exceeds NUM_CTX` | lower `MAX_MATERIAL_WORDS` or raise `NUM_CTX` in `.env` |
| `HTTP 401/403 ... rejected` | wrong key or model not allowed for it: check `.env`, ask the platform team |
| `HTTP 404 ... model id unknown` | copy the exact id from `uv run doctor.py --models` |
| `CERTIFICATE_VERIFY_FAILED` | company TLS inspection: set `CA_BUNDLE` to the company `.pem` |
| `Can this laptop reach the gateway` | VPN/proxy, or the URL works only inside their cluster |
| `Index was built with ...` | embedding model changed: `uv run index.py build` |
| `Recordings don't seem to cover` | other wording, or lower `MIN_SCORE` |
| `Spending lock: ...` | daily or per-command limit reached; raise it in `.env` only if expected (max 100/day) |
| `CHAT_MODEL is empty` | paste the Sonnet id from the e-mail |
| `Both gateway routes failed` | check the address and the model id; send the message (without the key) |
| presenter visible on slides in the video | `SLIDE_MASK`, then `--rerender` |
| very slow storyboard locally | `CHAT_MODEL=llama3.2:3b`, or shorter `--minutes` |

## Files

| File | Role |
|---|---|
| `config.py` | all settings, read from `.env` |
| `llm.py` | every model call: gateway or Ollama, retries, cost log, budget stop |
| `inventory.py`, `ingest.py`, `slides.py`, `describe_slides.py`, `index.py`, `pipeline.py`, `transcribe.py` | preparing recordings |
| `lesson.py` | text lesson |
| `storyboard.py`, `tts.py`, `render.py`, `make_video.py` | narrated video |
| `quality.py` | checks of every result |
| `export_notebooklm.py` | NotebookLM source files |
| `pronunciation.csv` | how the voice says special terms |
| `tests/` | automatic tests with a stand-in gateway and voice |
| `store.py`, `media.py` | files on disk; ffmpeg / Tesseract |
| `skills/compose-video/`, `skills/compose-lesson/` | the prompts (SKILL.md, the platform's format) |
| `evaluate.py`, `eval/` | scoring lessons, videos and courses against a golden set |
| `slides_html.py` | designer slides drawn by a browser engine (icons, cards, diagrams, stats, charts, formulas, calculations, motion, the presenter) |
| `facts.py`, `safety.py` | numbers checked against the sources, calculations recomputed / inappropriate content, icons, hidden instructions, AI-picture check |
| `illustrations.py`, `images.py` | people and office illustrations (Humaaans, CC0) / AI pictures via the gateway |
| `assets/` | icons (Lucide, ISC), font (Inter, OFL), people (Humaaans, CC0) with their licences |
| `documents.py` | Word, PowerPoint, PDF, Excel, HTML quizzes, JSON, XML, images: read, filtered, ingested |
| `ask.py`, `course.py`, `topics.py` | requests in plain words, courses with chapters, topic map |
| `docs/DESIGN.md` | how it works and why (decisions, quality, limits) |
| `NEXT_STEPS.md` | what's left to the demo (`doctor.py` reminds you) |
| `doctor.py` | checks the setup |
