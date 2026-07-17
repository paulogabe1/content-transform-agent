# Content Transformation Agent

A small multi-agent pipeline that turns raw technical material (docs,
whitepapers, call/video transcripts) into a blog draft and a set of
social posts, using two LLM-backed agents in sequence rather than one
prompt trying to do everything at once.

Built as a portfolio piece to explore agent frameworks and LLM-driven
content workflows -- small enough to read end to end in five minutes,
real enough to actually run.

## Architecture

```
source text (doc / transcript)
        |
        v
+---------------------+      +-----------------------+
|  Content Analyst     | ---> |   Technical Writer     |
|  (extracts the       |      |  (drafts blog, then    |
|   one real insight)  |      |   social posts)        |
+---------------------+      +-----------------------+
        |                             |
        +-------------+---------------+
                       |
                       v
               output.md (or JSON via API)
```

Two agents, three sequential tasks, built with [CrewAI](https://docs.crewai.com)
calling Groq's free tier (Llama 3.3 70B). The Analyst pass exists because
skipping straight from raw transcript to blog post tends to produce
generic marketing copy -- separating "figure out what's actually
interesting" from "write it well" gets noticeably better output for
not much extra cost.

## Why CrewAI

Chose CrewAI over a bare LangChain chain because the job is naturally
role-based (an analyst and a writer genuinely do different things),
and CrewAI's `Agent`/`Task`/`Crew` primitives map onto that directly
without extra scaffolding. For a single-shot pipeline like this one,
LangGraph would be overkill -- no branching state machine needed here.

## Running it

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY (free at console.groq.com/keys)

# CLI -- writes output.md, works on a file OR a YouTube URL
python main.py sample_input/transcript_sample.txt
python main.py https://www.youtube.com/watch?v=VIDEO_ID

# or the frontend -- one server, no separate client needed
uvicorn api:app --reload
# then open http://localhost:8000 in a browser
```

The frontend (`frontend/index.html`) is served directly by `api.py` at
`/` -- paste text (or click "Load sample transcript" to pull in
`sample_input/transcript_sample.txt` via the `/sample` endpoint), hit
Generate, and the three outputs land in tabs. Same origin as the API
itself, so there's no CORS setup and no manual JSON to write by hand --
the page just calls `/generate` directly. The pipeline strip at the top
cycles through stage names while a request is in flight; it's a sense
of where things likely are, not a literal live status, since a single
`/generate` call doesn't report progress mid-run.

## YouTube input

Both the CLI and `/generate` accept a YouTube URL instead of raw text
-- `youtube_source.py` pulls the video's caption track via
`youtube-transcript-api` (no API key needed) and feeds that transcript
into the same pipeline. Two things worth knowing:

- It reads YouTube's public caption data directly, not an official
  Google API -- there isn't one for arbitrary public videos. This is
  an established, widely-used approach, but it's unofficial, so it can
  break if YouTube changes something internally.
- It only works on videos that have captions at all (manual or
  auto-generated). A video with captions disabled has no transcript to
  extract, and the error message says so directly rather than failing
  silently.

For scripted/automated calls without the browser, `main.py` (CLI) or a
plain `curl -X POST http://localhost:8000/generate -d '{"source_text":"..."}'`
(or `{"youtube_url": "..."}`)
both still work -- the frontend is for interactive use, not a
replacement for those.

## The no-code layer

`n8n/workflow.json` is an importable n8n workflow: a webhook receives
a source doc, calls the FastAPI `/generate` endpoint above, and
returns the draft. The idea is that a real version of this would be
triggered by a new file landing in a shared drive or CMS, rather than
a manual webhook call -- n8n (or Zapier/Make) is the piece that
watches for that trigger so nobody has to run the script by hand.

### Proof it runs

![n8n workflow executing successfully, all three nodes green](n8n/execution.gif)

Imported into a real n8n instance, triggered via its test webhook, and
run end to end: the webhook received a source document, the HTTP
Request node called the FastAPI `/generate` endpoint above, and the
response came back through to the Respond node -- all three nodes
completed successfully in a single live execution, not just a workflow
file that looks plausible on paper.

## Notes

- Sample input (`sample_input/transcript_sample.txt`) is original
  writing for this demo -- a short explainer on why traditional IAM
  breaks down for AI agents/non-human identities, picked because it's
  a genuinely interesting problem in the identity/security space, not
  because it was handed to me.
- No API key is included anywhere in this repo -- `.env` is gitignored.