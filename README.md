# Content Transformation Agent

Turns raw technical material (docs, transcripts, YouTube videos) into a blog draft and social posts, using two LLM agents in sequence (CrewAI + Groq's Llama 3.3 70B): a Content Analyst extracts the core insight, then a Technical Writer drafts the blog and social posts. Splitting "what's interesting" from "write it well" beats one prompt trying to do both.

## Setup

### 1. Check your Python version

CrewAI requires a specific version range and fails silently outside it — it installs a stale, broken version instead of erroring. Check the current requirement and your own version before creating a venv:

**bash**
```bash
curl -s https://pypi.org/pypi/crewai/json | python3 -c "import json,sys; print(json.load(sys.stdin)['info']['requires_python'])"
python3 --version
```

**PowerShell**
```powershell
(Invoke-RestMethod https://pypi.org/pypi/crewai/json).info.requires_python
python --version
```

### 2. If your Python isn't in range

Two options:

1. **Install a matching version directly** (e.g. 3.12) from [python.org](https://www.python.org/downloads/), then point the venv at it explicitly in step 3.

   OR

2. **Use [uv](https://astral.sh/uv/)** to install an isolated Python version without touching your system install:

    **bash**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv python install 3.12
    ```

    **PowerShell**
    ```powershell
    irm https://astral.sh/uv/install.ps1 | iex
    uv python install 3.12
    ```

### 3. Create the venv and install dependencies

Pick whichever path you took in step 2.

**Standard venv (system Python, or a version installed directly)**

bash (use `python3.12` instead of `python3` if you installed a second version (e.g. 3.12)):
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PowerShell (use `py -3.12` instead of `python` if you installed a second version (e.g. 3.12)):
```powershell
py -3.12 -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

OR

**With uv**

bash:
```bash
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install -r requirements.txt
```

PowerShell:
```powershell
uv venv --python 3.12 venv
venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

### 4. Configure environment variables

Same command in bash and PowerShell:
```bash
cp .env.example .env
```
Fill in `GROQ_API_KEY` (free at [console.groq.com/keys](https://console.groq.com/keys)).

### 5. Run it

```bash
# CLI — takes a file or a YouTube URL, writes output.md
python main.py sample_input/transcript_sample.txt
python main.py https://www.youtube.com/watch?v=VIDEO_ID

# or the frontend (same server, no separate client)
uvicorn api:app --reload   # http://localhost:8000
```

Runs quietly by default. Add `--verbose` (before or after the file/URL) to print agent/task progress as it runs — doesn't apply to the API, which never prints crew progress either way.

YouTube input uses `youtube-transcript-api` to pull the caption track (no API key; unofficial; only works if the video has captions).

## Outputs

- **`output.md`** (CLI only, overwritten every run) — three sections: the Analyst's brief, the blog draft, and the social posts, each from one of the crew's tasks.
- **API (`/generate`)** — no file is written; the same three pieces come back as JSON (`analyst_brief`, `blog_draft`, `social_posts`), plus `output`, the same three sections combined into one string like `output.md`.

## The no-code layer

`n8n/workflow.json` — a webhook triggers the FastAPI `/generate` endpoint above. Verified end to end in a live n8n instance:

![n8n workflow executing successfully, all three nodes green](n8n/execution.gif)

## What's next

- A research/competitive-intel agent feeding this one — built, see [growth-intel-agent](https://github.com/paulogabe1/growth-intel-agent)
- Guardrails on the Writer agent (a style-guide review step)
- Structured output instead of markdown-in-a-string

## Notes

No API key is included anywhere in this repo. `.env` is gitignored.
