"""
FastAPI wrapper around the content crew. 
1. Serves a basic frontend for interactive use, and 
2. exposes /generate as a plain HTTP POST so no-code automation tools 
    (n8n, Zapier, Make) can trigger it too. 
This is the piece an n8n "HTTP Request" node calls in n8n/workflow.json.

/generate accepts either pasted text OR a YouTube URL. Not both, not neither.

Run locally:
    uvicorn api:app --reload
Then open http://localhost:8000 in a browser.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from agents import build_crew
from youtube_source import get_transcript_from_youtube

# Load parent .env if nested inside growth-intel-agent.
# Load current .env as needed to override.
parent_dir = Path(__file__).parent.parent
if (parent_dir / "bridge.py").exists():
    load_dotenv(parent_dir / ".env")
load_dotenv(override=True)
app = FastAPI(title="Content Transformation Agent")

FRONTEND_PATH = Path(__file__).parent / "frontend" / "index.html"
SAMPLE_PATH = Path(__file__).parent / "sample_input" / "transcript_sample.txt"


class GenerateRequest(BaseModel):
    source_text: str | None = None
    youtube_url: str | None = None


class GenerateResponse(BaseModel):
    analyst_brief: str
    blog_draft: str
    social_posts: str
    output: str  # all three combined, same format as main.py's output.md


@app.get("/", response_class=HTMLResponse)
def frontend():
    return FRONTEND_PATH.read_text(encoding="utf-8")


@app.get("/sample")
def sample():
    return {"source_text": SAMPLE_PATH.read_text(encoding="utf-8")}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if req.youtube_url and req.source_text:
        raise HTTPException(
            status_code=400,
            detail="Provide either source_text or youtube_url, not both.",
        )

    if req.youtube_url:
        try:
            source_text = get_transcript_from_youtube(req.youtube_url)
        except ValueError as e:
            # ValueError passed from get_transcript_from_youtube
            raise HTTPException(status_code=400, detail=str(e))
    elif req.source_text:
        source_text = req.source_text
    else:
        raise HTTPException(
            status_code=400,
            detail="Provide either source_text or youtube_url.",
        )

    crew = build_crew(source_text)
    result = crew.kickoff()

    analyst_brief, blog_draft, social_posts = (t.raw for t in result.tasks_output)
    combined = "\n\n---\n\n".join(
        [
            f"## Analyst Brief\n\n{analyst_brief}",
            f"## Blog Draft\n\n{blog_draft}",
            f"## Social Posts\n\n{social_posts}",
        ]
    )
    return GenerateResponse(
        analyst_brief=analyst_brief,
        blog_draft=blog_draft,
        social_posts=social_posts,
        output=combined,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
