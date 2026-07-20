"""
CLI entry point: run the content transformation crew on a local text
file OR a YouTube URL.

Usage:
    python main.py sample_input/transcript_sample.txt
    python main.py --verbose https://www.youtube.com/watch?v=VIDEO_ID

Writes the result to output.md in the current directory.
"""
import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents import build_crew
from youtube_source import get_transcript_from_youtube


def _looks_like_youtube_url(value: str) -> bool:
    return "youtube.com" in value or "youtu.be" in value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the content transformation crew on a file or YouTube URL."
    )
    parser.add_argument("source", help="Path to a source text file, or a YouTube URL")
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", default=False,
        help="Print agent/task progress as it runs",
    )
    parser.add_argument(
        "--quiet", dest="verbose", action="store_false",
        help="Suppress agent/task progress output (default)",
    )
    return parser.parse_args()


def main():
    # Load parent .env if nested inside growth-intel-agent, then this
    # project's own on top.
    parent_dir = Path(__file__).parent.parent
    if (parent_dir / "bridge.py").exists():
        load_dotenv(parent_dir / ".env")
    load_dotenv(override=True)

    args = parse_args()

    if _looks_like_youtube_url(args.source):
        try:
            source_text = get_transcript_from_youtube(args.source)
        except ValueError as e:
            print(f"Could not get a transcript: {e}")
            sys.exit(1)
    else:
        source_path = Path(args.source)
        if not source_path.exists():
            print(f"File not found: {source_path}")
            sys.exit(1)
        source_text = source_path.read_text(encoding="utf-8")

    crew = build_crew(source_text, verbose=args.verbose)
    result = crew.kickoff()

    section_titles = ["Analyst Brief", "Blog Draft", "Social Posts"]
    sections = [
        f"## {title}\n\n{task_output.raw}"
        for title, task_output in zip(section_titles, result.tasks_output)
    ]

    output_path = Path("output.md")
    output_path.write_text("\n\n---\n\n".join(sections), encoding="utf-8")
    print(f"\nDone. Output written to {output_path}")


if __name__ == "__main__":
    main()
