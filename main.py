"""
CLI entry point: run the content transformation crew on a local text
file OR a YouTube URL.

Usage:
    python main.py sample_input/transcript_sample.txt
    python main.py https://www.youtube.com/watch?v=VIDEO_ID

Writes the result to output.md in the current directory.
"""
import sys
from pathlib import Path

from dotenv import load_dotenv

from agents import build_crew
from youtube_source import get_transcript_from_youtube


def _looks_like_youtube_url(value: str) -> bool:
    return "youtube.com" in value or "youtu.be" in value


def main():
    load_dotenv()

    if len(sys.argv) != 2:
        print("Usage: python main.py <path-to-source-text-file | youtube-url>")
        sys.exit(1)

    arg = sys.argv[1]

    if _looks_like_youtube_url(arg):
        try:
            source_text = get_transcript_from_youtube(arg)
        except ValueError as e:
            print(f"Could not get a transcript: {e}")
            sys.exit(1)
    else:
        source_path = Path(arg)
        if not source_path.exists():
            print(f"File not found: {source_path}")
            sys.exit(1)
        source_text = source_path.read_text(encoding="utf-8")

    crew = build_crew(source_text)
    result = crew.kickoff()

    # crew.kickoff() runs all three tasks, but str(result) only returns
    # the LAST task's raw output (CrewOutput.__str__ just returns
    # self.raw). Build the full document from result.tasks_output so the
    # analyst's brief and blog draft aren't silently dropped.
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
