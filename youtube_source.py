"""
Pulls a plain-text transcript from a YouTube URL so the pipeline can
run directly on a video, not just pasted text or a file.

Uses youtube-transcript-api (reads YouTube's public captions, no API
key needed -- Google's official API only covers videos you own).
Unofficial, so it can break if YouTube changes something, and it only
works on videos that actually have captions.
"""
import re

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    InvalidVideoId,
    RequestBlocked,
    IpBlocked,
    AgeRestricted,
)
from youtube_transcript_api.formatters import TextFormatter

_VIDEO_ID_PATTERN = re.compile(
    r"(?:youtube\.com/watch\?v=|youtube\.com/shorts/|youtube\.com/embed/|youtu\.be/)([\w-]{11})"
)


def extract_video_id(url: str) -> str:
    match = _VIDEO_ID_PATTERN.search(url)
    if not match:
        raise ValueError(
            f"Couldn't find a YouTube video ID in that URL: {url}"
        )
    return match.group(1)


def get_transcript_from_youtube(url: str) -> str:
    video_id = extract_video_id(url)
    ytt_api = YouTubeTranscriptApi()

    try:
        try:
            fetched = ytt_api.fetch(video_id, languages=("en",))
        except NoTranscriptFound:
            # No English transcript -- fall back to whatever language exists.
            transcript_list = ytt_api.list(video_id)
            first_available = next(iter(transcript_list))
            fetched = first_available.fetch()
    except TranscriptsDisabled:
        raise ValueError(
            "This video has captions disabled. No transcript is available."
        )
    except VideoUnavailable:
        raise ValueError(
            "This video is unavailable (private, deleted, or region-locked)."
        )
    except InvalidVideoId:
        raise ValueError(f"'{video_id}' isn't a valid YouTube video ID.")
    except (RequestBlocked, IpBlocked):
        raise ValueError(
            "YouTube blocked this request (common on cloud/server IPs, "
            "less common on a home connection). Try again, or from a "
            "different network."
        )
    except AgeRestricted:
        raise ValueError(
            "This video is age-restricted and can't be fetched without "
            "an authenticated session."
        )

    return TextFormatter().format_transcript(fetched)
