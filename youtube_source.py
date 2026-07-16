"""
Extracts a plain-text transcript from a YouTube video URL, so the same
transcript -> blog -> social pipeline works directly on a video link,
not just pasted text or a local file.

Uses youtube-transcript-api, which reads YouTube's public caption data
directly and needs no API key -- Google's official Data API can only
fetch captions for videos you own via OAuth, not arbitrary public
videos, so this unofficial library is the standard way to do this.
It's reading a public but undocumented endpoint, so two limitations
are worth knowing:
  - It can break if YouTube changes something internally.
  - It doesn't work on videos with no captions at all (auto-generated
    or manual) -- a real minority of videos, but not zero.
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
            # No English transcript specifically -- fall back to
            # whatever language actually exists rather than giving up.
            transcript_list = ytt_api.list(video_id)
            first_available = next(iter(transcript_list))
            fetched = first_available.fetch()
    except TranscriptsDisabled:
        raise ValueError(
            "This video has captions disabled -- no transcript is available."
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
