"""
Agent and task definitions for the content transformation crew.

Two-agent, three-task pipeline:
  1. Content Analyst  -> reads raw technical material and extracts the
                          strongest, most specific insight worth writing about
  2. Technical Writer  -> turns that analysis into a blog draft, then a set
                          of social posts

Built with a workload-identity / AI-agent-security audience in mind (see
sample_input/), but the pipeline works on any technical source material:
docs, whitepapers, webinar transcripts, changelogs.
"""

import os
import re
import time
from crewai import Agent, Task, Crew, Process, LLM

# --- Workaround for a known CrewAI bug (GitHub issue #5886) ---
# CrewAI injects an Anthropic-only prompt-caching marker into every
# message regardless of provider. Groq (and other OpenAI-compatible
# APIs) reject it outright, so every request fails with something like:
#   GroqException - property 'cache_breakpoint' is unsupported
# This patches CrewAI's marker function to a no-op. Safe to delete once
# https://github.com/crewAIInc/crewAI/issues/5886 is fixed upstream.
import crewai.llms.cache as _crewai_cache

_crewai_cache.mark_cache_breakpoint = lambda msg: msg


def build_llm() -> LLM:
    """
    Model id is read from the environment so it's easy to swap without
    touching code. Defaults to Groq's Llama 3.3 70B - genuinely free
    (no card), and Groq's dedicated LPU hardware means far less
    contention than a popular hosted model like Gemini Flash under
    high demand. Get a key at https://console.groq.com/keys and check
    https://console.groq.com/docs/models for current models/limits.

    timeout + max_retries give a hard deadline per request instead of
    letting a stalled or overloaded call hang indefinitely.
    """
    model = os.getenv("GROQ_MODEL", "groq/llama-3.3-70b-versatile")
    return LLM(model=model, temperature=0.4, timeout=60, max_retries=5)


def build_agents(llm: LLM):
    analyst = Agent(
        role="Technical Content Analyst",
        goal=(
            "Read raw technical material and extract the strongest, most "
            "specific idea worth turning into marketing content - no fluff, "
            "no generic takeaways."
        ),
        backstory=(
            "A former security engineer who moved into content strategy. "
            "Reads whitepapers and transcripts the way an editor reads a "
            "first draft: looking for the one idea that actually matters, "
            "and cutting everything that's filler."
        ),
        llm=llm,
        verbose=True,
        max_execution_time=90,  # fail loudly instead of hanging forever
    )

    writer = Agent(
        role="Technical Content Writer",
        goal=(
            "Turn the analyst's brief into a clear, credible blog draft and "
            "a set of social posts that a technical audience won't roll "
            "their eyes at."
        ),
        backstory=(
            "Writes for developers and security practitioners. Prioritizes "
            "precision and a direct voice over marketing-speak. Never opens "
            "with 'In today's rapidly evolving landscape...'."
        ),
        llm=llm,
        verbose=True,
        max_execution_time=90,
    )

    return analyst, writer


def build_tasks(analyst: Agent, writer: Agent, source_text: str):
    analyze = Task(
        description=(
            "Read the following technical source material and extract:\n"
            "1. The single strongest, most specific claim or insight in it\n"
            "2. 3-5 supporting facts or details worth keeping\n"
            "3. Who the intended reader is and what they'd want to know next\n\n"
            f"SOURCE MATERIAL:\n---\n{source_text}\n---"
        ),
        expected_output=(
            "A short brief with three sections: core insight (1-2 sentences), "
            "supporting points (bulleted list), target reader (1 sentence)."
        ),
        agent=analyst,
    )

    write_blog = Task(
        description=(
            "Using the analyst's brief, write a 400-500 word blog post "
            "draft. Direct opening (never 'In today's world...'), one clear "
            "argument carried through, concrete examples over abstractions. "
            "Include a short, specific title - not a generic one: 'Why Your "
            "IAM Model Breaks the Moment an Agent Touches It' works, "
            "'Securing Agent IAM' does not. Do not end with 'In conclusion,' "
            "'In summary,' or any generic wrap-up phrase - end on the "
            "concrete example or a direct claim instead. Do not repeat the "
            "same phrase or sentence structure more than once in the piece."
        ),
        expected_output="A titled blog post draft, 400-500 words, in markdown.",
        agent=writer,
        context=[analyze],
    )

    write_social = Task(
        description=(
            "Using the analyst's brief and the blog draft, write 3 short "
            "social posts (X/LinkedIn style, under 280 characters each). "
            "Each post must take a genuinely different angle:\n"
            "1. Lead with the risk/failure scenario (what goes wrong today)\n"
            "2. Lead with a provocative question or contrarian claim\n"
            "3. Lead with the concrete fix/solution as a takeaway\n"
            "Do not summarize the blog three times - each post should be "
            "readable on its own with a distinct hook."
        ),
        expected_output="3 numbered social posts, each under 280 characters.",
        agent=writer,
        context=[analyze, write_blog],
    )

    return [analyze, write_blog, write_social]


# --- Handling transcripts too long for a single request ---
# There's no reliable way to ask a provider "what's my current rate
# limit?" before making a request -- some expose it in response
# headers after a call, but that's reactive, not something you can
# check upfront. So these are configurable defaults, not an
# auto-detected value: they're conservative numbers tuned for Groq's
# free tier (12,000 tokens/minute, far below Llama 3.3 70B's real
# 131k token context window). If you're on a higher-limit tier or a
# different model, raise these in .env instead of editing code --
# there's nothing here that inspects which model or tier you're
# actually on.
#
# These are deliberately well under a naive "4 chars/token" estimate.
# Conversational transcripts (fillers, contractions, dialogue
# punctuation) tend to tokenize less efficiently than clean prose, and
# the 12,000 limit is cumulative per minute across ALL calls, not a
# ceiling on any one request -- so each individual chunk needs real
# headroom, not just barely-under-the-line sizing.
MAX_SAFE_CHARS = int(os.getenv("MAX_SAFE_CHARS", "12000"))  # ~3,000 tokens, worst case
CHUNK_SIZE_CHARS = int(os.getenv("CHUNK_SIZE_CHARS", "8000"))  # ~2,000 tokens/chunk, worst case
CHUNK_DELAY_SECONDS = int(os.getenv("CHUNK_DELAY_SECONDS", "5"))


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE_CHARS) -> list[str]:
    """
    Split text into pieces no larger than chunk_size characters.

    The first version of this only split on blank-line paragraph
    breaks ("\\n\\n") -- which works for prose, but not for YouTube
    transcripts. TextFormatter (see youtube_source.py) joins caption
    lines with a single "\\n", never a blank line, so a real transcript
    has zero blank-line breaks in it at all. Splitting on "\\n\\n" alone
    would find nowhere to cut and silently return the entire transcript
    as one "chunk" -- which is exactly what happened here: the pipeline
    still failed with a request nearly as large as the original,
    because chunking never actually occurred.

    This version tries progressively finer separators -- blank lines,
    then single newlines, then sentence breaks -- and only moves to
    the next one if the current one fails to produce small enough
    pieces. If nothing natural works, it hard-cuts by character count
    as a last resort, so this can never silently fail to split again.
    """
    if len(text) <= chunk_size:
        return [text]

    pieces = [text]
    separator = ""
    for candidate_separator in ["\n\n", "\n", ". "]:
        candidate_pieces = text.split(candidate_separator)
        if all(len(p) <= chunk_size for p in candidate_pieces):
            pieces = candidate_pieces
            separator = candidate_separator
            break

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        addition = f"{current}{separator}{piece}" if current else piece
        if current and len(addition) > chunk_size:
            chunks.append(current)
            current = piece
        else:
            current = addition
    if current:
        chunks.append(current)

    # Safety net: if the separator search above never found one that
    # worked (e.g. one giant unbroken line longer than chunk_size),
    # hard-cut whatever is still oversized rather than returning it
    # as-is.
    final_chunks: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            final_chunks.extend(
                chunk[i : i + chunk_size] for i in range(0, len(chunk), chunk_size)
            )
    return final_chunks


def condense_if_needed(source_text: str, llm: LLM) -> str:
    """
    If source_text is already small, this is a no-op. If not, it's
    split into chunks, each chunk is condensed to its key points with
    its own separate, much smaller LLM call, and the condensed pieces
    are joined into one short document -- that's what actually gets
    passed into the main Analyst/Writer pipeline afterward.

    The per-request limit is only half the problem: the free tier's
    token cap is cumulative across ALL calls in a rolling minute, not
    just a ceiling on any one request. Running several chunk calls
    back-to-back could still add up past 12,000 tokens even though
    each individual one fits comfortably. CHUNK_DELAY_SECONDS spreads
    calls out to reduce that risk -- it doesn't eliminate it, since
    there's no way to guarantee timing against a shared rate limit
    from the client side. build_llm()'s timeout + max_retries are the
    backstop for whatever this doesn't catch.
    """
    if len(source_text) <= MAX_SAFE_CHARS:
        return source_text

    chunks = _split_into_chunks(source_text)
    condensed_parts = []
    for i, chunk in enumerate(chunks):
        response = llm.call(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract the 3-5 most specific, substantive points from "
                        "this excerpt of a longer transcript. Skip filler, small "
                        "talk, and repetition -- keep only points a reader would "
                        "actually want. Be concise.\n\n"
                        f"EXCERPT (part {i + 1} of {len(chunks)}):\n---\n{chunk}\n---"
                    ),
                }
            ]
        )
        condensed_parts.append(response)
        if i < len(chunks) - 1:
            time.sleep(CHUNK_DELAY_SECONDS)

    return "\n\n".join(condensed_parts)


def build_crew(source_text: str) -> Crew:
    llm = build_llm()
    source_text = condense_if_needed(source_text, llm)
    analyst, writer = build_agents(llm)
    tasks = build_tasks(analyst, writer, source_text)
    return Crew(
        agents=[analyst, writer],
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )
