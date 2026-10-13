"""Create study-ready notes from a completed lecture transcript.

Audio remains in Buzz.  This module only sends the transcript text to the
configured OpenAI-compatible endpoint when the user explicitly requests notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from openai import OpenAI


DEFAULT_CLASS_NOTES_MODEL = "gpt-4o-mini"
MAX_CHUNK_CHARACTERS = 24_000


@dataclass(frozen=True)
class ClassNotes:
    """The Markdown returned by the class-notes workflow."""

    markdown: str


def split_transcript(transcript: str, max_characters: int = MAX_CHUNK_CHARACTERS) -> list[str]:
    """Split long transcripts at paragraph or word boundaries.

    A class can be much longer than a single model request.  Keeping boundaries
    readable also makes the partial summaries useful if a later request fails.
    """
    normalized = transcript.strip()
    if not normalized:
        return []
    if len(normalized) <= max_characters:
        return [normalized]

    chunks: list[str] = []
    current = ""
    for paragraph in normalized.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        candidates: Iterable[str]
        if len(paragraph) > max_characters:
            candidates = (paragraph[index:index + max_characters]
                          for index in range(0, len(paragraph), max_characters))
        else:
            candidates = (paragraph,)

        for candidate in candidates:
            joined = candidate if not current else f"{current}\n\n{candidate}"
            if current and len(joined) > max_characters:
                chunks.append(current)
                current = candidate
            else:
                current = joined
    if current:
        chunks.append(current)
    return chunks


class ClassNotesGenerator:
    """Generates faithful, editable class notes from local transcript text."""

    def __init__(self, api_key: str, model: str = DEFAULT_CLASS_NOTES_MODEL,
                 base_url: str | None = None) -> None:
        self.client = OpenAI(api_key=api_key, base_url=base_url or None, max_retries=0)
        self.model = model

    def _complete(self, system_prompt: str, user_content: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            timeout=600.0,
        )
        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("The notes service returned an empty response.")
        return content.strip()

    def generate(self, transcript: str) -> ClassNotes:
        chunks = split_transcript(transcript)
        if not chunks:
            raise ValueError("Record or paste a transcript before generating class notes.")

        partial_notes = [
            self._complete(
                "Create accurate, concise notes for this section of a class. "
                "Do not invent facts. Preserve equations, units, numbered questions, "
                "worked calculations, definitions, examples, assignments, and places "
                "where the lecturer corrects a common misconception or expresses uncertainty.",
                chunk,
            )
            for chunk in chunks
        ]
        combined = "\n\n---\n\n".join(partial_notes)
        markdown = self._complete(
            "Turn these lecture-section notes into one editable Markdown study guide. "
            "Infer a specific title from the subject. When the lecture contains numbered "
            "problems, organize the notes under a heading for each problem and show: "
            "the result, a concise explanation, formulas with units, and the worked steps. "
            "Use clear tables for comparisons when they help. Include a ## Common mistakes "
            "and corrections section only when the lecturer corrected or contrasted an idea. "
            "End with ## Exam-ready takeaways and ## Study questions. Only include "
            "assignments or deadlines explicitly present in the notes; write 'None mentioned' "
            "when there are none. Do not invent information or silently resolve uncertainty.",
            combined,
        )
        return ClassNotes(markdown=markdown)
