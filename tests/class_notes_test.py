from unittest.mock import MagicMock

import pytest

from buzz.class_notes import ClassNotesGenerator, split_transcript


def test_split_transcript_keeps_paragraphs_together():
    transcript = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    assert split_transcript(transcript, max_characters=35) == [
        "First paragraph.\n\nSecond paragraph.",
        "Third paragraph.",
    ]


def test_split_transcript_returns_empty_list_for_blank_input():
    assert split_transcript("  \n\n ") == []


def test_generate_summarizes_sections_then_builds_study_guide(monkeypatch):
    client = MagicMock()
    responses = iter(["Section one", "Section two", "# Class notes\nFinal notes"])
    client.chat.completions.create.side_effect = [
        MagicMock(choices=[MagicMock(message=MagicMock(content=content))])
        for content in responses
    ]
    monkeypatch.setattr("buzz.class_notes.OpenAI", lambda **_: client)
    monkeypatch.setattr("buzz.class_notes.split_transcript", lambda _: ["first", "second"])
    generator = ClassNotesGenerator("test-key")

    notes = generator.generate("First section.\n\nSecond section.")

    assert notes.markdown == "# Class notes\nFinal notes"
    assert client.chat.completions.create.call_count == 3
    final_system_prompt = client.chat.completions.create.call_args_list[-1].kwargs["messages"][0]["content"]
    assert "numbered problems" in final_system_prompt
    assert "Exam-ready takeaways" in final_system_prompt


def test_generate_rejects_empty_transcript(monkeypatch):
    monkeypatch.setattr("buzz.class_notes.OpenAI", MagicMock())

    with pytest.raises(ValueError, match="Record or paste"):
        ClassNotesGenerator("test-key").generate("")
