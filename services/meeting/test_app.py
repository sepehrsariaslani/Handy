"""Fast, model-free checks for alignment and source validation."""
import app as meeting


def test_word_to_speaker_assignment():
    words = [
        {"start": 0.1, "end": 0.4, "text": "Hello"},
        {"start": 0.5, "end": 0.8, "text": " there"},
        {"start": 1.2, "end": 1.5, "text": "Yes"},
    ]
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
    assigned = meeting._assign_speakers(words, turns)
    assert [item["speaker"] for item in assigned] == [
        "SPEAKER_00", "SPEAKER_00", "SPEAKER_01"
    ]
    grouped = meeting._utterances(assigned)
    assert len(grouped) == 2
    assert grouped[0]["id"] == "s1"
    assert grouped[0]["text"] == "Hello there"
    assert grouped[1]["speaker"] == "SPEAKER_01"


def test_overlapping_word_uses_largest_overlap():
    words = [{"start": 0.9, "end": 1.4, "text": "agreed"}]
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")]
    assert meeting._assign_speakers(words, turns)[0]["speaker"] == "SPEAKER_01"


def test_summary_refuses_unsupported_actions(monkeypatch):
    def fake_model(_, __):
        return {
            "summary": "Summary",
            "decisions": [],
            "tasks": [
                {"text": "Call the client", "source_ids": ["s1"], "assignee": None, "due_date": None},
                {"text": "Invented task", "source_ids": ["s999"], "assignee": "Someone", "due_date": None},
            ],
            "open_questions": [],
        }

    monkeypatch.setattr(meeting, "_chat_json", fake_model)
    analysis = meeting._summarize(
        [{"id": "s1", "speaker": "SPEAKER_00", "start": 0.0, "text": "Call the client"}]
    )
    assert len(analysis["tasks"]) == 1
    assert analysis["tasks"][0]["source_ids"] == ["s1"]
    assert analysis["tasks"][0]["assignee"] is None


def test_empty_meeting_is_empty():
    assert meeting._summarize([])["tasks"] == []
