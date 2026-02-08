"""Tests for the extract_json() function in server.py."""

import json
import pytest
from server import extract_json


class TestExtractJsonValidInput:
    """Tests for valid JSON inputs."""

    def test_plain_json_object(self):
        text = '{"number": 1, "result": "correct", "front": "Q?", "back": "A.", "notes": "", "tags": ["a"]}'
        result = extract_json(text)
        assert result["number"] == 1
        assert result["result"] == "correct"
        assert result["front"] == "Q?"

    def test_json_with_whitespace(self):
        text = '  \n  {"number": 1, "result": "correct"}  \n  '
        result = extract_json(text)
        assert result["number"] == 1

    def test_json_with_nested_objects(self):
        text = '{"outer": {"inner": "value"}, "list": [1, 2, 3]}'
        result = extract_json(text)
        assert result["outer"]["inner"] == "value"
        assert result["list"] == [1, 2, 3]

    def test_json_with_special_characters(self):
        text = r'{"front": "What is the \"answer\"?", "back": "Line1\nLine2"}'
        result = extract_json(text)
        assert "answer" in result["front"]

    def test_json_with_unicode(self):
        text = '{"front": "What is the m\\u00fcller muscle?", "tags": []}'
        result = extract_json(text)
        assert "müller" in result["front"]

    def test_json_with_null_values(self):
        text = '{"number": null, "result": "correct", "notes": null}'
        result = extract_json(text)
        assert result["number"] is None
        assert result["notes"] is None

    def test_json_with_boolean_values(self):
        text = '{"active": true, "deleted": false}'
        result = extract_json(text)
        assert result["active"] is True
        assert result["deleted"] is False

    def test_json_with_empty_arrays(self):
        text = '{"tags": [], "notes": ""}'
        result = extract_json(text)
        assert result["tags"] == []


class TestExtractJsonMarkdownWrapped:
    """Tests for JSON wrapped in markdown code blocks."""

    def test_json_in_json_code_block(self):
        text = '```json\n{"number": 5, "result": "incorrect"}\n```'
        result = extract_json(text)
        assert result["number"] == 5
        assert result["result"] == "incorrect"

    def test_json_in_generic_code_block(self):
        text = '```\n{"number": 5, "result": "correct"}\n```'
        result = extract_json(text)
        assert result["number"] == 5

    def test_json_code_block_with_surrounding_text(self):
        text = 'Here is the result:\n```json\n{"number": 3, "result": "correct"}\n```\nDone!'
        result = extract_json(text)
        assert result["number"] == 3

    def test_json_code_block_with_extra_whitespace(self):
        text = '```json\n  {"number": 7, "result": "incorrect"}  \n```'
        result = extract_json(text)
        assert result["number"] == 7


class TestExtractJsonEmbeddedInText:
    """Tests for JSON embedded in explanatory text."""

    def test_json_with_prefix_text(self):
        text = 'Based on the transcription, here is the parsed data: {"number": 12, "result": "correct", "front": "Q?", "back": "A."}'
        result = extract_json(text)
        assert result["number"] == 12

    def test_json_with_suffix_text(self):
        text = '{"number": 8, "result": "incorrect", "front": "Q?", "back": "A."}\n\nI hope this helps!'
        result = extract_json(text)
        assert result["number"] == 8

    def test_json_with_both_prefix_and_suffix(self):
        text = 'Result:\n{"number": 15, "result": "correct"}\nEnd of response.'
        result = extract_json(text)
        assert result["number"] == 15


class TestExtractJsonTrailingCommas:
    """Tests for JSON with trailing commas (common LLM mistake)."""

    def test_trailing_comma_before_closing_brace(self):
        text = '{"number": 1, "result": "correct",}'
        result = extract_json(text)
        assert result["number"] == 1

    def test_trailing_comma_before_closing_bracket(self):
        text = '{"tags": ["a", "b", "c",]}'
        result = extract_json(text)
        assert result["tags"] == ["a", "b", "c"]

    def test_multiple_trailing_commas(self):
        text = '{"number": 1, "tags": ["a", "b",],}'
        result = extract_json(text)
        assert result["number"] == 1
        assert result["tags"] == ["a", "b"]


class TestExtractJsonMissingQuotes:
    """Tests for JSON with missing opening quotes (common LLM error)."""

    def test_missing_opening_quote_on_front(self):
        text = '{"number": 1, "result": "correct", "front": What is the question?", "back": "Answer."}'
        result = extract_json(text)
        assert result["number"] == 1
        assert "What is the question?" in result["front"]

    def test_missing_opening_quote_on_back(self):
        text = '{"number": 2, "result": "incorrect", "front": "Question?", "back": The answer is X.", "notes": ""}'
        result = extract_json(text)
        assert "The answer is X." in result["back"]

    def test_missing_opening_quote_on_notes(self):
        text = '{"number": 3, "result": "correct", "front": "Q?", "back": "A.", "notes": Some detailed notes here."}'
        result = extract_json(text)
        assert "Some detailed notes" in result["notes"]

    def test_missing_opening_quote_on_section(self):
        text = '{"number": 5, "result": "correct", "front": "Q?", "back": "A.", "section": Upper Limb"}'
        result = extract_json(text)
        assert result["section"] == "Upper Limb"


class TestExtractJsonMalformedInput:
    """Tests for malformed or empty input."""

    def test_empty_string_raises(self):
        with pytest.raises(Exception):
            extract_json('')

    def test_whitespace_only_raises(self):
        with pytest.raises(Exception):
            extract_json('   \n\t  ')

    def test_no_json_raises(self):
        with pytest.raises(Exception):
            extract_json('This is just plain text with no JSON.')

    def test_incomplete_json_raises(self):
        with pytest.raises(Exception):
            extract_json('{"number": 1, "result":')

    def test_just_opening_brace_raises(self):
        with pytest.raises(Exception):
            extract_json('{')


class TestExtractJsonFullPeresteResponse:
    """Tests with realistic Pereste Parse-style LLM responses."""

    def test_full_pereste_parse_response(self):
        response = json.dumps({
            "number": 42,
            "result": "correct",
            "front": "What nerve is most commonly injured in a midshaft humerus fracture?",
            "back": "The radial nerve. It runs in the radial groove on the posterior aspect of the humerus.",
            "notes": "The radial nerve (C5-T1) is the most commonly injured nerve in midshaft humerus fractures.\n- Runs in the spiral/radial groove\n- Presents with wrist drop\n- Loss of extension at wrist and MCP joints",
            "tags": ["radial_nerve", "humerus_fracture", "upper_limb"]
        })
        result = extract_json(response)
        assert result["number"] == 42
        assert result["result"] == "correct"
        assert "humerus fracture" in result["front"].lower()
        assert len(result["tags"]) == 3

    def test_response_with_multiline_notes(self):
        text = '{"number": 1, "result": "correct", "front": "Q?", "back": "A.", "notes": "Line1\\nLine2\\n- Bullet1\\n- Bullet2", "tags": ["anatomy"]}'
        result = extract_json(text)
        assert "Line1" in result["notes"]
        assert "Bullet2" in result["notes"]
