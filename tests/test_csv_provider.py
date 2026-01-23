"""
Tests for the CSV dataset provider.
"""

import csv
import tempfile
from pathlib import Path

import pytest

from cio_agent.local_datasets.csv_provider import (
    CsvFinanceDatasetProvider,
    REQUIRED_COLUMNS,
    _map_difficulty,
    _parse_rubric,
)
from cio_agent.models import TaskCategory, TaskDifficulty


class TestMapDifficulty:
    """Tests for difficulty mapping from expert time."""

    def test_easy_difficulty(self):
        """Expert time <= 5 minutes should be EASY."""
        assert _map_difficulty(1) == TaskDifficulty.EASY
        assert _map_difficulty(5) == TaskDifficulty.EASY

    def test_medium_difficulty(self):
        """Expert time 6-15 minutes should be MEDIUM."""
        assert _map_difficulty(6) == TaskDifficulty.MEDIUM
        assert _map_difficulty(15) == TaskDifficulty.MEDIUM

    def test_hard_difficulty(self):
        """Expert time 16-30 minutes should be HARD."""
        assert _map_difficulty(16) == TaskDifficulty.HARD
        assert _map_difficulty(30) == TaskDifficulty.HARD

    def test_expert_difficulty(self):
        """Expert time > 30 minutes should be EXPERT."""
        assert _map_difficulty(31) == TaskDifficulty.EXPERT
        assert _map_difficulty(60) == TaskDifficulty.EXPERT


class TestParseRubric:
    """Tests for rubric JSON parsing."""

    def test_empty_rubric(self):
        """Empty rubric should return empty lists."""
        criteria, penalties = _parse_rubric("")
        assert criteria == []
        assert penalties == []

    def test_required_type(self):
        """type='required' should be parsed as required criteria."""
        rubric = '[{"type": "required", "criteria": "Must identify growth trend"}]'
        criteria, penalties = _parse_rubric(rubric)
        assert criteria == ["Must identify growth trend"]
        assert penalties == []

    def test_penalty_type(self):
        """type='penalty' should be parsed as penalty conditions."""
        rubric = '[{"type": "penalty", "criteria": "Should not confuse metrics"}]'
        criteria, penalties = _parse_rubric(rubric)
        assert criteria == []
        assert penalties == ["Should not confuse metrics"]

    def test_mixed_types(self):
        """Both required and penalty types should be parsed correctly."""
        rubric = '[{"type": "required", "criteria": "Check revenue"}, {"type": "penalty", "criteria": "Avoid wrong data"}]'
        criteria, penalties = _parse_rubric(rubric)
        assert criteria == ["Check revenue"]
        assert penalties == ["Avoid wrong data"]

    def test_multiple_required(self):
        """Multiple required criteria should all be captured."""
        rubric = '[{"type": "required", "criteria": "First"}, {"type": "required", "criteria": "Second"}]'
        criteria, penalties = _parse_rubric(rubric)
        assert criteria == ["First", "Second"]
        assert penalties == []

    def test_invalid_json_falls_back(self):
        """Invalid JSON should fall back to raw string."""
        rubric = "This is not valid JSON"
        criteria, penalties = _parse_rubric(rubric, row_index=0)
        assert criteria == [rubric]
        assert penalties == []

    def test_missing_criteria_skipped(self):
        """Items without criteria should be skipped."""
        rubric = '[{"type": "required"}, {"type": "required", "criteria": "Valid"}]'
        criteria, penalties = _parse_rubric(rubric)
        assert criteria == ["Valid"]

    def test_missing_type_treated_as_required(self):
        """Items without type should be treated as required (default behavior)."""
        rubric = '[{"criteria": "No type"}, {"type": "required", "criteria": "Has type"}]'
        criteria, penalties = _parse_rubric(rubric, row_index=0)
        assert criteria == ["No type", "Has type"]

    def test_unknown_type_skipped(self):
        """Unknown type values should be skipped with warning."""
        rubric = '[{"type": "unknown", "criteria": "Skip this"}, {"type": "required", "criteria": "Keep this"}]'
        criteria, penalties = _parse_rubric(rubric, row_index=0)
        assert criteria == ["Keep this"]
        assert penalties == []


class TestCsvFinanceDatasetProvider:
    """Tests for CSV dataset provider."""

    def _create_csv(self, rows: list[dict], path: Path) -> None:
        """Helper to create a test CSV file."""
        if not rows:
            path.write_text("")
            return
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_valid_csv_loads_successfully(self, tmp_path):
        """Valid CSV with required columns should load."""
        csv_path = tmp_path / "test.csv"
        self._create_csv(
            [
                {
                    "Question": "What is the revenue?",
                    "Answer": "$1M",
                    "Question Type": "Quantitative Retrieval",
                    "Expert time (mins)": "10",
                    "Rubric": "[]",
                }
            ],
            csv_path,
        )

        provider = CsvFinanceDatasetProvider(csv_path)
        examples = provider.load()

        assert len(examples) == 1
        assert examples[0].question == "What is the revenue?"
        assert examples[0].answer == "$1M"
        assert examples[0].category == TaskCategory.QUANTITATIVE_RETRIEVAL
        assert examples[0].difficulty == TaskDifficulty.MEDIUM

    def test_missing_required_column_raises(self, tmp_path):
        """Missing required column should raise ValueError."""
        csv_path = tmp_path / "test.csv"
        self._create_csv(
            [
                {
                    "Question": "What is the revenue?",
                    # Missing "Question Type"
                }
            ],
            csv_path,
        )

        provider = CsvFinanceDatasetProvider(csv_path)
        with pytest.raises(ValueError) as exc_info:
            provider.load()

        assert "missing required columns" in str(exc_info.value).lower()
        assert "Question Type" in str(exc_info.value)

    def test_empty_csv_raises(self, tmp_path):
        """Empty CSV should raise ValueError."""
        csv_path = tmp_path / "test.csv"
        csv_path.write_text("")

        provider = CsvFinanceDatasetProvider(csv_path)
        with pytest.raises(ValueError) as exc_info:
            provider.load()

        assert "empty" in str(exc_info.value).lower()

    def test_to_templates_converts_correctly(self, tmp_path):
        """to_templates should convert examples to FABQuestionTemplate."""
        csv_path = tmp_path / "test.csv"
        self._create_csv(
            [
                {
                    "Question": "Analyze the trends",
                    "Answer": "Growing",
                    "Question Type": "Trends",
                    "Expert time (mins)": "25",
                    "Rubric": "[]",
                }
            ],
            csv_path,
        )

        provider = CsvFinanceDatasetProvider(csv_path)
        templates = provider.to_templates()

        assert len(templates) == 1
        assert templates[0].template == "Analyze the trends"
        assert templates[0].category == TaskCategory.TRENDS
        assert templates[0].difficulty == TaskDifficulty.HARD

    def test_unknown_question_type_defaults(self, tmp_path):
        """Unknown question type should default to QUALITATIVE_RETRIEVAL."""
        csv_path = tmp_path / "test.csv"
        self._create_csv(
            [
                {
                    "Question": "Unknown type question",
                    "Question Type": "Unknown Category",
                    "Expert time (mins)": "5",
                }
            ],
            csv_path,
        )

        provider = CsvFinanceDatasetProvider(csv_path)
        examples = provider.load()

        assert examples[0].category == TaskCategory.QUALITATIVE_RETRIEVAL

    def test_missing_expert_time_defaults_to_medium(self, tmp_path):
        """Missing expert time should default to 10 (MEDIUM difficulty)."""
        csv_path = tmp_path / "test.csv"
        self._create_csv(
            [
                {
                    "Question": "Test question",
                    "Question Type": "Trends",
                    # Missing "Expert time (mins)"
                }
            ],
            csv_path,
        )

        provider = CsvFinanceDatasetProvider(csv_path)
        examples = provider.load()

        assert examples[0].difficulty == TaskDifficulty.MEDIUM
