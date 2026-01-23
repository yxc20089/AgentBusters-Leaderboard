"""
Unit tests for BizFinBench.v2 dataset provider.
"""

import pytest
from pathlib import Path

from cio_agent.local_datasets import BizFinBenchProvider, BaseJSONLProvider
from cio_agent.models import TaskCategory, TaskDifficulty


class TestBizFinBenchProvider:
    """Test BizFinBenchProvider functionality."""

    @pytest.fixture
    def bizfin_dir(self):
        """Return path to BizFinBench.v2 directory."""
        return Path("data/BizFinBench.v2")

    def test_list_task_types(self):
        """Test that all task types are available."""
        task_types = BizFinBenchProvider.list_task_types()
        assert len(task_types) == 7
        assert "financial_quantitative_computation" in task_types
        assert "event_logic_reasoning" in task_types
        assert "stock_price_predict" in task_types

    def test_list_task_types_english(self):
        """Test that English task types are available."""
        en_tasks = BizFinBenchProvider.list_task_types("en")
        assert len(en_tasks) == 7
        assert "event_logic_reasoning" in en_tasks

    def test_list_task_types_chinese(self):
        """Test that Chinese task types are available."""
        cn_tasks = BizFinBenchProvider.list_task_types("cn")
        assert len(cn_tasks) == 7
        assert "event_logic_reasoning" in cn_tasks

    def test_list_task_types_by_language(self):
        """Test list_task_types_by_language returns both languages."""
        by_lang = BizFinBenchProvider.list_task_types_by_language()
        assert "en" in by_lang
        assert "cn" in by_lang
        assert len(by_lang["en"]) == 7
        assert len(by_lang["cn"]) == 7

    def test_task_category_mapping(self):
        """Test that all task types have category mappings."""
        for task_type in BizFinBenchProvider.list_task_types():
            assert task_type in BizFinBenchProvider.TASK_CATEGORY_MAP
            category = BizFinBenchProvider.TASK_CATEGORY_MAP[task_type]
            assert isinstance(category, TaskCategory)

    def test_task_difficulty_mapping(self):
        """Test that all task types have difficulty mappings."""
        for task_type in BizFinBenchProvider.list_task_types():
            assert task_type in BizFinBenchProvider.TASK_DIFFICULTY_MAP
            difficulty = BizFinBenchProvider.TASK_DIFFICULTY_MAP[task_type]
            assert isinstance(difficulty, TaskDifficulty)

    def test_invalid_task_type_raises_error(self):
        """Test that invalid task type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown task type"):
            BizFinBenchProvider(
                base_path="data/BizFinBench.v2",
                task_type="invalid_task",
                language="en",
            )

    def test_invalid_language_raises_error(self):
        """Test that invalid language raises ValueError."""
        with pytest.raises(ValueError, match="Invalid language"):
            BizFinBenchProvider(
                base_path="data/BizFinBench.v2",
                task_type="event_logic_reasoning",
                language="fr",
            )

    @pytest.mark.skipif(
        not Path("data/BizFinBench.v2").exists(),
        reason="BizFinBench.v2 dataset not available",
    )
    def test_load_event_logic_reasoning(self, bizfin_dir):
        """Test loading event_logic_reasoning task type."""
        provider = BizFinBenchProvider(
            base_path=bizfin_dir,
            task_type="event_logic_reasoning",
            language="en",
            limit=3,
        )
        examples = provider.load()
        
        assert len(examples) == 3
        for ex in examples:
            assert ex.question
            assert ex.answer
            assert ex.category == TaskCategory.QUALITATIVE_RETRIEVAL
            assert ex.difficulty == TaskDifficulty.MEDIUM

    @pytest.mark.skipif(
        not Path("data/BizFinBench.v2").exists(),
        reason="BizFinBench.v2 dataset not available",
    )
    def test_load_financial_quantitative_computation(self, bizfin_dir):
        """Test loading financial_quantitative_computation task type."""
        provider = BizFinBenchProvider(
            base_path=bizfin_dir,
            task_type="financial_quantitative_computation",
            language="en",
            limit=3,
        )
        examples = provider.load()
        
        assert len(examples) == 3
        for ex in examples:
            assert ex.question
            assert ex.answer
            assert ex.category == TaskCategory.NUMERICAL_REASONING
            assert ex.difficulty == TaskDifficulty.MEDIUM

    @pytest.mark.skipif(
        not Path("data/BizFinBench.v2").exists(),
        reason="BizFinBench.v2 dataset not available",
    )
    def test_to_templates(self, bizfin_dir):
        """Test converting examples to FAB templates."""
        provider = BizFinBenchProvider(
            base_path=bizfin_dir,
            task_type="event_logic_reasoning",
            language="en",
            limit=2,
        )
        templates = provider.to_templates()
        
        assert len(templates) == 2
        for tpl in templates:
            assert tpl.template_id.startswith("bizfinbench_")
            assert tpl.category == TaskCategory.QUALITATIVE_RETRIEVAL
            assert tpl.template  # question text
            assert tpl.rubric

    @pytest.mark.skipif(
        not Path("data/BizFinBench.v2").exists(),
        reason="BizFinBench.v2 dataset not available",
    )
    def test_all_task_types_loadable(self, bizfin_dir):
        """Test that all 9 task types can be loaded."""
        for task_type in BizFinBenchProvider.list_task_types():
            # financial_report_analysis is cn only
            language = "cn" if task_type == "financial_report_analysis" else "en"
            
            provider = BizFinBenchProvider(
                base_path=bizfin_dir,
                task_type=task_type,
                language=language,
                limit=1,
            )
            examples = provider.load()
            assert len(examples) >= 1, f"Failed to load {task_type}"


class TestBaseJSONLProvider:
    """Test BaseJSONLProvider base class."""

    def test_abstract_methods_raise_not_implemented(self, tmp_path):
        """Test that abstract methods raise NotImplementedError."""
        # Create a temp JSONL file
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"test": "data"}\n')
        
        provider = BaseJSONLProvider(jsonl_file, limit=1)
        
        # Test that abstract methods raise NotImplementedError
        with pytest.raises(NotImplementedError):
            provider._extract_question({"test": "data"})
        
        with pytest.raises(NotImplementedError):
            provider._extract_answer({"test": "data"})

    def test_parse_jsonl_with_invalid_json(self, tmp_path):
        """Test that invalid JSON lines are skipped."""
        jsonl_file = tmp_path / "test.jsonl"
        jsonl_file.write_text('{"valid": "json"}\ninvalid json\n{"also": "valid"}\n')
        
        provider = BaseJSONLProvider(jsonl_file)
        items = list(provider._parse_jsonl())
        
        # Should have 2 valid items, skipping the invalid one
        assert len(items) == 2

    def test_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        provider = BaseJSONLProvider("/nonexistent/path.jsonl")
        
        with pytest.raises(FileNotFoundError):
            list(provider._parse_jsonl())
