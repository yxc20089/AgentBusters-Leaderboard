"""
Evaluation components for CIO-Agent FAB++ system.

This module provides the hierarchical evaluation framework:
- MacroEvaluator: Strategic reasoning assessment
- FundamentalEvaluator: Data accuracy validation
- ExecutionEvaluator: Action quality assessment
- OptionsEvaluator: Options trading task assessment

Dataset-specific evaluators:
- BaseDatasetEvaluator: Abstract base for dataset evaluators
- BizFinBenchEvaluator: For BizFinBench.v2 dataset
- PublicCsvEvaluator: For public.csv (FAB++) dataset
- OptionsEvaluator: Options trading task assessment
"""

from evaluators.macro import MacroEvaluator
from evaluators.fundamental import FundamentalEvaluator
from evaluators.execution import ExecutionEvaluator
from evaluators.cost_tracker import CostTracker, LLMCallRecord
from evaluators.options import OptionsEvaluator, OptionsScore, OPTIONS_CATEGORIES

# Core evaluators (require structlog and other dependencies)
_core_evaluators_available = False
try:
    from evaluators.macro import MacroEvaluator
    from evaluators.fundamental import FundamentalEvaluator
    from evaluators.execution import ExecutionEvaluator
    from evaluators.cost_tracker import CostTracker, LLMCallRecord
    _core_evaluators_available = True
except ImportError:
    MacroEvaluator = None
    FundamentalEvaluator = None
    ExecutionEvaluator = None
    CostTracker = None
    LLMCallRecord = None

# Dataset-specific evaluators (minimal dependencies)
from evaluators.base import BaseDatasetEvaluator, EvalResult
from evaluators.bizfinbench_evaluator import BizFinBenchEvaluator
from evaluators.public_csv_evaluator import PublicCsvEvaluator

# Build __all__ dynamically
__all__ = [
    "MacroEvaluator",
    "FundamentalEvaluator",
    "ExecutionEvaluator",
    "OptionsEvaluator",
    "OptionsScore",
    "OPTIONS_CATEGORIES",
    "CostTracker",
    "LLMCallRecord",
    "BaseDatasetEvaluator",
    "EvalResult",
    "BizFinBenchEvaluator",
    "PublicCsvEvaluator",
]

# Only export core evaluators if they were successfully imported
if _core_evaluators_available:
    __all__.extend([
        "MacroEvaluator",
        "FundamentalEvaluator",
        "ExecutionEvaluator",
        "CostTracker",
        "LLMCallRecord",
    ])
