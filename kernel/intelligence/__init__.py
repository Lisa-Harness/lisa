"""Kernel intelligence contracts.

Runtime-neutral contracts for extension-owned intelligence specs live here.
"""

from kernel.intelligence.runtime import StageExecuteContext, StageExecutionHostHooks, StageResult
from kernel.intelligence.spec import (
    ExtensionIntelligenceSpec,
    IntelligenceStageKind,
    IntelligenceStageSpec,
    validate_intelligence_spec,
)

__all__ = [
    "ExtensionIntelligenceSpec",
    "StageExecuteContext",
    "StageExecutionHostHooks",
    "StageResult",
    "IntelligenceStageKind",
    "IntelligenceStageSpec",
    "validate_intelligence_spec",
]
