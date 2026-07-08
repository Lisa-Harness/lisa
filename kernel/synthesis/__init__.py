"""Kernel synthesis — post-collection intelligence pipeline contracts.

Runtime-neutral contracts for extension-owned synthesis stages live here.
The synthesis pipeline runs after intake (connectors, sources, knowledge)
has collected raw data and produces analysis, synthesis, and memory
artifacts through a staged chain.
"""

from kernel.synthesis.runtime import StageExecuteContext, StageExecutionHostHooks, StageResult
from kernel.synthesis.spec import (
    ExtensionIntelligenceSpec,
    IntelligenceStageKind,
    IntelligenceStageSpec,
    validate_intelligence_spec,
)

__all__ = [
    "ExtensionIntelligenceSpec",
    "IntelligenceStageKind",
    "IntelligenceStageSpec",
    "StageExecuteContext",
    "StageExecutionHostHooks",
    "StageResult",
    "validate_intelligence_spec",
]
