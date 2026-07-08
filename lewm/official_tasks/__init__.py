"""Task contracts designed for the released LeWM host.

These tasks use the released Reacher observation/action cadence without
modifying the legacy V19 task registry.  Public names are semantic because
the resulting artifacts are intended to be readable in a paper and review.
"""

from lewm.official_tasks.shell_game_capacity import (
    CAPACITY_STAGES,
    CounterfactualAuditError,
    OfficialHostBaseBatch,
    ShellGameAdmissionInputs,
    ShellGameCapacityBatch,
    ShellGameCapacityContract,
    ShellGameCapacityStage,
    audit_paired_counterfactual,
    build_admission_inputs,
    gather_sequence,
    get_capacity_stage,
    paired_counterfactual_batches,
    require_paired_counterfactual,
)
from lewm.official_tasks.shell_game_admission import (
    ShellGameAdmissionThresholds,
    ShellGameFrozenFeatures,
    evaluate_frozen_admission,
    evaluate_frozen_admission_inputs,
    frozen_feature_inputs,
)

__all__ = [
    "CAPACITY_STAGES",
    "CounterfactualAuditError",
    "OfficialHostBaseBatch",
    "ShellGameAdmissionInputs",
    "ShellGameAdmissionThresholds",
    "ShellGameCapacityBatch",
    "ShellGameCapacityContract",
    "ShellGameCapacityStage",
    "ShellGameFrozenFeatures",
    "audit_paired_counterfactual",
    "build_admission_inputs",
    "evaluate_frozen_admission",
    "evaluate_frozen_admission_inputs",
    "frozen_feature_inputs",
    "gather_sequence",
    "get_capacity_stage",
    "paired_counterfactual_batches",
    "require_paired_counterfactual",
]
