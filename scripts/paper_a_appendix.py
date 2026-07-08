"""Generate the result-complete Paper A appendix from validated aggregates.

The caller is responsible for validating ``summary`` and ``config`` before
invoking :func:`render_appendix`.  This module performs formatting only: it
does not read experiment files, recompute statistics, or choose claims from
unaggregated values.
"""

from __future__ import annotations

from typing import Any, Mapping


TASKS = ("t1", "t3")
TASK_NAMES = {
    "t1": "Transient-marker recall",
    "t3": "Drifting-color recall",
    "t4": "Occluded-target prediction",
}
ARMS = ("none", "gru", "lstm", "ssm", "fixed_trust")
ARM_NAMES = {
    "none": "No carrier",
    "gru": "Action-conditioned GRU",
    "lstm": "Action-conditioned LSTM",
    "ssm": "Diagonal SSM",
    "fixed_trust": "Fixed-trust predict--correct",
}
OBJECTIVES = ("one_step", "overshoot_8")
OBJECTIVE_NAMES = {
    "one_step": "One-step",
    "overshoot_8": "Eight-step overshooting",
}
TASK_LABELS = {"t1": "marker", "t3": "color"}
HORIZONS = (1, 2, 4, 8, 16)


def _number(value: Any, digits: int = 3, *, sign: bool = False) -> str:
    number = float(value)
    if abs(number) < 0.5 * 10 ** (-digits):
        number = 0.0
    return f"{number:+.{digits}f}" if sign else f"{number:.{digits}f}"


def _stat(statistic: Mapping[str, Any], digits: int = 3,
          *, sign: bool = False) -> str:
    low, high = statistic["ci95"]
    return (
        "$" + _number(statistic["mean"], digits, sign=sign)
        + r"\;[" + _number(low, digits, sign=sign) + ", "
        + _number(high, digits, sign=sign) + "]$"
    )


def _mean_sd(statistic: Mapping[str, Any], digits: int = 3) -> str:
    """Compact descriptive summary for dense, result-complete tables."""
    return (
        "$" + _number(statistic["mean"], digits)
        + r"\!\pm\!" + _number(statistic["sample_sd"], digits) + "$"
    )


def _scalar(value: Any, digits: int = 3, *, sign: bool = False) -> str:
    return "$" + _number(value, digits, sign=sign) + "$"


def _row(*values: str) -> str:
    return " & ".join(values) + r" \\"


def _carrier_result_tables(summary: Mapping[str, Any]) -> list[str]:
    frozen = summary["frozen_carrier_swap"]
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.7pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Complete frozen-carrier results. Final accuracy is the primary "
        r"causal endpoint; trajectory accuracy is an exploratory temporally "
        r"aggregated diagnostic. Arm entries are mean $\pm$ sample SD; paired "
        r"contrasts retain 95\% seed-bootstrap intervals.}",
        r"\label{tab:app-frozen-full}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}"
        r">{\raggedright\arraybackslash}p{0.19\linewidth}rrrr@{}}",
        r"\toprule",
        _row("Task", "Carrier", "Final acc.", "Trajectory acc.",
             r"$\Delta$ vs. none", "Next-latent MSE"),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        item = frozen["tasks"][task]
        for arm_index, arm in enumerate(ARMS):
            arm_item = item["arms"][arm]
            delta = ("--" if arm == "none" else
                     _stat(item["paired_vs_no_carrier"][arm], sign=True))
            lines.append(_row(
                TASK_NAMES[task] if arm_index == 0 else "",
                ARM_NAMES[arm],
                _mean_sd(arm_item["accuracy"]),
                _mean_sd(arm_item["trajectory_accuracy"]),
                delta,
                _mean_sd(arm_item["validation_next_latent_mse"], 4),
            ))
        if task_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    lines += [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Fixed-trust paired contrasts. Positive values favor "
        r"fixed-trust; task-level wins are paired model seeds, while pooled "
        r"wins count task--seed cells.}",
        r"\label{tab:app-frozen-contrasts}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        _row("Scope", "Comparator", r"Difference [95\% CI]", "Wins", "Ties"),
        r"\midrule",
    ]
    for task in TASKS:
        contrasts = frozen["tasks"][task]["paired_contrasts"]
        for index, reference in enumerate(("gru", "lstm", "ssm")):
            statistic = contrasts[reference]
            lines.append(_row(
                TASK_NAMES[task] if index == 0 else "",
                ARM_NAMES[reference], _stat(statistic, sign=True),
                f"{statistic['wins']}/{statistic['n']}",
                str(statistic["ties"]),
            ))
        lines.append(r"\addlinespace[1pt]")
    for index, reference in enumerate(("gru", "lstm", "ssm")):
        statistic = frozen["pooled_equal_task_contrasts"][reference]
        lines.append(_row(
            "Equal-task pooled" if index == 0 else "",
            ARM_NAMES[reference], _stat(statistic, sign=True),
            f"{statistic['positive_task_seed_wins']}/"
            f"{statistic['total_task_seed_pairs']}",
            str(statistic["ties"]),
        ))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\caption{Seed-level final pre-observation accuracy. The no-carrier "
        r"columns repeat by construction because that reference has no "
        r"optimized state.}",
        r"\label{tab:app-frozen-seeds}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        _row("Task", "Seed", "None", "GRU", "LSTM", "SSM", "Fixed-trust"),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        arms = frozen["tasks"][task]["arms"]
        value_maps = {
            arm: dict(zip(arms[arm]["accuracy"]["seeds"],
                          arms[arm]["accuracy"]["values"]))
            for arm in ARMS
        }
        for seed in range(5):
            lines.append(_row(
                TASK_NAMES[task] if seed == 0 else "", str(seed),
                *[_number(value_maps[arm][seed]) for arm in ARMS],
            ))
        if task_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def _context_tables(summary: Mapping[str, Any]) -> list[str]:
    context = summary["long_context"]["tasks"]
    lines = [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Complete long-context results. Raw-context accuracy is "
        r"seed-independent; predictor and MSE entries are mean $\pm$ sample "
        r"SD. Paired changes retain 95\% seed-bootstrap intervals. Cue coverage "
        r"is measured on 240 validation episodes.}",
        r"\label{tab:app-context-full}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.21\linewidth}"
        r"rrrrrY@{}}",
        r"\toprule",
        _row("Task", "$H$", "Cue frames (mean)", "Episodes with cue",
             "Raw acc.", "Predictor acc.", "Next-latent MSE"),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        for history_index, history in enumerate((3, 16, 32, 56)):
            item = context[task]["contexts"][str(history)]
            raw = item["raw_legal_context_readout"]
            coverage = raw["validation_coverage"]
            lines.append(_row(
                TASK_NAMES[task] if history_index == 0 else "", str(history),
                _number(coverage["cue_frames_reachable_mean"], 1),
                f"{coverage['cue_any_frame_reachable']}/"
                f"{coverage['episodes']}",
                _number(raw["value"]),
                _mean_sd(item["trained_predictor_semantic_accuracy"]),
                _mean_sd(item["validation_next_latent_mse"], 4),
            ))
        if task_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    lines += [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Paired long-context changes relative to $H=3$. Negative "
        r"MSE differences indicate better local prediction.}",
        r"\label{tab:app-context-contrasts}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        _row("Task", "Comparison", r"$\Delta$ predictor accuracy",
             r"$\Delta$ next-latent MSE"),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        for history in (16, 32, 56):
            item = context[task]["paired_vs_short_context"][str(history)]
            lines.append(_row(
                TASK_NAMES[task] if history == 16 else "",
                f"$H={history}$ minus $H=3$",
                _stat(item["trained_semantic_accuracy_delta"], sign=True),
                _stat(item["validation_mse_delta"], 4, sign=True),
            ))
        if task_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def _rollout_tables(summary: Mapping[str, Any]) -> list[str]:
    rollout = summary["learned_rollout"]["tasks"]
    lines: list[str] = []
    for task in TASKS:
        lines += [
            r"\begin{table}[H]",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3.0pt}",
            r"\caption{Complete learned-rollout primary diagnostics for "
            + TASK_NAMES[task]
            + r". Entries are mean [95\% seed-bootstrap CI].}",
            r"\label{tab:app-rollout-primary-" + TASK_LABELS[task] + "}",
            r"\begin{tabular}{lrrrrr}",
            r"\toprule",
            _row("Objective", "$K$", "Normalized MSE", "Copy-last MSE",
                 "Model/copy ratio", "True-action advantage"),
            r"\midrule",
        ]
        for objective_index, objective in enumerate(OBJECTIVES):
            horizons = rollout[task]["objectives"][objective]["horizons"]
            for horizon in HORIZONS:
                item = horizons[str(horizon)]
                lines.append(_row(
                    OBJECTIVE_NAMES[objective] if horizon == 1 else "",
                    str(horizon), _stat(item["normalized_latent_mse"]),
                    _stat(item["copy_last_normalized_mse"]),
                    _stat(item["model_to_copy_ratio"]),
                    _stat(item["true_action_advantage"]),
                ))
            if objective_index == 0:
                lines.append(r"\midrule")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    lines += [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.8pt}",
        r"\caption{Secondary learned-rollout diagnostics. Shuffled-action "
        r"MSE tests action dependence; pose MAE is a linear physical-state "
        r"readout. Entries are mean $\pm$ sample SD; the plotted rank ratio is "
        r"computed per seed before aggregation.}",
        r"\label{tab:app-rollout-secondary}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{0.18\linewidth}"
        r">{\raggedright\arraybackslash}p{0.18\linewidth}rrrrY@{}}",
        r"\toprule",
        _row("Task", "Objective", "$K$", "Shuffled MSE", "Pose MAE",
             "Predicted rank", "Target rank"),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        for objective_index, objective in enumerate(OBJECTIVES):
            horizons = rollout[task]["objectives"][objective]["horizons"]
            for horizon in HORIZONS:
                item = horizons[str(horizon)]
                lines.append(_row(
                    TASK_NAMES[task]
                    if objective_index == 0 and horizon == 1 else "",
                    OBJECTIVE_NAMES[objective] if horizon == 1 else "",
                    str(horizon), _mean_sd(item["shuffled_action_normalized_mse"]),
                    _mean_sd(item["pose_angular_mae"]),
                    _mean_sd(item["predicted_effective_rank"], 2),
                    _mean_sd(item["target_effective_rank"], 2),
                ))
            if objective_index == 0:
                lines.append(r"\addlinespace[1pt]")
        if task_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}", ""]

    # Place the visual immediately after the secondary table it summarizes;
    # the paired objective-contrast table then begins the following page.
    lines += [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/fig_a_appendix_rollout.pdf}",
        r"\caption{Secondary rollout diagnostics. Pose MAE is obtained from a "
        r"linear state readout; effective rank is normalized by the paired target "
        r"rank within seed. Ratios near one show no obvious global covariance-rank "
        r"collapse in this diagnostic, but do not establish cue retention or "
        r"physical correctness. The rank panel uses a zoomed vertical scale.}",
        r"\label{fig:app-rollout}",
        r"\end{figure}",
        "",
    ]

    lines += [
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\caption{Paired eight-step-overshooting minus one-step contrasts. "
        r"Negative MSE and pose differences are improvements; positive "
        r"action-advantage differences indicate stronger action dependence.}",
        r"\label{tab:app-rollout-contrasts}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        _row("Task", "$K$", r"$\Delta$ normalized MSE", r"$\Delta$ pose MAE",
             r"$\Delta$ action advantage"),
        r"\midrule",
    ]
    for task_index, task in enumerate(TASKS):
        contrasts = rollout[task]["paired_overshoot_minus_one_step"]
        for horizon in HORIZONS:
            item = contrasts[str(horizon)]
            lines.append(_row(
                TASK_NAMES[task] if horizon == 1 else "", str(horizon),
                _stat(item["normalized_latent_mse"], 4, sign=True),
                _stat(item["pose_angular_mae"], 4, sign=True),
                _stat(item["true_action_advantage"], 4, sign=True),
            ))
        if task_index == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return lines


def render_appendix(summary: Mapping[str, Any],
                    config: Mapping[str, Any]) -> str:
    """Return publication-ready Markdown/raw-LaTeX appendix content."""

    availability = summary["availability"]
    frozen = summary["frozen_carrier_swap"]
    completion = summary["completion"]
    host = config["official_host"]
    carrier_cfg = config["frozen_carrier_swap"]
    context_cfg = config["long_context"]
    rollout_cfg = config["learned_rollout"]

    params = {
        arm: int(frozen["tasks"]["t1"]["arms"][arm]["carrier_parameters"])
        for arm in ARMS
    }
    target = params["fixed_trust"]
    mismatch = {
        arm: (None if arm == "none" else abs(params[arm] - target) / target)
        for arm in ARMS
    }

    lines = [
        "## Experimental Protocol and Data Contract",
        "",
        "The appendix reports the complete validated grid behind the main-paper "
        "aggregates. It preserves semantic task names and separates primary "
        "endpoints from exploratory diagnostics. No appendix result expands the "
        "scope to memory-conditioned control.",
        "",
        r"\begin{table}[H]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Complete formal experiment matrix. Admission probes are "
        r"computed before this 86-cell training grid.}",
        r"\label{tab:app-grid}",
        r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xrrrr@{}}",
        r"\toprule",
        _row("Study", "Axes", "Seeds", "Epochs", "Cells", "Status"),
        r"\midrule",
        _row("Frozen carrier", r"2 tasks $\times$ 5 carrier variants", "5",
             str(carrier_cfg["epochs"]), "50", "Complete"),
        _row("Long context", r"2 tasks $\times$ 4 context lengths", "3",
             str(context_cfg["epochs"]), "24", "Complete"),
        _row("Learned rollout", r"2 tasks $\times$ 2 objectives", "3",
             str(rollout_cfg["epochs"]), "12", "Complete"),
        r"\midrule",
        _row("Total", "", "", "", str(completion["completed_metrics"]),
             f"{completion['completed_metrics']}/{completion['expected_metrics']}"),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\caption{Optimization and module-freezing contract. The rollout "
        r"weight decay is a launcher setting rather than a serialized cell field.}",
        r"\label{tab:app-training}",
        r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X"
        r">{\raggedright\arraybackslash}Xrrrr@{}}",
        r"\toprule",
        _row("Study", "Frozen", "Trainable", "Batch", "LR", "Weight decay", "Seeds"),
        r"\midrule",
        _row("Carrier", "Encoder--projector, action encoder, predictor, output projection",
             "Carrier only", str(carrier_cfg["batch_size"]),
             _number(carrier_cfg["learning_rate"], 4),
             _number(carrier_cfg["weight_decay"], 5), "5"),
        _row("Context", "Encoder--projector", "Action encoder, predictor, output projection",
             str(context_cfg["batch_size"]), _number(context_cfg["learning_rate"], 4),
             _number(context_cfg["weight_decay"], 3), "3"),
        _row("Rollout", "Encoder--projector", "Action encoder, predictor, output projection",
             "64", "0.0001", "0.001*", "3"),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        "All episodes contain 64 observations and five simulator actions per "
        "observation interval, flattened to one 10-D action block. Each task "
        "uses 1,200 training and 240 fixed validation episodes. The official "
        f"host has a {host['latent_dim']}-D latent and context $H={host['context']}$. "
        "The long-context batch size was amended from 64 to 256 before any "
        "context metric completed; objectives, epochs, seeds, and endpoints "
        "were unchanged.",
        "",
        "## Endpoint Legality and Statistical Contract",
        "",
        r"\begin{table}[H]",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\caption{Readout-relative claim contract. Each row identifies what "
        r"the corresponding endpoint can and cannot establish.}",
        r"\label{tab:app-endpoints}",
        r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}X"
        r">{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}",
        r"\toprule",
        _row("Question", "Legal endpoint", "Supports", "Does not support"),
        r"\midrule",
        _row("Availability", "Four frozen cue-window latents; pre-gap latents for the continuous target",
             "Linear target availability", "Complete information or downstream use"),
        _row("Final retention", "$[z_{60},z_{61},z_{62},b_{63}]$; $b_{63}$ is read before $z_{63}$",
             "Conditional final carrier effect", "End-to-end capacity or control"),
        _row("Trajectory diagnostic", "Final raw context, mean post-cue prior, and $b_{63}$",
             "Aggregate-trajectory linear decodability", "Decision-time retention"),
        _row("Raw context", "Mean and last latent in $z[q-H:q]$",
             "Finite evidence access", "Predictor exposure"),
        _row("Predictor", "One contextual prediction from legal latents/actions",
             "Output decodability", "An isolated causal bottleneck"),
        _row("Rollout", "Fixed anchor $t=24$; copy-last and shuffled-action references",
             "Action-sensitive latent dynamics", "Planning, reward, or control"),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        "The primary carrier endpoint excludes the decision observation and "
        "contains no temporal aggregation. The exploratory trajectory feature "
        "averages pre-observation priors from two frames after cue offset through "
        "$q=63$; it is intentionally a different, more permissive readout. All "
        "reported intervals are deterministic 20,000-draw percentile bootstraps "
        "over model/optimizer seeds. Carrier contrasts pair the same seed and "
        "pool tasks with equal weight; fixed data splits are never resampled.",
        "",
        "## Carrier Implementations and Parameter Matching",
        "",
        r"\begin{table}[H]",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{Carrier capacity matching. Relative mismatch is measured "
        r"against the 76,032-parameter fixed-trust carrier.}",
        r"\label{tab:app-carriers}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        _row("Carrier", "State width", "Parameters", r"$\Delta$ params",
             "Rel. mismatch", "Imagined update"),
        r"\midrule",
        _row("No carrier", "0", "0", "--", "--", "No state"),
        _row("GRU", "74", f"{params['gru']:,}", f"{params['gru']-target:+,}",
             _number(mismatch["gru"], 4), "Zero-latent, action-only"),
        _row("LSTM", "61", f"{params['lstm']:,}", f"{params['lstm']-target:+,}",
             _number(mismatch["lstm"], 4), "Zero-latent, action-only"),
        _row("Diagonal SSM", "192", f"{params['ssm']:,}",
             f"{params['ssm']-target:+,}", _number(mismatch["ssm"], 4),
             "Zero-latent, action-only"),
        _row("Fixed-trust", "192", f"{params['fixed_trust']:,}", "0", "0.0000",
             "Predict without correction"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        "All learned reads are zero-initialized and all variants use the same "
        "residual injection point. The GRU, LSTM, and SSM baselines emit a prior "
        "from the state before the current frame; fixed-trust memory emits the "
        "predictive mean before correction. The common frozen host is checked "
        "before and after every run.",
        "",
        "## Complete Frozen-Carrier Results",
        "",
    ]
    lines.extend(_carrier_result_tables(summary))
    lines += [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/fig_a_appendix_probe.pdf}",
        r"\caption{Exploratory temporal aggregation versus the primary final "
        r"causal endpoint. Differences are computed within seed. Positive values "
        r"mean that the trajectory-average feature yields higher linear-probe "
        r"accuracy than the primary final feature. Because their temporal support "
        r"and feature maps differ, this is not evidence for decision-time retention "
        r"and does not replace the main endpoint.}",
        r"\label{fig:app-probe}",
        r"\end{figure}",
        "",
        "The exploratory diagnostic does not alter the pre-specified primary "
        "endpoint, and it produces a different carrier ordering. SSM and "
        "fixed-trust trajectory-average features are substantially more linearly "
        "decodable than their final features, whose causal reads remain near chance; "
        "GRU and LSTM show little such gap. This localizes decodable signal to the "
        "aggregate post-cue trajectory, not to any particular transient state. "
        "Because temporal support and feature maps differ, this remains a descriptive "
        "mechanism diagnostic rather than a formal paired endpoint.",
        "",
        "## Complete Long-Context Controls",
        "",
    ]
    lines.extend(_context_tables(summary))
    lines += [
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=\linewidth]{figures/fig_a_appendix_context.pdf}",
        r"\caption{Local prediction loss versus delayed semantic exposure. Each "
        r"point is a separately trained context model; pale paths connect the "
        r"same model seed across context lengths. The cue first enters legal "
        r"context at $H=56$, but predictor semantic accuracy does not follow the "
        r"raw-context jump. Lines are descriptive and are not fitted trends.}",
        r"\label{fig:app-context}",
        r"\end{figure}",
        "",
        "Both tasks achieve their lowest local next-latent error at $H=32$, "
        "where no validation window contains a cue frame. At $H=56$, every "
        "validation episode contains cue evidence and raw readout rises sharply, "
        "yet predictor semantic accuracy remains near chance. The change from "
        "$H=32$ to $H=56$ therefore improves access while worsening the local "
        "MSE objective. Since models and readouts are re-estimated for each "
        "$H$, this supports a descriptive objective--semantic dissociation, not "
        "a causal decomposition.",
        "",
        "## Complete Learned-Rollout Analysis",
        "",
    ]
    lines.extend(_rollout_tables(summary))
    lines += [
        "Overshooting exhibits a horizon- and task-dependent trade-off. It "
        "increases short-horizon latent error on both tasks. For drifting-color "
        "recall it improves latent MSE from $K=4$ onward and improves pose MAE "
        "at the longest horizons. For transient-marker recall, the $K=16$ latent "
        "MSE improves while pose MAE worsens. Consequently, lower latent rollout "
        "error is not a uniform proxy for physical readout quality. Across all "
        "conditions, predicted/target effective-rank ratios stay close to one; "
        "this diagnostic shows no evidence of an obvious global rank collapse, "
        "but says nothing about the earlier cue semantics.",
        "",
        "## Reproducibility Ledger and Claim Boundary",
        "",
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Publication-artifact validation ledger. Full hashes and all "
        r"86 source-metric records are retained in the machine-readable manifest.}",
        r"\label{tab:app-ledger}",
        r"\begin{tabularx}{\linewidth}{@{}l>{\raggedright\arraybackslash}Xl@{}}",
        r"\toprule",
        _row("Check", "Validated condition", "Status"),
        r"\midrule",
        _row("Grid", f"{completion['completed_metrics']} of "
             f"{completion['expected_metrics']} metric cells present", "Pass"),
        _row("Official host", "Released source, commit, and weight SHA-256 match",
             "Pass"),
        _row("Frozen host", "Before/after state digest identical in every carrier run",
             "Pass"),
        _row("Carrier checkpoints", "Embedded metrics, state digest, and cache receipts match",
             "Pass"),
        _row("No carrier", "Repeated seed records are exactly deterministic",
             "Pass"),
        _row("Aggregation", "Seed-level statistics and 20,000-draw intervals regenerated",
             "Pass"),
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
        "The study does not establish that fixed-trust memory is superior to "
        "GRU or SSM, that the continuous target is absent from the representation, "
        "or that longer context causally identifies an objective bottleneck. "
        "The rollout prerequisite contains no reward, CEM return, or executed "
        "policy result; passing it cannot be promoted to planning or control. "
        "Intervals cover optimization/model seeds on fixed splits, not new-task "
        "or new-environment uncertainty. These boundaries apply equally to the "
        "main paper and the additional appendix diagnostics.",
        "",
        "The official host source, commit, complete weight hash, configuration "
        "record, cache manifests, and source-metric hashes are distributed with "
        "the artifact. The PDF prints the scientific contract and aggregate "
        "results; the machine-readable ledger remains the authority for exact "
        "provenance.",
    ]
    return "\n".join(lines).rstrip() + "\n"
