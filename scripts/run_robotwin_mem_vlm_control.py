#!/usr/bin/env python3
"""Strong frozen-VLM oracle control for RoboTwin-MeM dataset admission.

This is a fixed action-ranking head, not CEM.  It uses the official
Qwen3-VL-4B-Instruct backbone named in EventVLA, the same prompt/controller for
every memory condition, and action-code calibration from ordinary training
demonstration outcomes.  Keyframe metadata is used only by explicitly named
oracle test conditions.  No event label or event time is used in calibration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_robotwin_mem_admission as base  # noqa: E402
from lewm.envs.robotwin_mem import (  # noqa: E402
    ALL_MEMORY_CONDITIONS,
    CAMERA_KEYS,
    DEFAULT_MEMORY_BUDGET,
    TASK_SPECS,
)


MODEL_REPOSITORY = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
MODEL_LICENSE = "Apache-2.0"
DEFAULT_MODEL = (
    base.OUTPUT / "external/Qwen3-VL-4B-Instruct"
)
CONTROL_SEEDS = base.MODEL_SEEDS


def _strict_values(
    text: str,
    *,
    key: str,
    count: int,
    candidates: int,
) -> list[int] | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*\[([^\]]*)\]', text)
    if match is None:
        return None
    values = [
        int(value)
        for value in re.findall(r"-?\d+", match.group(1))
    ]
    if len(values) != int(count):
        return None
    if any(value < 0 or value >= int(candidates) for value in values):
        return None
    return values


def _panel_cache(
    dataset_root: Path,
    task_id: str,
    row: Mapping[str, Any],
    extra_indices: Sequence[int] = (),
) -> dict[int, Image.Image]:
    indices = {
        int(value)
        for value in row["encoding_frame_indices"]
        if int(value) >= 0
    }
    indices.update(
        int(value)
        for value in extra_indices
        if 0 <= int(value) < int(row["length"])
    )
    selected = sorted(indices)
    decoded = [
        base._decode_selected_frames(
            base._episode_video(
                dataset_root,
                task_id,
                int(row["episode_index"]),
                camera,
            ),
            selected,
        )
        for camera in CAMERA_KEYS
    ]
    return {
        index: Image.fromarray(
            np.concatenate(
                [view[index] for view in decoded],
                axis=1,
            )
        )
        for index in selected
    }


def _null_panel() -> Image.Image:
    return Image.fromarray(np.zeros((480, 640 * len(CAMERA_KEYS), 3), np.uint8))


def _calibration_examples(
    *,
    task_id: str,
    rows: Sequence[Mapping[str, Any]],
    dataset_root: Path,
    seed: int,
    examples_per_code: int = 3,
) -> list[dict[str, Any]]:
    """Choose one ordinary demonstrated action outcome per candidate code."""

    candidates = TASK_SPECS[task_id].action_candidates
    rng = np.random.default_rng(int(seed))
    by_code: dict[int, list[tuple[Mapping[str, Any], int]]] = {
        code: [] for code in range(candidates)
    }
    for row in rows:
        if row["split"] != "train":
            continue
        for query_position, code in enumerate(
            row["action_candidate_sequence"]
        ):
            by_code[int(code)].append((row, query_position))
    examples = []
    for code in range(candidates):
        choices = by_code[code]
        if not choices:
            raise RuntimeError(f"{task_id}: no train action for code {code}")
        count = min(int(examples_per_code), len(choices))
        selected = rng.choice(len(choices), size=count, replace=False)
        for selected_index in np.asarray(selected).reshape(-1):
            row, query_position = choices[int(selected_index)]
            before = int(row["query_steps"][query_position])
            after = min(int(row["length"]) - 1, before + 45)
            after_late = min(int(row["length"]) - 1, before + 70)
            panels = _panel_cache(
                dataset_root,
                task_id,
                row,
                extra_indices=(before, after, after_late),
            )
            examples.append(
                {
                    "code": code,
                    "episode_index": int(row["episode_index"]),
                    "query_position": int(query_position),
                    "before_step": before,
                    "after_step": after,
                    "after_late_step": after_late,
                    "before": panels[before],
                    "after": panels[after],
                    "after_late": panels[after_late],
                }
            )
    return examples


def _physical_candidates(task_id: str) -> str:
    if task_id == "pick_the_unhidden_block":
        return (
            "the four always-visible colored blocks along the front row, "
            "numbered 0,1,2,3 from image-left to image-right"
        )
    if task_id == "pick_objects_in_order":
        return (
            "the three always-visible objects along the front row, numbered "
            "0,1,2 from image-left to image-right"
        )
    if task_id == "cover_blocks_hard":
        return (
            "the four brown covers, numbered 0,1,2,3 from image-left to "
            "image-right"
        )
    raise ValueError(task_id)


def _infer_codebook(
    *,
    model: Any,
    processor: Any,
    task_id: str,
    calibration: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, int], list[dict[str, Any]]]:
    """Infer action-code to spatial-position mapping from train outcomes."""

    from scipy.optimize import linear_sum_assignment

    candidates = TASK_SPECS[task_id].action_candidates
    scores = np.zeros((candidates, candidates), dtype=np.float64)
    receipts = []
    for example in calibration:
        content = [
            {
                "type": "text",
                "text": (
                    "This is an ordinary successful training action. Each "
                    "panel concatenates the untouched official head, "
                    "left-wrist, and right-wrist RGB views."
                ),
            },
            {"type": "text", "text": "Immediately before the action:"},
            {"type": "image", "image": example["before"]},
            {"type": "text", "text": "45 control steps after the action began:"},
            {"type": "image", "image": example["after"]},
            {"type": "text", "text": "70 control steps after the action began:"},
            {"type": "image", "image": example["after_late"]},
            {
                "type": "text",
                "text": (
                    f"Determine which physical candidate was acted on: "
                    f"{_physical_candidates(task_id)}. Rank every candidate "
                    "from most likely acted-on position to least likely. Return "
                    "strict JSON only with each position exactly once: "
                    f'{{"position_ranking":['
                    f'{",".join(["x"] * candidates)}]}}.'
                ),
            },
        ]
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You identify the spatial target of demonstrated "
                            "robot actions from before/after RGB observations."
                        ),
                    }
                ],
            },
            {"role": "user", "content": content},
        ]
        values, raw, retried = _predict(
            model=model,
            processor=processor,
            messages=messages,
            query_count=candidates,
            candidates=candidates,
            output_key="position_ranking",
        )
        ranking = (
            None
            if values is None or len(set(values)) != candidates
            else [int(value) for value in values]
        )
        if ranking is not None:
            for rank, position in enumerate(ranking):
                scores[int(example["code"]), position] += candidates - rank
        receipts.append(
            {
                "code": int(example["code"]),
                "episode_index": int(example["episode_index"]),
                "query_position": int(example["query_position"]),
                "position_ranking": ranking,
                "raw_response": raw,
                "schema_retry": retried,
            }
        )
    if np.any(scores.sum(axis=1) == 0):
        raise RuntimeError(f"{task_id}: no valid visual codebook rankings")
    code_rows, positions = linear_sum_assignment(-scores)
    code_to_position = {
        int(code): int(position)
        for code, position in zip(code_rows, positions)
    }
    receipts.append(
        {
            "assignment_score_matrix": scores.tolist(),
            "assignment_method": (
                "maximum-weight one-to-one Hungarian assignment over frozen-VLM "
                "rankings from ordinary action outcomes"
            ),
        }
    )
    return code_to_position, receipts


def _task_query(task_id: str) -> str:
    if task_id == "pick_the_unhidden_block":
        return (
            "The cover-inspection phase is finished. Identify the three small "
            "colored cubes transiently exposed behind the moved brown covers "
            "(ignore the four always-visible front-row cubes). In the current "
            "panel, choose the front-row cube whose color was absent from the "
            "hidden cubes. Return exactly ONE demonstrated action code."
        )
    if task_id == "pick_objects_in_order":
        return (
            "The cover-inspection phase is finished. The memory panels "
            "transiently reveal the objects under the left, middle, and right "
            "covers in that chronological order. In the current panel, the "
            "three matching objects are visible along the front row. First "
            "transcribe hidden_objects and front_objects independently. Then, "
            "for each hidden object in order, find the equal name in "
            "front_objects and output that front-list index. Do not output "
            "[0,1,2] unless the two transcribed lists are actually identical. "
            "Return exactly THREE physical positions."
        )
    if task_id == "cover_blocks_hard":
        return (
            "The first inspection pass is finished. The chronological memory "
            "panels reveal the block colors under the leftmost, left-middle, "
            "right-middle, and rightmost covers. Transcribe those four colors "
            "as slot_colors in left-to-right order. Then output the index of "
            "red in slot_colors, followed by the indices of green, blue, and "
            "yellow. Return exactly FOUR physical positions."
        )
    raise ValueError(task_id)


def _reasoning_schema(task_id: str) -> str:
    if task_id == "pick_the_unhidden_block":
        return (
            '{"hidden_colors":["color","color","color"],'
            '"front_colors":["color","color","color","color"],'
            '"positions":[x]}'
        )
    if task_id == "pick_objects_in_order":
        return (
            '{"hidden_objects":["object","object","object"],'
            '"front_objects":["object","object","object"],'
            '"positions":[x,x,x]}'
        )
    if task_id == "cover_blocks_hard":
        return (
            '{"slot_colors":["color","color","color","color"],'
            '"positions":[x,x,x,x]}'
        )
    raise ValueError(task_id)


def _evidence_vocabulary(task_id: str) -> str:
    if task_id == "pick_the_unhidden_block":
        return (
            "Replace every color placeholder with an observed member of "
            "{red, green, blue, yellow}; never emit the literal word color."
        )
    if task_id == "pick_objects_in_order":
        return (
            "Replace every object placeholder with an observed member of "
            "{toy car, mouse, stapler, playing cards}; never emit the literal "
            "word object."
        )
    if task_id == "cover_blocks_hard":
        return (
            "Replace every color placeholder with an observed member of "
            "{red, green, blue, yellow}; never emit the literal word color."
        )
    raise ValueError(task_id)


def _controller_prompt(
    *,
    task_id: str,
    memory_panels: Sequence[Image.Image],
    current_panel: Image.Image,
    proprio: Sequence[float],
) -> list[dict[str, Any]]:
    spec = TASK_SPECS[task_id]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "Test episode memory follows in chronological order. Each panel "
                "concatenates the untouched official head, left-wrist, and "
                "right-wrist RGB views. A black panel is an explicit "
                "null-memory slot and carries no evidence."
            ),
        }
    ]
    for position, panel in enumerate(memory_panels):
        content.extend(
            [
                {
                    "type": "text",
                    "text": f"Test memory slot {position + 1}:",
                },
                {"type": "image", "image": panel},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": "Current delayed-query observation:"},
            {"type": "image", "image": current_panel},
            {
                "type": "text",
                "text": (
                    f"Official task instruction: {spec.instruction}\n"
                    f"Current 14D proprioception: "
                    f"{[round(float(value), 5) for value in proprio]}\n"
                    f"{_task_query(task_id)}\n"
                    f"Physical candidates are {_physical_candidates(task_id)}. "
                    "First transcribe the relevant visual evidence inside the "
                    "JSON, then derive the physical positions. Output strict "
                    f"JSON only with no prose using this schema: "
                    f"{_reasoning_schema(task_id)}. "
                    f"{_evidence_vocabulary(task_id)} Replace every x with a "
                    "derived integer; never copy placeholder tokens. "
                    f"The positions array must contain exactly "
                    f"{spec.query_count} integers."
                ),
            },
        ]
    )
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a frozen visual robot action-ranking function. "
                        "Never invent observations. Obey output cardinality and "
                        "return JSON only."
                    ),
                }
            ],
        },
        {"role": "user", "content": content},
    ]


def _predict(
    *,
    model: Any,
    processor: Any,
    messages: Sequence[Mapping[str, Any]],
    query_count: int,
    candidates: int,
    output_key: str = "positions",
) -> tuple[list[int] | None, str, bool]:
    import torch

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=96,
            do_sample=False,
            use_cache=True,
        )
    text = processor.batch_decode(
        generated[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )[0]
    actions = _strict_values(
        text,
        key=output_key,
        count=query_count,
        candidates=candidates,
    )
    retried = False
    if actions is None:
        retried = True
        retry = list(messages) + [
            {"role": "assistant", "content": [{"type": "text", "text": text}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Your output violated the schema. Return only "
                            f'{{"{output_key}":['
                            f'{",".join(["x"] * query_count)}]}} '
                            f"with exactly {query_count} valid integers and no "
                            "other keys or prose."
                        ),
                    }
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            retry,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=48,
                do_sample=False,
                use_cache=True,
            )
        text = processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )[0]
        actions = _strict_values(
            text,
            key=output_key,
            count=query_count,
            candidates=candidates,
        )
    return actions, text, retried


def _score(
    actions: Sequence[int] | None,
    target: Sequence[int],
    *,
    candidates: int,
) -> dict[str, Any]:
    target_array = np.asarray(target, dtype=np.int64)
    valid = actions is not None and len(actions) == len(target_array)
    if valid:
        prediction = np.asarray(actions, dtype=np.int64)
    else:
        prediction = np.full_like(target_array, -1)
    correct = prediction == target_array
    # The VLM emits ranks rather than calibrated logits. Store a finite,
    # explicit rank surrogate instead of fabricating probabilities.
    reciprocal = np.where(correct, 1.0, 1.0 / int(candidates))
    return {
        "prediction": prediction.astype(int).tolist(),
        "target": target_array.astype(int).tolist(),
        "per_query_correct": correct.tolist(),
        "per_query_accuracy": float(correct.mean()),
        "exact_success": bool(correct.all()),
        "mean_reciprocal_rank": float(reciprocal.mean()),
        "cross_entropy": None,
        "valid_schema": bool(valid),
    }


def _memory_indices(
    row: Mapping[str, Any],
    condition: str,
    oracle_event_position: int | None = None,
) -> list[int]:
    if condition == "oracle_best_event":
        if oracle_event_position is None:
            raise ValueError("oracle event position required")
        event = int(row["keyframe_steps"][oracle_event_position])
        return [event] + [-1] * (DEFAULT_MEMORY_BUDGET - 1)
    return [int(value) for value in row["selections"][condition]]


def _run_task(args: argparse.Namespace) -> None:
    base.assert_gpu_contract(require_visible=True)
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    task_id = args.task
    spec = TASK_SPECS[task_id]
    manifest = base.read_json(args.output / "episode_manifest.json")
    rows = manifest["tasks"][task_id]["episodes"]
    test_rows = [row for row in rows if row["split"] == "test"]
    if len(test_rows) != base.TEST_EPISODES:
        raise RuntimeError("unexpected test episode count")
    excluded = {int(value) for value in args.exclude_episodes}
    test_rows = [
        row for row in test_rows if int(row["episode_index"]) not in excluded
    ]
    if args.max_test_episodes > 0:
        test_rows = test_rows[: int(args.max_test_episodes)]
    registration = base.read_json(
        args.output / "vlm_control_protocol_registration.json"
    )
    registered = registration["tasks"][task_id]
    if registered["script_sha256"] != base.sha256_file(Path(__file__)):
        raise RuntimeError("VLM control changed after protocol registration")
    if registered["excluded_development_episodes"] != sorted(excluded):
        raise RuntimeError("excluded episodes differ from registered protocol")
    if registered["confirmatory_test_episodes"] != [
        int(row["episode_index"]) for row in test_rows
    ]:
        raise RuntimeError("confirmatory episodes differ from registration")
    if int(registered["max_visual_tokens"]) != int(args.max_visual_tokens):
        raise RuntimeError("visual token budget differs from registration")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=int(args.min_visual_tokens) * 28 * 28,
        max_pixels=int(args.max_visual_tokens) * 28 * 28,
    )
    output_rows = []
    calibration_receipts = {}
    null = _null_panel()
    for seed in args.control_seeds:
        calibration = _calibration_examples(
            task_id=task_id,
            rows=rows,
            dataset_root=args.dataset_root,
            seed=seed,
        )
        code_to_position, codebook_inference = _infer_codebook(
            model=model,
            processor=processor,
            task_id=task_id,
            calibration=calibration,
        )
        position_to_code = {
            int(position): int(code)
            for code, position in code_to_position.items()
        }
        calibration_receipts[str(seed)] = {
            "examples": [
                {
                    key: value
                    for key, value in example.items()
                    if key not in {"before", "after", "after_late"}
                }
                for example in calibration
            ],
            "code_to_physical_position": {
                str(code): position
                for code, position in code_to_position.items()
            },
            "inference": codebook_inference,
        }
        for episode_number, row in enumerate(test_rows, start=1):
            bank = base._feature_bank(
                args.output, task_id, int(row["episode_index"])
            )
            cache = _panel_cache(
                args.dataset_root,
                task_id,
                row,
            )
            current = cache[int(row["query_steps"][0])]
            episode_result = {
                "task": task_id,
                "model_seed": seed,
                "episode_index": int(row["episode_index"]),
                "episode_seed": int(row["episode_seed"]),
                "conditions": {},
            }

            def evaluate(
                condition: str, event_position: int | None = None
            ) -> dict[str, Any]:
                indices = _memory_indices(row, condition, event_position)
                panels = [null if index < 0 else cache[index] for index in indices]
                messages = _controller_prompt(
                    task_id=task_id,
                    memory_panels=panels,
                    current_panel=current,
                    proprio=bank["state"],
                )
                positions, raw, retried = _predict(
                    model=model,
                    processor=processor,
                    messages=messages,
                    query_count=spec.query_count,
                    candidates=spec.action_candidates,
                    output_key="positions",
                )
                actions = (
                    None
                    if positions is None
                    else [position_to_code[int(position)] for position in positions]
                )
                scored = _score(
                    actions,
                    row["action_candidate_sequence"],
                    candidates=spec.action_candidates,
                )
                scored.update(
                    {
                        "raw_response": raw,
                        "schema_retry": retried,
                        "predicted_physical_positions": positions,
                        "code_to_physical_position": code_to_position,
                        "memory_indices": indices,
                    }
                )
                return scored

            for condition in ALL_MEMORY_CONDITIONS:
                if condition == "oracle_best_event":
                    candidates = [
                        evaluate(condition, event_position)
                        for event_position in range(len(row["keyframe_steps"]))
                    ]
                    best = max(
                        range(len(candidates)),
                        key=lambda position: (
                            int(candidates[position]["exact_success"]),
                            candidates[position]["per_query_accuracy"],
                            candidates[position]["mean_reciprocal_rank"],
                            -position,
                        ),
                    )
                    result = dict(candidates[best])
                    result["oracle_event_position"] = best
                    result["oracle_candidate_results"] = [
                        dict(candidate) for candidate in candidates
                    ]
                else:
                    result = evaluate(condition)
                episode_result["conditions"][condition] = result
            episode_result["recent_suffix_probe"] = dict(
                episode_result["conditions"]["recent_only"]
            )
            output_rows.append(episode_result)
            print(
                f"[{task_id} seed={seed}] "
                f"{episode_number}/{len(test_rows)} episode="
                f"{row['episode_index']} recent="
                f"{int(episode_result['conditions']['recent_only']['exact_success'])} "
                f"oracle="
                f"{int(episode_result['conditions']['oracle_event_set']['exact_success'])}",
                flush=True,
            )
    receipt = {
        "task": task_id,
        "controller": {
            "kind": "frozen multimodal action-ranking head",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "model_license": MODEL_LICENSE,
            "model_path": str(args.model),
            "model_config_sha256": base.sha256_file(args.model / "config.json"),
            "processor_config_sha256": base.sha256_file(
                args.model / "preprocessor_config.json"
            ),
            "control_seeds": [int(value) for value in args.control_seeds],
            "seed_definition": (
                "seeded choice of ordinary train action-outcome calibration "
                "example per action code"
            ),
            "same_prompt_and_model_across_conditions": True,
            "event_labels_or_times_used_for_calibration": False,
            "task_state_labels_used": False,
            "manual_crop_or_saliency": False,
            "multiview_serialization": (
                "head/left-wrist/right-wrist concatenated without cropping"
            ),
            "min_visual_tokens": int(args.min_visual_tokens),
            "max_visual_tokens": int(args.max_visual_tokens),
            "excluded_development_episodes": sorted(excluded),
            "confirmatory_test_episodes": [
                int(row["episode_index"]) for row in test_rows
            ],
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "calibration": calibration_receipts,
        "rows": output_rows,
    }
    base.write_json(
        args.output / f"predictions_vlm_{task_id}.json", receipt
    )
    print(f"completed frozen-VLM oracle control for {task_id}")


def _register(args: argparse.Namespace) -> None:
    manifest = base.read_json(args.output / "episode_manifest.json")
    test_rows = [
        row
        for row in manifest["tasks"][args.task]["episodes"]
        if row["split"] == "test"
    ]
    excluded = {int(value) for value in args.exclude_episodes}
    confirmatory = [
        int(row["episode_index"])
        for row in test_rows
        if int(row["episode_index"]) not in excluded
    ]
    if args.max_test_episodes > 0:
        confirmatory = confirmatory[: int(args.max_test_episodes)]
    path = args.output / "vlm_control_protocol_registration.json"
    registration = (
        base.read_json(path)
        if path.exists()
        else {
            "protocol": "robotwin-mem-frozen-vlm-control-v1",
            "status": "secondary confirmatory control after DINO admission failure",
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "model_license": MODEL_LICENSE,
            "conditions": list(ALL_MEMORY_CONDITIONS),
            "tasks": {},
        }
    )
    registration["tasks"][args.task] = {
        "script_sha256": base.sha256_file(Path(__file__)),
        "model_config_sha256": base.sha256_file(args.model / "config.json"),
        "control_seeds": [int(value) for value in args.control_seeds],
        "excluded_development_episodes": sorted(excluded),
        "confirmatory_test_episodes": confirmatory,
        "development_exclusion_reason": (
            "prompt/controller inspected on these episodes before freeze"
        ),
        "memory_budget_raw_frames": DEFAULT_MEMORY_BUDGET,
        "same_model_prompt_and_candidates_across_conditions": True,
        "max_visual_tokens": int(args.max_visual_tokens),
        "min_visual_tokens": int(args.min_visual_tokens),
        "primary_metric": "exact delayed-query action-candidate sequence",
        "gate": base.protocol_receipt()["gate"],
    }
    base.write_json(path, registration)
    print(
        f"registered frozen-VLM confirmatory control for {args.task}: "
        f"{confirmatory}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASK_SPECS), required=True)
    parser.add_argument("--output", type=Path, default=base.OUTPUT)
    parser.add_argument("--dataset-root", type=Path, default=base.DATASET_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--min-visual-tokens", type=int, default=32)
    parser.add_argument("--max-visual-tokens", type=int, default=128)
    parser.add_argument(
        "--control-seeds",
        nargs="+",
        type=int,
        default=list(CONTROL_SEEDS),
    )
    parser.add_argument("--max-test-episodes", type=int, default=0)
    parser.add_argument("--exclude-episodes", nargs="*", type=int, default=[])
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.dataset_root = args.dataset_root.resolve()
    args.model = args.model.resolve()
    return args


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.register_only:
        _register(args)
    else:
        _run_task(args)
    print(f"elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
