"""Explicit paired cases for tasks 1/2; no model fitting or GPU execution.

Every case contains the same parameter keys.  Inactive fields retain their
documented defaults, so the same physical baseline appearing in both tasks
can be recognized without changing its scientific interpretation.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path


CASE_KEYS = (
    "id", "task", "question", "moving_radius", "eps_final", "boundary_norm",
    "perturb_kind", "logD_delta", "region_width", "C_cut", "C_smooth",
)
PHYSICS_KEYS = tuple(key for key in CASE_KEYS if key not in ("id", "task"))
DEFAULTS = dict(eps_final=0.0, boundary_norm="current_dry_density",
                perturb_kind="none", logD_delta=0.0, region_width=0.1,
                C_cut=0.2, C_smooth=0.02)


def _signed(value):
    return ("m" if value < 0 else "p" if value > 0 else "z") + f"{round(abs(value) * 1000):03d}"


def _make(task, question, moving_radius, **changes):
    case = dict(task=int(task), question=int(question), moving_radius=bool(moving_radius), **DEFAULTS)
    case.update(changes)
    geometry = "mov" if case["moving_radius"] else "fix"
    boundary = "cur" if case["boundary_norm"] == "current_dry_density" else "ini"
    base = f"t{task}_q{question}_{geometry}_{boundary}"
    if task == 1:
        identifier = f"{base}_eps_{_signed(case['eps_final'])}"
    elif case["perturb_kind"] == "none":
        identifier = f"{base}_baseline"
    else:
        identifier = f"{base}_{case['perturb_kind']}_d_{_signed(case['logD_delta'])}"
        if case["perturb_kind"] in ("outer", "inner"):
            identifier += f"_w{round(case['region_width'] * 1000):03d}"
        if case["perturb_kind"] == "lowC":
            identifier += f"_c{round(case['C_cut'] * 1000):03d}_s{round(case['C_smooth'] * 1000):03d}"
    case["id"] = identifier
    return {key: case[key] for key in CASE_KEYS}


def physical_key(case):
    """Hashable identity excluding reporting id/task; no physics is discarded."""
    return tuple(case[key] for key in PHYSICS_KEYS)


def validate_cases(cases):
    """Reject ambiguous ids, inconsistent task parameters and invalid numbers."""
    identifiers = set()
    for case in cases:
        if set(case) != set(CASE_KEYS):
            raise ValueError(f"Case keys mismatch: {case.get('id', '<unnamed>')}")
        if not isinstance(case["id"], str) or not case["id"] or case["id"] in identifiers:
            raise ValueError("Every case must have a unique nonempty id")
        identifiers.add(case["id"])
        if case["task"] not in (1, 2) or case["question"] not in (3, 4):
            raise ValueError(f"Unknown task/question: {case['id']}")
        if not isinstance(case["moving_radius"], bool):
            raise ValueError("moving_radius must be a boolean")
        if case["boundary_norm"] not in ("current_dry_density", "initial_dry_density"):
            raise ValueError("Unknown boundary normalization")
        if case["perturb_kind"] not in ("none", "outer", "inner", "lowC", "global"):
            raise ValueError("Unknown diffusion perturbation kind")
        for key in ("eps_final", "logD_delta", "region_width", "C_cut", "C_smooth"):
            if not math.isfinite(float(case[key])):
                raise ValueError(f"{key} must be finite")
        if abs(case["eps_final"]) > 0.15 + 1e-12:
            raise ValueError("This study restricts |eps_final| to at most 0.15")
        if not case["moving_radius"] and case["eps_final"] != 0:
            raise ValueError("Fixed-radius controls use eps_final=0")
        if not 0 < case["region_width"] < 1 or case["C_cut"] <= 0.03 or case["C_smooth"] <= 0:
            raise ValueError("Invalid region/window width")
        if case["task"] == 1 and (case["perturb_kind"] != "none" or case["logD_delta"] != 0):
            raise ValueError("Task 1 must not simultaneously perturb diffusivity")
        if case["task"] == 2 and (case["eps_final"] != 0 or case["boundary_norm"] != "current_dry_density"):
            raise ValueError("Task 2 retains the affine current-density boundary baseline")
        if (case["perturb_kind"] == "none") != (case["logD_delta"] == 0):
            raise ValueError("A baseline must use perturb_kind='none' and logD_delta=0")
    return cases


def make_cases(suite="pilot", task=None):
    """Return 27 pilot cases or 107 full cases (optional task filtering).

    Full task 2 has eight extra low-C smoothing-width diagnostics.  Their
    C_smooth is 0.01 or 0.04; all primary cases use the default 0.02.
    Positive/negative changes are additive in log(D), so the multiplier is
    exp(logD_delta * region_weight), NOT 1 + logD_delta * region_weight.
    """
    if suite not in ("pilot", "full"):
        raise ValueError("suite must be 'pilot' or 'full'")
    if task not in (None, 1, 2):
        raise ValueError("task must be None, 1 or 2")
    cases = []
    eps_values = (-0.10, 0.0, 0.10) if suite == "pilot" else (-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
    for boundary in ("current_dry_density", "initial_dry_density"):
        for epsilon in eps_values:
            cases.append(_make(1, 4, True, eps_final=epsilon, boundary_norm=boundary))
    # The fourth affine anchor, Q4 measured/current/eps=0, is already present.
    for question, moving in ((3, False), (3, True), (4, False)):
        cases.append(_make(1, question, moving))

    deltas = (-0.10, 0.10) if suite == "pilot" else (-0.10, -0.05, 0.05, 0.10)
    widths = (0.10,) if suite == "pilot" else (0.05, 0.10, 0.20)
    cutoffs = (0.20,) if suite == "pilot" else (0.15, 0.20, 0.30)
    for question, moving in ((3, False), (4, True)):
        cases.append(_make(2, question, moving))
        for kind in ("outer", "inner"):
            for width in widths:
                for delta in deltas:
                    cases.append(_make(2, question, moving, perturb_kind=kind,
                                       logD_delta=delta, region_width=width))
        for cutoff in cutoffs:
            for delta in deltas:
                cases.append(_make(2, question, moving, perturb_kind="lowC",
                                   logD_delta=delta, C_cut=cutoff))
        for delta in deltas:
            cases.append(_make(2, question, moving, perturb_kind="global", logD_delta=delta))
        if suite == "full":
            for smooth in (0.01, 0.04):
                for delta in (-0.10, 0.10):
                    cases.append(_make(2, question, moving, perturb_kind="lowC",
                                       logD_delta=delta, C_cut=0.20, C_smooth=smooth))
    validate_cases(cases)
    expected = 27 if suite == "pilot" else 107
    if len(cases) != expected:
        raise AssertionError(f"Internal design count is {len(cases)}, expected {expected}")
    return cases if task is None else [case for case in cases if case["task"] == task]


def describe_case(case):
    geometry = "实测半径" if case["moving_radius"] else "固定半径"
    if case["task"] == 1:
        boundary = "当前干密度" if case["boundary_norm"] == "current_dry_density" else "初始干密度"
        return f"Q{case['question']} / {geometry} / ε末值={case['eps_final']:+.2f} / {boundary}边界口径"
    return (f"Q{case['question']} / {geometry} / {case['perturb_kind']} / "
            f"ΔlogD={case['logD_delta']:+.2f} / w={case['region_width']:.2f} / "
            f"C_cut={case['C_cut']:.2f} / 平滑宽度={case['C_smooth']:.2f}")


def design_summary(cases):
    return {"reported_cases": len(cases), "unique_physical_cases": len({physical_key(c) for c in cases}),
            "by_task": dict(Counter(c["task"] for c in cases)),
            "by_question": dict(Counter(c["question"] for c in cases))}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--task", type=int, choices=(1, 2))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = make_cases(args.suite, args.task)
    payload = {"suite": args.suite, "summary": design_summary(cases), "cases": cases}
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps(payload["summary"], ensure_ascii=False))
    else:
        print(serialized)
