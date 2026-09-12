"""NumPy/CUDA implementation gate with on-site tight SciPy BDF references.

This is a same-grid time integration / implementation check, not spatial
convergence or experimental validation. A CUDA request never falls back to CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import traceback

import numpy as np

from gpu_solver import block_pcr, colored_jacobian, select_backend, solve_batch
from physics import Inputs, make_grid, prepare_cases
from cpu_solver import solve_case


CASES = [
    {"id": "q3_baseline", "question": 3},
    {"id": "q4_baseline", "question": 4},
    {"id": "q4_nonaffine_initial_density", "question": 4, "eps_final": 0.15,
     "boundary_norm": "initial_dry_density"},
    {"id": "q4_lowC_perturbation", "question": 4, "perturb_kind": "lowC", "logD_delta": 0.1},
]


def host(xp, value):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def sanitize(value):
    if isinstance(value, dict):
        return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
    if isinstance(value, np.generic):
        return sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def source_fingerprint():
    directory = Path(__file__).resolve().parent
    names = ("gpu_solver.py", "physics.py", "cpu_solver.py", "validate_gpu.py", "data/inputs.npz")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names}


def linear_algebra_checks(xp):
    rng = np.random.default_rng(8911)
    records = []
    for n in (3, 4, 7, 17, 65):
        a = rng.normal(size=(2, n, 2, 2)) * 0.05
        c = rng.normal(size=a.shape) * 0.05
        b = np.broadcast_to(3 * np.eye(2), a.shape).copy() + rng.normal(size=a.shape) * 0.05
        a[:, 0] = 0
        c[:, -1] = 0
        d = rng.normal(size=(2, n, 2))
        answer = host(xp, block_pcr(xp.asarray(a), xp.asarray(b), xp.asarray(c), xp.asarray(d), xp))
        maximum_error = maximum_residual = 0.0
        for batch in range(2):
            matrix = np.zeros((2 * n, 2 * n))
            for i in range(n):
                matrix[2*i:2*i+2, 2*i:2*i+2] = b[batch, i]
                if i > 0:
                    matrix[2*i:2*i+2, 2*i-2:2*i] = a[batch, i]
                if i + 1 < n:
                    matrix[2*i:2*i+2, 2*i+2:2*i+4] = c[batch, i]
            reference = np.linalg.solve(matrix, d[batch].ravel())
            maximum_error = max(maximum_error, float(np.max(np.abs(answer[batch].ravel() - reference))))
            maximum_residual = max(maximum_residual, float(np.max(np.abs(matrix @ answer[batch].ravel() - d[batch].ravel()))))
        records.append({"name": f"block_PCR_vs_dense_N{n}", "max_solution_error": maximum_error,
                        "max_residual": maximum_residual,
                        "passed": maximum_error < 1e-10 and maximum_residual < 1e-10})
    n = 17
    y = rng.uniform(0.2, 1.5, size=(2, n, 2))
    a = rng.normal(size=(2, n, 2, 2)) * 0.1
    b = rng.normal(size=a.shape)
    c = rng.normal(size=a.shape) * 0.1
    a[:, 0] = 0
    c[:, -1] = 0
    ag, bg, cg = (xp.asarray(v) for v in (a, b, c))
    def function(t, state):
        out = xp.einsum("bnij,bnj->bni", bg, state) + 0.02 * state**2
        out[:, 1:] += xp.einsum("bnij,bnj->bni", ag[:, 1:], state[:, :-1])
        out[:, :-1] += xp.einsum("bnij,bnj->bni", cg[:, :-1], state[:, 1:])
        return out
    yg = xp.asarray(y)
    lower, diagonal, upper = (host(xp, v) for v in colored_jacobian(function, 0.0, yg, function(0.0, yg), xp))
    expected_diagonal = b.copy()
    expected_diagonal[..., 0, 0] += 0.04 * y[..., 0]
    expected_diagonal[..., 1, 1] += 0.04 * y[..., 1]
    error = max(float(np.max(np.abs(lower - a))), float(np.max(np.abs(diagonal - expected_diagonal))),
                float(np.max(np.abs(upper - c))))
    records.append({"name": "six_color_nonlinear_cross_component_jacobian", "max_error": error,
                    "passed": error < 1e-6})
    return records


def analytic_event_check(backend):
    class AnalyticInputs:
        boundary_end = 5.0
        radius_end = 20.0
    def decay(t, y, cases, grid, inputs, xp, phase):
        result = xp.zeros_like(y)
        result[..., 0] = -1000 * (y[..., 0] - 30.0)
        result[..., 1] = -0.2 * y[..., 1]
        return result
    y0 = np.empty((2, 5, 2))
    y0[..., 0], y0[..., 1] = 28.0, 0.5
    result = solve_batch({"question": np.array([3, 4])}, {"a": np.linspace(0, 1, 5), "weights": np.ones(5)},
                         AnalyticInputs(), backend=backend, rhs_fn=decay, y0=y0, t_end=10.0,
                         sample_times=np.arange(11.0), rtol=1e-5, atol=(1e-7, 1e-9), event_time_tol=0.01)
    exact = -np.log(0.15 / 0.5) / 0.2
    error = float(np.max(np.abs(result.t_critical - exact)))
    width = float(np.nanmax(np.diff(result.max_event_brackets, axis=1)))
    no_samples_after_stop = bool(np.all(~result.sample_valid[result.sample_times[None, :] > result.final_times[:, None]]))
    return {"name": "stiff_analytic_decay_event", "exact_event_s": float(exact),
            "maximum_event_error_s": error, "maximum_bracket_width_s": width,
            "no_samples_after_stop": no_samples_after_stop,
            "passed": bool(error < 0.005 and width <= 0.0100001 and no_samples_after_stop and result.stats["failed_cases"] == 0)}


def physical_comparison(backend, N, end, name):
    inputs = Inputs()
    references = []
    for case in CASES:
        print(f"现场CPU BDF参照：{name} {case['id']} N={N}", flush=True)
        references.append(solve_case(case, N=N, t_end=end, inputs=inputs, rtol=2e-9, atol=2e-11, profile_count=31))
    sample_times = np.unique(np.concatenate([profile["t_s"] for _, profile in references]))
    if len(sample_times) > 500:
        raise RuntimeError("Validation sample plan exceeded solver output limit")
    result = solve_batch(prepare_cases(CASES), make_grid(N), inputs, backend=backend, t_end=end,
                         sample_times=sample_times, rtol=2e-5, atol=(1e-5, 1e-7),
                         event_time_tol=30.0, max_step=300.0)
    rows = []
    for i, (case, (reference, profile)) in enumerate(zip(CASES, references)):
        index = np.searchsorted(sample_times, profile["t_s"])
        valid = result.sample_valid[i, index]
        if valid.sum() < 20:
            raise RuntimeError("Too few valid overlapping profile times")
        error_C = float(np.max(np.abs(result.samples_C[i, index[valid]] - profile["C"][valid])))
        error_T = float(np.max(np.abs(result.samples_T[i, index[valid]] - profile["T"][valid])))
        reference_tc, reference_tm = reference["tcritical_s"], reference["tmean_s"]
        reached = np.isfinite(result.t_critical[i])
        match_max_status = bool(reached == (reference_tc is not None))
        match_mean_status = bool(np.isfinite(result.t_mean_threshold[i]) == (reference_tm is not None))
        delta_tc = float(result.t_critical[i] - reference_tc) if reference_tc is not None and reached else None
        delta_tm = float(result.t_mean_threshold[i] - reference_tm) if reference_tm is not None and np.isfinite(result.t_mean_threshold[i]) else None
        passed = (match_max_status and match_mean_status and error_C < 1e-3 and error_T < 0.02
                  and (delta_tc is None or abs(delta_tc) < 60.0)
                  and (delta_tm is None or abs(delta_tm) < 60.0)
                  and not str(result.status[i]).startswith("failed"))
        row = {"name": name + "/" + case["id"], "case": case, "N": N,
               "BDF_rtol": 2e-9, "BDF_atol": 2e-11, "SDIRK_rtol": 2e-5,
               "comparison_profile_times": int(valid.sum()), "max_C_profile_error": error_C,
               "max_T_profile_error_K": error_T, "BDF_tcritical_s": reference_tc,
               "SDIRK_tcritical_s": result.t_critical[i], "delta_tcritical_s": delta_tc,
               "BDF_tmean_s": reference_tm, "SDIRK_tmean_s": result.t_mean_threshold[i],
               "delta_tmean_s": delta_tm, "max_event_status_matches": match_max_status,
               "mean_event_status_matches": match_mean_status, "solver_status": str(result.status[i]),
               "reference_status": reference["status"], "passed": bool(passed)}
        rows.append(row)
        print(json.dumps(sanitize(row), ensure_ascii=False), flush=True)
    return rows, result.stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("numpy", "cupy"), default="numpy")
    parser.add_argument("--N", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("validation_gpu"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "validation_report.json"
    started = perf_counter()
    report = {"passed": False, "cuda_executed": False, "requested_backend": args.backend,
              "N": args.N, "created_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "machine": platform.machine(),
              "source_and_input_sha256": source_fingerprint(),
              "scope": "same-grid implementation/time-integration gate; not spatial convergence or experimental validation",
              "thresholds": {"event_error_s": 60.0, "C_profile_error": 1e-3, "T_profile_error_K": 0.02},
              "tests": [], "solver_stats": {}}
    def save():
        report["wall_time_s"] = perf_counter() - started
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(sanitize(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
    try:
        xp, device = select_backend(args.backend)
        report["device"] = device
        report["cuda_executed"] = device["cuda_executed"]
        report["tests"].extend(linear_algebra_checks(xp))
        report["tests"].append(analytic_event_check(args.backend))
        save()
        for name, end in (("short_1800s", 1800.0), ("full_72h_or_event", 72 * 3600.0)):
            rows, stats = physical_comparison(args.backend, args.N, end, name)
            report["tests"].extend(rows)
            report["solver_stats"][name] = stats
            save()
        report["passed"] = all(row["passed"] for row in report["tests"])
        if report["source_and_input_sha256"] != source_fingerprint():
            report["passed"] = False
            raise RuntimeError("Source/input files changed during validation; rerun on frozen files")
        if not report["passed"]:
            report["error"] = "One or more preflight checks failed; production GPU execution is not approved by this gate."
        save()
    except Exception as error:
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        save()
    print(f"Validation report: {destination.resolve()} | passed={report['passed']} | cuda_executed={report['cuda_executed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
