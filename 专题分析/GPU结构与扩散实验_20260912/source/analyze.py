"""Paired, censoring-aware summaries for tasks 1/2 (no model execution).

Region perturbations are mathematical sensitivity scenarios.  An outer-region
response is not evidence that a physical crust has formed.  Fixed material
regions and a state-dependent low-C window are reported separately.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cases import CASE_KEYS, make_cases, validate_cases


EVENTS = ("tcritical_s", "tmean_s", "tail_s")
FRACTIONS = ("perturb_support_dry_mass_fraction", "perturb_weighted_dry_mass_fraction")


def _finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _failed(row):
    status = " ".join(value.lower() for value in (row.get("status"), row.get("solver_status"))
                      if isinstance(value, str))
    return any(token in status for token in ("error", "fail", "not_run", "missing", "nan", "diverg"))


def _event(row, key):
    value = row.get(key, np.nan)
    if key in ("tcritical_s", "tail_s") and "not_reached" in str(row.get("status", "")).lower():
        return np.nan
    return float(value) if not _failed(row) and _finite(value) and float(value) >= 0 else np.nan


def _integral_average(t, values):
    return float(values[0]) if len(t) == 1 else float(np.trapezoid(values, t) / (t[-1] - t[0]))


def summarize_profile(path, case=None):
    """Integrate a saved mask with initial dry-mass weights.

    The profile weights may be raw annular integral weights (sum 1/2) or
    normalized mass fractions (sum 1).  Explicit normalization gives the same
    result in both representations.  Support means mask > 0.5, so tanh tails
    do not falsely declare the entire specimen to be the active support.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as profile:
        t = np.atleast_1d(np.asarray(profile["t_s"], dtype=float))
        a = np.asarray(profile["a"], dtype=float)
        C = np.asarray(profile["C"], dtype=float)
        T = np.asarray(profile["T"], dtype=float) if "T" in profile else None
        weights = np.asarray(profile["weights"], dtype=float)
        mask = np.asarray(profile["mask"], dtype=float)
    if t.ndim != 1 or not len(t) or np.any(~np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError(f"{path.name}: profile times must be finite and strictly increasing")
    if a.ndim != 1 or a.size < 2 or np.any(np.diff(a) <= 0) or a[0] < 0 or a[-1] > 1:
        raise ValueError(f"{path.name}: invalid material coordinates")
    if C.shape != (len(t), len(a)) or np.any(~np.isfinite(C)) or np.any(C <= 0):
        raise ValueError(f"{path.name}: invalid moisture profile")
    if T is not None and (T.shape != C.shape or np.any(~np.isfinite(T)) or np.any(T <= -273.15)):
        raise ValueError(f"{path.name}: invalid temperature profile")
    if weights.shape != a.shape or np.any(weights <= 0) or np.any(~np.isfinite(weights)):
        raise ValueError(f"{path.name}: invalid initial dry-mass weights")
    mass_weights = weights / weights.sum()
    mask = np.broadcast_to(mask, C.shape)
    if np.any(~np.isfinite(mask)) or np.any(mask < -1e-12) or np.any(mask > 1 + 1e-12):
        raise ValueError(f"{path.name}: mask must lie in [0,1]")
    weighted = mask @ mass_weights
    supported = (mask > 0.5) @ mass_weights
    result = {
        "id": case["id"] if case else path.stem,
        "profile_path": str(path), "profile_samples": len(t), "profile_nodes": len(a),
        "profile_start_s": float(t[0]), "profile_end_s": float(t[-1]),
        "profile_weighted_fraction_time_average": _integral_average(t, weighted),
        "profile_support_fraction_time_average": _integral_average(t, supported),
        "profile_weighted_fraction_initial": float(weighted[0]),
        "profile_weighted_fraction_end": float(weighted[-1]),
        "profile_support_fraction_initial": float(supported[0]),
        "profile_support_fraction_end": float(supported[-1]),
        "profile_mean_C_end": float(C[-1] @ mass_weights),
        "profile_max_C_end": float(C[-1].max()),
        "profile_mass_fraction_C_above_015_end": float((C[-1] > .15) @ mass_weights),
    }
    if T is not None:
        result.update(profile_min_T_C=float(T.min()), profile_max_T_C=float(T.max()))
    if case and case["perturb_kind"] == "none" and np.max(np.abs(mask)) > 1e-12:
        raise ValueError(f"{path.name}: a baseline profile unexpectedly has an active perturbation mask")
    return result


def _read_results(results):
    if isinstance(results, pd.DataFrame):
        return results.copy()
    if isinstance(results, (str, Path)):
        path = Path(results)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return pd.DataFrame(payload["results"] if isinstance(payload, dict) else payload)
    return pd.DataFrame(results)


def _make_joined(results, cases):
    raw = _read_results(results)
    if "id" not in raw or raw["id"].duplicated().any():
        raise ValueError("Results require a unique id column")
    design = pd.DataFrame(cases)
    extras = set(raw["id"]) - set(design["id"])
    if extras:
        raise ValueError(f"Results contain unknown case ids: {sorted(extras)}")
    # Design is authoritative for input parameters.  Reject mismatches instead
    # of silently overwriting a stale result with a new case definition.
    reference = design.set_index("id")
    for _, row in raw.iterrows():
        for key in set(CASE_KEYS) & set(raw.columns) - {"id"}:
            expected = reference.loc[row["id"], key]
            actual = row[key]
            if key == "moving_radius":
                equal = str(actual).lower() == str(expected).lower() or actual == expected
            elif isinstance(expected, (int, float, np.number)):
                equal = _finite(actual) and math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-12)
            else:
                equal = actual == expected
            if not equal:
                raise ValueError(f"Result/design mismatch for {row['id']}: {key}")
    metrics = raw.drop(columns=[k for k in CASE_KEYS if k != "id" and k in raw], errors="ignore")
    joined = design.merge(metrics, on="id", how="left", validate="one_to_one")
    if "status" not in joined:
        joined["status"] = "not_run"
    else:
        joined["status"] = joined["status"].fillna("not_run")
    for name in (*EVENTS, *FRACTIONS, "mass_audit_rel", "max_C_end", "mean_C_end", "t_end_s"):
        if name not in joined:
            joined[name] = np.nan
        joined[name] = pd.to_numeric(joined[name], errors="coerce")
    return joined


def _base_for(case, rows):
    for row in rows:
        if (row["task"] == case["task"] and row["question"] == case["question"]
                and row["moving_radius"] == case["moving_radius"]
                and row["boundary_norm"] == case["boundary_norm"]
                and row["eps_final"] == 0 and row["perturb_kind"] == "none"
                and row["logD_delta"] == 0):
            return row
    return None


def _pair(case, reference, kind):
    reference = reference or {"id": None, "status": "missing_reference"}
    output = {key: case[key] for key in CASE_KEYS}
    output.update(comparison=kind, reference_id=reference["id"],
                  candidate_status=case["status"], reference_status=reference["status"],
                  effect_definition="candidate minus reference")
    for fraction in FRACTIONS:
        output[fraction] = case.get(fraction, np.nan)
        output["reference_" + fraction] = reference.get(fraction, np.nan)
    for key in EVENTS:
        a, b = _event(case, key), _event(reference, key)
        available = np.isfinite(a) and np.isfinite(b)
        output[key] = a
        output["reference_" + key] = b
        output["delta_" + key] = a - b if available else np.nan
        output["relative_delta_" + key] = (a - b) / b if available and b > 0 else np.nan
        output[key + "_pair_available"] = bool(available)
    output["comparison_status"] = (
        "solver_or_missing_result" if _failed(case) or _failed(reference)
        else "both_critical_reached" if output["tcritical_s_pair_available"]
        else "critical_censored; available mean crossings may still be compared")
    delta = float(case["logD_delta"])
    fraction = case.get("perturb_weighted_dry_mass_fraction", np.nan)
    dose = delta * fraction if _finite(fraction) else np.nan
    output["weighted_logD_dose"] = dose
    output["mask_interpretation"] = (
        "state-dependent low-C window; time-averaged exposure is descriptive, not a fixed material-region dose"
        if case["perturb_kind"] == "lowC" else "fixed material-label mask" if case["perturb_kind"] in ("outer", "inner")
        else "global mask" if case["perturb_kind"] == "global" else "no diffusion perturbation")
    for key in EVENTS:
        change = output["delta_" + key]
        output["d_" + key + "_per_logD_amplitude"] = change / delta if delta and _finite(change) else np.nan
        output["d_" + key + "_per_weighted_logD_dose"] = change / dose if _finite(dose) and abs(dose) > 1e-14 and _finite(change) else np.nan
    return output


def analyze(results, cases=None, profile_dir=None):
    """Return descriptive tables; absent/censored events remain missing.

    Full parameter dictionaries must either be supplied through cases, or be
    present in the result table.  Optional profiles add independent support
    diagnostics without changing the recorded solver result.
    """
    raw = _read_results(results)
    if cases is None:
        missing = set(CASE_KEYS) - set(raw.columns)
        if missing:
            raise ValueError(f"Supply cases; input results do not contain design columns {sorted(missing)}")
        cases = raw[list(CASE_KEYS)].to_dict(orient="records")
    validate_cases(cases)
    frame = _make_joined(raw, cases)
    profiles, issues = [], []
    for index, row in frame.iterrows():
        case = {key: row[key] for key in CASE_KEYS}
        path = row.get("profile_path")
        if isinstance(path, str) and path:
            path = Path(path)
            if not path.is_absolute() and profile_dir is not None:
                path = Path(profile_dir) / path
        else:
            path = Path(profile_dir) / (row["id"] + ".npz") if profile_dir is not None else None
        if path is not None and path.is_file():
            try:
                summary = summarize_profile(path, case)
                profiles.append(summary)
                for key, source in (("perturb_support_dry_mass_fraction", "profile_support_fraction_time_average"),
                                    ("perturb_weighted_dry_mass_fraction", "profile_weighted_fraction_time_average")):
                    if not _finite(row.get(key)):
                        frame.loc[index, key] = summary[source]
                        frame.loc[index, key + "_source"] = "saved-profile trapezoidal average"
            except ValueError as error:
                issues.append({"id": row["id"], "severity": "error", "issue": str(error)})
        elif profile_dir is not None:
            issues.append({"id": row["id"], "severity": "missing", "issue": "Saved profile not found"})

    rows = frame.to_dict(orient="records")
    pairs, boundary, geometry, localization = [], [], [], []
    for row in rows:
        pairs.append(_pair(row, _base_for(row, rows), "same-task affine or diffusivity baseline"))
        if _failed(row):
            issues.append({"id": row["id"], "severity": "error", "issue": "Solver failure or missing result; excluded from paired event calculations"})
        for key in FRACTIONS:
            if _finite(row[key]) and not 0 <= row[key] <= 1 + 1e-12:
                issues.append({"id": row["id"], "severity": "error", "issue": f"{key} lies outside [0,1]"})
        tc, tm, tail = (_event(row, k) for k in EVENTS)
        if _finite(tc) and _finite(tm) and tc + 1e-6 < tm:
            issues.append({"id": row["id"], "severity": "error", "issue": "Maximum-moisture crossing precedes mean crossing"})
        if _finite(tc) and _finite(tm) and _finite(tail) and abs(tail - (tc - tm)) > 1e-4 * max(1, abs(tc - tm)):
            issues.append({"id": row["id"], "severity": "error", "issue": "tail_s differs from tcritical_s-tmean_s"})
        if not _finite(tc) and _finite(tail):
            issues.append({"id": row["id"], "severity": "error", "issue": "Tail cannot be finite without a critical crossing"})
        if "not_reached" in str(row["status"]).lower() and _finite(row.get("tcritical_s")):
            issues.append({"id": row["id"], "severity": "error", "issue": "Critical time conflicts with not-reached status"})
        if row["task"] == 1 and row["question"] == 4 and row["moving_radius"] and row["boundary_norm"] == "initial_dry_density":
            other = next((r for r in rows if r["task"] == 1 and r["question"] == 4 and r["moving_radius"]
                          and r["eps_final"] == row["eps_final"] and r["boundary_norm"] == "current_dry_density"), None)
            boundary.append(_pair(row, other, "initial-density minus current-density boundary"))
        if row["task"] == 1 and row["moving_radius"] and row["eps_final"] == 0 and row["boundary_norm"] == "current_dry_density":
            other = next((r for r in rows if r["task"] == 1 and r["question"] == row["question"]
                          and not r["moving_radius"] and r["eps_final"] == 0 and r["boundary_norm"] == "current_dry_density"), None)
            geometry.append(_pair(row, other, "measured-affine-radius minus fixed-radius control"))
        if row["task"] == 2 and row["perturb_kind"] == "outer":
            other = next((r for r in rows if r["task"] == 2 and r["question"] == row["question"]
                          and r["perturb_kind"] == "inner" and r["region_width"] == row["region_width"]
                          and r["logD_delta"] == row["logD_delta"]), None)
            localization.append(_pair(row, other, "outer minus complementary inner; unequal dry-mass support"))

    central = []
    for plus in rows:
        if plus["task"] != 2 or plus["logD_delta"] <= 0:
            continue
        minus = next((r for r in rows if r["task"] == 2 and r["question"] == plus["question"]
                      and r["perturb_kind"] == plus["perturb_kind"] and r["region_width"] == plus["region_width"]
                      and r["C_cut"] == plus["C_cut"] and r["C_smooth"] == plus["C_smooth"]
                      and r["logD_delta"] == -plus["logD_delta"]), None)
        base = _base_for(plus, rows)
        if minus is None or base is None:
            continue
        result = {key: plus[key] for key in ("question", "moving_radius", "perturb_kind", "region_width", "C_cut", "C_smooth")}
        result.update(positive_id=plus["id"], negative_id=minus["id"], baseline_id=base["id"],
                      amplitude=plus["logD_delta"], mask_is_state_dependent=plus["perturb_kind"] == "lowC")
        fp, fm = (r.get("perturb_weighted_dry_mass_fraction", np.nan) for r in (plus, minus))
        fraction = (fp + fm) / 2 if _finite(fp) and _finite(fm) else np.nan
        result["paired_mean_weighted_mass_fraction"] = fraction
        for key in EVENTS:
            p, m, b = (_event(r, key) for r in (plus, minus, base))
            available = _finite(p) and _finite(m)
            derivative = (p - m) / (2 * plus["logD_delta"]) if available else np.nan
            result["central_d_" + key + "_per_logD"] = derivative
            result["central_d_" + key + "_per_weighted_logD"] = derivative / fraction if _finite(derivative) and _finite(fraction) and fraction > 1e-14 else np.nan
            result["relative_central_d_" + key] = derivative / b if _finite(derivative) and _finite(b) and b > 0 else np.nan
            result["nonlinear_asymmetry_" + key] = p + m - 2 * b if available and _finite(b) else np.nan
        central.append(result)

    counts = frame.groupby(["task", "question", "status"], dropna=False).size().reset_index(name="cases")
    return {"case_results": frame, "paired_changes": pd.DataFrame(pairs),
            "boundary_normalization": pd.DataFrame(boundary), "geometry_controls": pd.DataFrame(geometry),
            "localization_contrasts": pd.DataFrame(localization), "central_sensitivities": pd.DataFrame(central),
            "profile_summary": pd.DataFrame(profiles),
            "integrity": pd.DataFrame(issues, columns=["id", "severity", "issue"]), "run_summary": counts}


def write_analysis(results, cases, out_dir, profile_dir=None):
    tables = analyze(results, cases=cases, profile_dir=profile_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out_dir / (name + ".csv"), index=False, encoding="utf-8-sig")
    notes = {
        "difference_direction": "candidate minus explicitly named reference",
        "censoring": "Unavailable critical/mean/tail events remain NaN; finite mean crossings can still be compared when critical times are censored.",
        "perturbation": "D_new = D_base * exp(logD_delta * mask). Outer/inner are complementary material-label masks.",
        "support": "Initial dry-mass weighted fraction with mask>0.5, averaged from t=0 to the recorded integration endpoint.",
        "strength": "Initial dry-mass weighted mask mean, time averaged; multiply by logD_delta to obtain weighted log-diffusivity exposure.",
        "lowC": "A dynamic state-dependent window. Its realized exposure is descriptive and cannot be treated as the same fixed input dose as outer/inner masks.",
        "interpretation": "These are controlled coefficient/deformation scenarios, not observations of real crust formation or parameter confidence intervals.",
        "integrity_errors": int((tables["integrity"]["severity"] == "error").sum()),
        "integrity_issues": len(tables["integrity"]),
    }
    (out_dir / "analysis_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    return tables


def create_analysis_figures(tables, out_dir):
    """Plot actual paired results with English labels; never execute a model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    settings = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                "axes.spines.right": False, "figure.facecolor": "white"}
    kinds = ("outer", "inner", "lowC", "global")
    labels = ("Outer", "Inner", "Low-C\n(dynamic)", "Global")
    colors = ("#4379ad", "#d58e43", "#759358", "#9171a1")
    case_frame, pairs = tables["case_results"], tables["paired_changes"]

    def finish(fig, name):
        path = out_dir / (name + ".png")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths[name] = path

    def primary(frame):
        return frame[(frame.task == 2) & np.isclose(frame.logD_delta, .1)
                     & np.isclose(frame.region_width, .1) & np.isclose(frame.C_cut, .2)
                     & np.isclose(frame.C_smooth, .02)]

    with plt.rc_context(settings):
        subset = case_frame[(case_frame.task == 1) & (case_frame.question == 4) & case_frame.moving_radius]
        if len(subset):
            fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), layout="constrained")
            for ax, key, title in zip(axes, ("tcritical_s", "tail_s"), ("Maximum-moisture threshold", "Tail: maximum minus mean threshold")):
                censored = 0
                for norm, label, color in (("current_dry_density", "Current surface dry density", colors[0]),
                                           ("initial_dry_density", "Initial dry density", colors[1])):
                    selected = subset[subset.boundary_norm == norm].sort_values("eps_final")
                    values = np.array([_event(row, key) for row in selected.to_dict("records")]) / 3600
                    censored += int(np.isnan(values).sum())
                    ax.plot(selected.eps_final, values, "o-", label=label, color=color)
                ax.set(xlabel="Final deformation parameter", ylabel="Time (h)", title=title)
                ax.set_xticks(np.sort(subset.eps_final.unique()))
                ax.grid(alpha=.2)
                if censored:
                    ax.text(.03, .03, f"{censored} unavailable/censored events omitted", transform=ax.transAxes, fontsize=8)
            axes[0].legend(fontsize=8)
            fig.suptitle("Task 1: Q4, identical measured outer radius")
            finish(fig, "analysis_task1_deformation")

        subset = primary(pairs)
        if len(subset):
            fig, axes = plt.subplots(2, 2, figsize=(10.4, 6.8), layout="constrained")
            for i, question in enumerate((3, 4)):
                selected = subset[subset.question == question].set_index("perturb_kind")
                for j, key in enumerate(("tcritical_s", "tail_s")):
                    values = [selected.loc[k, "delta_" + key] / 3600 if k in selected.index else np.nan for k in kinds]
                    ax = axes[i, j]
                    bars = ax.bar(labels, values, color=colors)
                    bars[2].set_hatch("//")
                    ax.axhline(0, color="black", linewidth=.7)
                    ax.grid(axis="y", alpha=.2)
                    ax.set(ylabel="Change from baseline (h)", title=f"Q{question}: " + ("critical time" if j == 0 else "tail duration"))
                    for index, value in enumerate(values):
                        if not np.isfinite(value):
                            ax.text(index, 0, "unavailable", rotation=90, ha="center", va="bottom", fontsize=8)
            fig.suptitle("Task 2: +0.10 log-diffusivity amplitude; unequal region support")
            finish(fig, "analysis_task2_effects")

        subset = primary(case_frame)
        if len(subset):
            fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), layout="constrained")
            x = np.arange(len(kinds))
            for ax, question in zip(axes, (3, 4)):
                selected = subset[subset.question == question].set_index("perturb_kind")
                weighted = [selected.loc[k, FRACTIONS[1]] if k in selected.index else np.nan for k in kinds]
                support = [selected.loc[k, FRACTIONS[0]] if k in selected.index else np.nan for k in kinds]
                ax.bar(x - .18, weighted, .36, label="Weighted mask", color=colors[0])
                ax.bar(x + .18, support, .36, label="Support: mask > 0.5", color=colors[1])
                ax.set(xticks=x, xticklabels=labels, ylim=(0, 1.08), ylabel="Initial dry-mass fraction", title=f"Q{question}")
                ax.grid(axis="y", alpha=.2)
            axes[0].legend(fontsize=8)
            fig.suptitle("Time averages over each recorded trajectory\nLow-C exposure is dynamic; these are coefficient scenarios")
            finish(fig, "analysis_task2_exposure")
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--suite", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--cases", type=Path, help="Optional design JSON; overrides --suite")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figures", type=Path, help="Optional directory for English-label figures")
    args = parser.parse_args()
    if args.cases:
        payload = json.loads(args.cases.read_text(encoding="utf-8"))
        cases = payload["cases"] if isinstance(payload, dict) else payload
    else:
        cases = make_cases(args.suite)
    tables = write_analysis(args.results, cases, args.output, profile_dir=args.profile_dir)
    if args.figures:
        for name, path in create_analysis_figures(tables, args.figures).items():
            print(name, path)
    print(tables["run_summary"].to_string(index=False))
    print("Integrity issues:", len(tables["integrity"]))
    print("Analysis directory:", args.output)
