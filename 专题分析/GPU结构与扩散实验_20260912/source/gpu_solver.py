"""FP64 NumPy/CuPy 批量 SDIRK2 求解器；CUDA 请求绝不静默退回 CPU。

状态形状 (batch, nodes, 2)，最后一维为摄氏温度、干基含水率。
采用六色差分块三对角 Jacobian、并行循环约减(PCR)、步长加倍误差
控制。批次共享时间步，因此最困难的活动样本决定步长；应按相近刚性
分批。此文件包含 GPU 执行路径，不表示已在 NVIDIA 硬件上实测。
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import platform
from time import perf_counter

import numpy as np


GAMMA = 1.0 - 1.0 / math.sqrt(2.0)


@dataclass
class BatchResult:
    sample_times: np.ndarray
    samples_T: np.ndarray
    samples_C: np.ndarray
    sample_valid: np.ndarray
    t_critical: np.ndarray
    t_mean_threshold: np.ndarray
    max_event_brackets: np.ndarray
    mean_event_brackets: np.ndarray
    final_state: np.ndarray
    final_times: np.ndarray
    status: np.ndarray
    stats: dict


def select_backend(backend):
    """显式返回真实执行后端与设备信息；不安装软件、不回退。"""
    if backend == "numpy":
        return np, {"backend": "numpy", "device": "CPU", "machine": platform.machine(),
                    "numpy_version": np.__version__, "cuda_executed": False, "dtype": "float64"}
    if backend != "cupy":
        raise ValueError("backend 必须明确为 'numpy' 或 'cupy'")
    try:
        cp = importlib.import_module("cupy")
        count = cp.cuda.runtime.getDeviceCount()
        if count < 1:
            raise RuntimeError("未发现 CUDA 设备")
        device_id = cp.cuda.runtime.getDevice()
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        probe = cp.ones(1, dtype=cp.float64)
        probe *= 2.0
        cp.cuda.get_current_stream().synchronize()
        if float(probe.get()[0]) != 2.0:
            raise RuntimeError("CUDA FP64 初始化探针失败")
    except Exception as error:
        raise RuntimeError("明确请求了CuPy/CUDA，但CUDA后端不可用；没有退回CPU。" + str(error)) from error
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return cp, {"backend": "cupy", "device": str(name), "device_id": int(device_id),
                "cupy_version": cp.__version__, "cuda_runtime_version": cp.cuda.runtime.runtimeGetVersion(),
                "cuda_executed": True, "dtype": "float64"}


def _host(xp, value):
    return np.asarray(value) if xp is np else xp.asnumpy(value)


def _mm(xp, left, right):
    return xp.einsum("bnij,bnjk->bnik", left, right)


def _mv(xp, matrix, vector):
    return xp.einsum("bnij,bnj->bni", matrix, vector)


def _inv2(xp, matrix):
    determinant = matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
    safe = xp.where(xp.abs(determinant) > 1e-290, determinant, xp.nan)
    inverse = xp.empty_like(matrix)
    inverse[..., 0, 0] = matrix[..., 1, 1] / safe
    inverse[..., 1, 1] = matrix[..., 0, 0] / safe
    inverse[..., 0, 1] = -matrix[..., 0, 1] / safe
    inverse[..., 1, 0] = -matrix[..., 1, 0] / safe
    return inverse


def block_pcr(lower, diagonal, upper, right_hand_side, xp=np):
    """批量2×2块三对角求解，PCR有O(log nodes)个并行消元层。

    块形状(batch,n,2,2)，右端(batch,n,2)。边界lower[:,0]和
    upper[:,-1]必须为0。输入不原地修改；奇数节点数同样适用。
    """
    n = diagonal.shape[1]
    a, b, c, d = lower.copy(), diagonal.copy(), upper.copy(), right_hand_side.copy()
    node = xp.arange(n)
    stride = 1
    while stride < n:
        inverse = _inv2(xp, b)
        left = _mm(xp, a, xp.roll(inverse, stride, axis=1))
        right = _mm(xp, c, xp.roll(inverse, -stride, axis=1))
        left = xp.where((node >= stride)[None, :, None, None], left, 0.0)
        right = xp.where((node + stride < n)[None, :, None, None], right, 0.0)
        b_new = b - _mm(xp, left, xp.roll(c, stride, axis=1)) - _mm(xp, right, xp.roll(a, -stride, axis=1))
        d_new = d - _mv(xp, left, xp.roll(d, stride, axis=1)) - _mv(xp, right, xp.roll(d, -stride, axis=1))
        a_new = -_mm(xp, left, xp.roll(a, stride, axis=1))
        c_new = -_mm(xp, right, xp.roll(c, -stride, axis=1))
        a, b, c, d = a_new, b_new, c_new, d_new
        stride *= 2
    return _mv(xp, _inv2(xp, b), d)


def colored_jacobian(fun, t, y, f0, xp=np):
    """三种节点颜色×两种状态分量：六次RHS计算获取块三对角Jacobian。

    前提是RHS确实只有本节点及左右节点耦合。输入非局部RHS不适用。
    """
    batch, n, _ = y.shape
    lower = xp.zeros((batch, n, 2, 2), dtype=xp.float64)
    diagonal = xp.zeros_like(lower)
    upper = xp.zeros_like(lower)
    steps = math.sqrt(np.finfo(np.float64).eps) * (1.0 + xp.abs(y))
    nodes = xp.arange(n)
    for component in range(2):
        for color in range(3):
            mask = nodes % 3 == color
            perturbed = y.copy()
            perturbed[..., component] += xp.where(mask[None, :], steps[..., component], 0.0)
            difference = fun(t, perturbed) - f0
            diagonal[..., :, component] = xp.where(mask[None, :, None],
                difference / steps[..., component, None], diagonal[..., :, component])
            left_mask = ((nodes - 1) % 3 == color) & (nodes > 0)
            lower[..., :, component] = xp.where(left_mask[None, :, None],
                difference / xp.roll(steps[..., component], 1, axis=1)[..., None], lower[..., :, component])
            right_mask = ((nodes + 1) % 3 == color) & (nodes + 1 < n)
            upper[..., :, component] = xp.where(right_mask[None, :, None],
                difference / xp.roll(steps[..., component], -1, axis=1)[..., None], upper[..., :, component])
    return lower, diagonal, upper


def solve_batch(case_arrays, grid, inputs, *, backend="numpy", t_end=259200.0,
                y0=None, sample_times=None, rtol=2e-5, atol=(1e-5, 1e-7),
                initial_step=0.1, max_step=300.0, min_step=1e-8,
                threshold=0.15, track_mean_event=True, stop_at_max_event=True,
                event_time_tol=1.0, max_steps=200000, max_newton=10,
                rhs_fn=None, progress=None):
    """批量L稳定二阶SDIRK与步长加倍控制；输出均为NumPy主机数组。

    event_time_tol限制检测到穿越时的半步时间括区宽度；括区内线性
    定位，记录完整括区，绝不把它声称为超出括区精度的精确根。
    sample_times最多500个，只在已求得且未停止的时域内线性插值。
    可注入rhs_fn用于独立数学验证；默认使用同目录physics.rhs。
    """
    if rhs_fn is None:
        from physics import rhs as rhs_fn
    xp, device = select_backend(backend)
    raw_cases = {key: _host(xp, value) if xp is not np and isinstance(value, xp.ndarray)
                 else np.asarray(value) for key, value in case_arrays.items()}
    lengths = [value.shape[0] for value in raw_cases.values() if value.ndim]
    if not lengths or len(set(lengths)) != 1:
        raise ValueError("case_arrays各数组第一维须为相同非零batch")
    batch = lengths[0]
    if batch < 1:
        raise ValueError("batch不能为空")
    cases = {}
    for key, value in raw_cases.items():
        if value.dtype.kind not in "biuf":
            raise ValueError(f"case_arrays[{key}]须为数值编码，不能传字符串/object")
        if value.ndim > 1:
            raise ValueError(f"case_arrays[{key}]必须为标量或(batch,)数组")
        cases[key] = xp.asarray(np.broadcast_to(value, (batch,)) if value.ndim == 0 else value)
    g = {key: xp.asarray(value, dtype=xp.float64) if isinstance(value, (list, tuple, np.ndarray)) else value
         for key, value in grid.items()}
    n = len(grid["a"])
    if n < 3:
        raise ValueError("至少需要3个径向节点")
    initial = np.empty((batch, n, 2), dtype=np.float64) if y0 is None else np.asarray(y0, dtype=np.float64)
    if y0 is None:
        initial[..., 0], initial[..., 1] = 28.0, 2.55
    if initial.shape != (batch, n, 2) or not np.isfinite(initial).all():
        raise ValueError("y0形状或有限性不符合(batch,n,2)")
    if np.any(initial[..., 1] <= 0) or np.any(initial[..., 0] <= -273.15):
        raise ValueError("初始状态超出正含水率/正绝对温度定义域")
    t_end = float(t_end)
    atols = np.asarray(atol, dtype=float)
    if atols.shape == ():
        atols = np.repeat(atols, 2)
    if atols.shape != (2,) or not np.isfinite(atols).all() or np.any(atols <= 0):
        raise ValueError("atol必须为正标量或(T,C)两个正数")
    positive_settings = np.array([t_end, rtol, initial_step, max_step, min_step, event_time_tol])
    if not np.isfinite(positive_settings).all() or np.any(positive_settings <= 0):
        raise ValueError("时域、步长、容差必须为正")
    if not np.isfinite(threshold) or threshold <= 0 or max_steps < 1 or max_newton < 1:
        raise ValueError("阈值必须为正有限数，最大步数和Newton次数必须为正")
    if np.any(raw_cases.get("moving_radius", False)) and t_end > inputs.radius_end:
        raise ValueError("移动半径案例的t_end超过实测半径范围")
    times = np.linspace(0.0, t_end, 241) if sample_times is None else np.asarray(sample_times, dtype=float)
    if times.ndim != 1 or len(times) > 500 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or np.any(times < 0) or np.any(times > t_end):
        raise ValueError("sample_times须为至多500个严格递增、位于积分时域内的时刻")
    samples = np.full((batch, len(times), n, 2), np.nan)
    valid = np.zeros((batch, len(times)), dtype=bool)
    critical = np.full(batch, np.nan)
    mean_event = np.full(batch, np.nan)
    max_bracket, mean_bracket = np.full((batch, 2), np.nan), np.full((batch, 2), np.nan)
    final = initial.copy()
    final_times = np.zeros(batch)
    status = np.full(batch, "running", dtype=object)
    active_host = np.ones(batch, dtype=bool)
    active = xp.asarray(active_host)
    y = xp.asarray(initial, dtype=xp.float64)
    final_device = y.copy()
    atol_gpu = xp.asarray(atols)[None, None, :]
    weights = xp.asarray(grid["weights"], dtype=xp.float64)
    weights = weights / weights.sum()
    stats = {**device, "method": "SDIRK2_step_doubling_PCR", "gamma": GAMMA,
             "rtol": float(rtol), "atol": atols.tolist(), "batch": batch, "nodes": n,
             "accepted_steps": 0, "rejected_steps": 0, "event_refinements": 0,
             "rhs_calls": 0, "jacobian_builds": 0, "newton_iterations": 0,
             "event_time_tol_s": event_time_tol, "event_method": "linear_within_refined_half_step_bracket",
             "sample_method": "piecewise_linear_between_accepted_half_steps",
             "maximum_accepted_error_norm": 0.0, "cuda_hardware_validation_claimed": False}
    start_clock = perf_counter()
    eye = xp.broadcast_to(xp.eye(2, dtype=xp.float64), (batch, n, 2, 2))
    phase = "observed"

    def evaluate(t, state):
        stats["rhs_calls"] += 1
        derivative = rhs_fn(t, state, cases, g, inputs, xp, phase)
        return xp.where(active[:, None, None], derivative, 0.0)

    def norm(value, left, right):
        scale = atol_gpu + rtol * xp.maximum(xp.abs(left), xp.abs(right))
        per_case = xp.max(xp.abs(value) / scale, axis=(1, 2))
        return float(_host(xp, xp.max(xp.where(active, per_case, 0.0))))

    def in_domain(state):
        okay = xp.all(xp.isfinite(state), axis=(1, 2)) & xp.all(state[..., 1] > 0, axis=1) & xp.all(state[..., 0] > -273.15, axis=1)
        return bool(_host(xp, xp.all(okay | ~active)))

    def implicit(base, tt, alpha, guess, jacobian=None):
        z = guess.copy()
        f = evaluate(tt, z)
        for iteration in range(max_newton):
            residual = z - base - alpha * f
            old_norm = norm(residual, base, z)
            if np.isfinite(old_norm) and old_norm <= 0.03:
                return z, f, jacobian, True
            if not np.isfinite(old_norm):
                return z, f, jacobian, False
            if jacobian is None or iteration == 4:
                jacobian = colored_jacobian(evaluate, tt, z, f, xp)
                stats["jacobian_builds"] += 1
            lower, diagonal, upper = jacobian
            delta = block_pcr(-alpha * lower, eye - alpha * diagonal, -alpha * upper, -residual, xp)
            stats["newton_iterations"] += 1
            accepted = False
            damping = 1.0
            for _ in range(10):
                candidate = z + damping * delta
                if in_domain(candidate):
                    f_candidate = evaluate(tt, candidate)
                    new_norm = norm(candidate - base - alpha * f_candidate, base, candidate)
                    if np.isfinite(new_norm) and (new_norm < old_norm or new_norm <= 0.03):
                        z, f, accepted = candidate, f_candidate, True
                        break
                damping *= 0.5
            if not accepted:
                return z, f, jacobian, False
        return z, f, jacobian, False

    def sdirk(tt, state, step):
        z1, f1, jac, success = implicit(state, tt + GAMMA * step, GAMMA * step, state)
        if not success:
            return z1, False
        base = state + (1.0 - GAMMA) * step * f1
        z2, _, _, success = implicit(base, tt + step, GAMMA * step, z1, jac)
        return z2, success

    def metrics(state):
        return (_host(xp, xp.max(state[..., 1], axis=1)),
                _host(xp, xp.sum(state[..., 1] * weights[None, :], axis=1)))

    maximum, mean = metrics(y)
    initial_max = maximum <= threshold
    initial_mean = mean <= threshold
    critical[initial_max] = 0.0
    max_bracket[initial_max] = 0.0
    if track_mean_event:
        mean_event[initial_mean] = 0.0
        mean_bracket[initial_mean] = 0.0
    if stop_at_max_event:
        active_host[initial_max] = False
        status[initial_max] = "initially_at_or_below_threshold"
        active = xp.asarray(active_host)
    if len(times) and times[0] == 0:
        samples[:, 0] = initial
        valid[:, 0] = True
    t, h = 0.0, min(initial_step, max_step)
    attempts = 0
    failure = None
    while t < t_end and active_host.any():
        attempts += 1
        if attempts > max_steps:
            failure = "failed_step_limit"
            break
        phase = "observed" if t < inputs.boundary_end else "plateau"
        stop = min(t_end, inputs.boundary_end) if phase == "observed" else t_end
        h = min(h, max_step, stop - t)
        if h < min_step:
            failure = "failed_minimum_step"
            break
        full, okay_full = sdirk(t, y, h)
        midpoint, okay_mid = sdirk(t, y, h / 2) if okay_full else (y, False)
        candidate, okay_end = sdirk(t + h / 2, midpoint, h / 2) if okay_mid else (y, False)
        if not okay_end:
            stats["rejected_steps"] += 1
            h *= 0.25
            continue
        error = norm((candidate - full) / 3.0, y, candidate)
        if not np.isfinite(error) or error > 1.0:
            stats["rejected_steps"] += 1
            h *= max(0.1, min(0.8, 0.8 * error**(-1.0 / 3.0))) if np.isfinite(error) else 0.1
            continue
        middle_max, middle_mean = metrics(midpoint)
        end_max, end_mean = metrics(candidate)
        crossings = active_host & np.isnan(critical) & (
            ((maximum > threshold) & (middle_max <= threshold)) |
            ((middle_max > threshold) & (end_max <= threshold)))
        if track_mean_event:
            crossings |= active_host & np.isnan(mean_event) & (
                ((mean > threshold) & (middle_mean <= threshold)) |
                ((middle_mean > threshold) & (end_mean <= threshold)))
        if crossings.any() and h / 2 > event_time_tol * (1 + 1e-12):
            stats["event_refinements"] += 1
            h = min(h * 0.5, 2 * event_time_tol)
            continue
        # 正常时间推进始终保留在设备；仅输出样本或定位新事件时搬运场。
        need_values = crossings.any() or np.any((times > t) & (times <= t + h))
        if need_values:
            old_host, mid_host, end_host = _host(xp, y), _host(xp, midpoint), _host(xp, candidate)
        else:
            old_host = mid_host = end_host = None
        accepted_active = active_host.copy()
        intervals = [(t, t + h / 2, old_host, mid_host, maximum, middle_max, mean, middle_mean),
                     (t + h / 2, t + h, mid_host, end_host, middle_max, end_max, middle_mean, end_mean)]
        for left_t, right_t, left_y, right_y, left_max, right_max, left_mean, right_mean in intervals:
            for event_values, brackets, v0, v1, enabled in [
                (critical, max_bracket, left_max, right_max, True),
                (mean_event, mean_bracket, left_mean, right_mean, track_mean_event),
            ]:
                if not enabled:
                    continue
                hit = accepted_active & np.isnan(event_values) & (v0 > threshold) & (v1 <= threshold)
                if hit.any():
                    fraction = (v0[hit] - threshold) / (v0[hit] - v1[hit])
                    event_values[hit] = left_t + fraction * (right_t - left_t)
                    brackets[hit, 0], brackets[hit, 1] = left_t, right_t
                    if event_values is critical and stop_at_max_event:
                        final[hit] = left_y[hit] + fraction[:, None, None] * (right_y[hit] - left_y[hit])
                        final_times[hit] = event_values[hit]
                        status[hit] = "reached_threshold"
            sample_indices = np.flatnonzero((times > left_t) & (times <= right_t))
            for index in sample_indices:
                sample_active = accepted_active & ((not stop_at_max_event) | np.isnan(critical) | (times[index] <= critical))
                fraction = (times[index] - left_t) / (right_t - left_t)
                samples[sample_active, index] = left_y[sample_active] + fraction * (right_y[sample_active] - left_y[sample_active])
                valid[sample_active, index] = True
        if stop_at_max_event:
            active_host &= np.isnan(critical)
        active = xp.asarray(active_host)
        t = float(stop if abs(t + h - stop) <= 8 * np.finfo(float).eps * max(1, abs(stop)) else t + h)
        final_times[active_host] = t
        final_device = xp.where(active[:, None, None], candidate, final_device)
        stopped_now = accepted_active & ~active_host
        if stopped_now.any():
            final_device[xp.asarray(stopped_now)] = xp.asarray(final[stopped_now])
        y = final_device.copy()
        maximum, mean = metrics(y)
        stats["accepted_steps"] += 1
        stats["maximum_accepted_error_norm"] = max(stats["maximum_accepted_error_norm"], error)
        if progress is not None and stats["accepted_steps"] % 100 == 0:
            progress({"time_s": t, "active_cases": int(active_host.sum()), **stats})
        factor = 3.0 if error < 1e-14 else max(0.3, min(3.0, 0.9 * error**(-1.0 / 3.0)))
        h *= factor
    if failure is not None:
        status[active_host] = failure
    else:
        status[active_host] = np.where(np.isfinite(critical[active_host]), "finished_horizon_threshold_reached", "not_reached_by_horizon")
    if xp is not np:
        xp.cuda.get_current_stream().synchronize()
    stats.update(wall_time_s=perf_counter() - start_clock, attempted_steps=attempts,
                 final_shared_time_s=t, successful_cases=int(np.sum(~np.char.startswith(status.astype(str), "failed"))),
                 failed_cases=int(np.sum(np.char.startswith(status.astype(str), "failed"))),
                 stop_at_max_event=bool(stop_at_max_event), boundary_split_s=float(inputs.boundary_end))
    final = _host(xp, final_device)
    return BatchResult(times, samples[..., 0], samples[..., 1], valid, critical, mean_event,
                       max_bracket, mean_bracket, final, final_times, status, stats)
