"""
generate_dataset.py
Synthetic Cooling Telemetry Generator -- Doha DC-1 Reference Campus (30 MW)
OptvanceAI Arabia -- CoolingHealthSentinel Pre-Onboarding Assignment

Generates 12 months (Jan-Dec 2025) of 15-minute interval cooling telemetry
(35,040 rows) per the Section 1 specification of the assignment brief.

DESIGN PRINCIPLE
----------------
Independent ("root") signals are generated first: weather, IT load, and
component health trends (chiller age decline, fouling drift, bearing wear).
The 8 anomaly events mutate these root signals within scheduled 2-6 hour
windows, each touching >= 3 sensor channels simultaneously, as required.
All dependent columns (inlet/outlet temps, PUE, CHS) are derived from the
final root signals so anomaly effects propagate physically rather than
being hardcoded onto output columns directly.

cooling_health_score is computed last as a nonlinear, noisy composite of
6 normalized feature-health terms -- not a clean linear function of any
single input -- per the spec's "not a trivially derived value" requirement.

ANOMALY DESIGN -- TWO POPULATIONS (resolved per instructor guidance, see
README_dataset.md for the full discussion):
Section 1.2's "~3% prevalence" and Section 1.3's "8 events of 2-6 hours"
(<=192 rows, ~0.5%) cannot both be literally true. Per instructor
clarification, both are kept, as two distinct anomaly populations rather
than one:

  1. STRUCTURED events (8 total, Section 1.3) -- multi-hour, >=3
     correlated channels degrading together, representing genuine
     failure-precursor incidents. anomaly_class = "structured".
  2. BACKGROUND anomalies (~2.7% of remaining normal rows) -- isolated
     single-timestep sensor perturbations (Gaussian noise at 1.5-2x the
     normal standard deviation, applied to 3 randomly chosen telemetry
     columns) representing low-grade sensor drift, NOT genuine health
     degradation. anomaly_class = "background". cooling_health_score is
     deliberately left untouched for these rows -- the underlying system
     is fine, only the sensor reading is noisy, which is the intended
     contrast with the structured population for Day 5's analysis of
     whether IsolationForest can tell the two apart.

Combined prevalence lands at ~3%, reconciling both spec lines.
"""

import numpy as np
import pandas as pd

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

START = "2025-01-01 00:00"
FREQ = "15min"
PERIODS = 35040  # 365 days * 96 intervals/day (2025 is not a leap year)
STEPS_PER_HOUR = 4

ANOMALY_TYPES = [
    "cooling_tower_fouling",
    "pump_bearing_wear",
    "refrigerant_leak_step",   # scheduled in month 8 -> the spec's step-drop event
    "crah_filter_clog",
    "chiller_control_oscillation",
    "flow_sensor_fault",
    "ai_thermal_overload",
    "tower_fan_degradation",
]

BACKGROUND_ANOMALY_FRACTION = 0.027  # ~2.7% of remaining normal rows
BACKGROUND_PERTURBATION_FEATURES = [
    "outdoor_temp_c", "outdoor_humidity_pct", "it_load_mw", "chiller_inlet_temp_c",
    "chiller_outlet_temp_c", "chiller_cop", "cooling_tower_approach_c", "crah_delta_t_c",
    "pump_vibration_mms", "pump_flow_rate_ls", "water_conductivity_us", "pue",
]

SPEC_RANGES = {
    "outdoor_temp_c": (18, 48), "outdoor_humidity_pct": (10, 85),
    "it_load_mw": (8, 28), "chiller_inlet_temp_c": (6, 14),
    "chiller_outlet_temp_c": (12, 22), "chiller_cop": (2.8, 6.5),
    "cooling_tower_approach_c": (2, 8), "crah_delta_t_c": (8, 18),
    "pump_vibration_mms": (0.5, 4.5), "pump_flow_rate_ls": (80, 220),
    "water_conductivity_us": (200, 900), "pue": (1.05, 1.45),
    "cooling_health_score": (55, 98),
}


# ---------------------------------------------------------------------------
# 1. Timestamps + time features
# ---------------------------------------------------------------------------
def build_time_index():
    ts = pd.date_range(START, periods=PERIODS, freq=FREQ)
    df = pd.DataFrame({"timestamp": ts})
    df["doy"] = df["timestamp"].dt.dayofyear
    df["month"] = df["timestamp"].dt.month
    df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
    df["_t_frac"] = np.arange(len(df)) / len(df)  # 0 at Jan 1, ~1 at Dec 31
    return df


# ---------------------------------------------------------------------------
# 2. outdoor_temp_c / outdoor_humidity_pct (Qatar climate)
# ---------------------------------------------------------------------------
def gen_weather(df):
    n = len(df)
    # Seasonal envelope: hottest Jun-Sep, peak ~day 200 (mid-July)
    seasonal = 33 + 11 * np.cos(2 * np.pi * (df["doy"] - 200) / 365)
    # Diurnal: peak ~15:00. A single 24h sinusoid is exactly antiphase
    # (12h) from its peak, so an exact peak=15 / trough=05 pair (10h gap)
    # isn't reachable with one harmonic -- this primary harmonic puts the
    # trough at ~03:00, ~2h earlier than the spec's 05:00 reference point.
    diurnal = 6 * np.cos(2 * np.pi * (df["hour"] - 15) / 24)
    outdoor_temp_c = seasonal + diurnal + rng.normal(0, 1.0, n)
    df["outdoor_temp_c"] = np.clip(outdoor_temp_c, 18, 48)

    # Inversely correlated with temp, calibrated to span the full 10-85 range
    humidity = 85 - 2.5 * (df["outdoor_temp_c"] - 18) + rng.normal(0, 5, n)
    df["outdoor_humidity_pct"] = np.clip(humidity, 10, 85)
    return df


# ---------------------------------------------------------------------------
# 3. it_load_mw: baseline ~12 MW + 2-5 AI bursts/day to 24-28 MW
# ---------------------------------------------------------------------------
def gen_it_load(df):
    n = len(df)
    daily_cycle = 1.0 * np.sin(2 * np.pi * (df["hour"] - 14) / 24)
    base_load = np.clip(12 + daily_cycle + rng.normal(0, 0.6, n), 8, 16)

    burst_level = np.full(n, np.nan)
    steps_per_day = 96
    n_days = n // steps_per_day
    for day in range(n_days):
        day_start = day * steps_per_day
        n_bursts = rng.integers(2, 6)  # 2-5 inclusive
        for _ in range(n_bursts):
            dur = rng.integers(2, 17)  # 30 min (2 steps) to 4h (16 steps)
            offset = rng.integers(0, max(steps_per_day - dur, 1))
            start_idx = day_start + offset
            end_idx = min(start_idx + dur, n)
            peak = rng.uniform(24, 28)
            ramp = np.minimum(np.arange(end_idx - start_idx) + 1,
                               (end_idx - start_idx) - np.arange(end_idx - start_idx))
            ramp = ramp / ramp.max() if ramp.max() > 0 else ramp
            window_vals = base_load[start_idx:end_idx] + ramp * (peak - base_load[start_idx:end_idx])
            existing = burst_level[start_idx:end_idx]
            burst_level[start_idx:end_idx] = np.where(
                np.isnan(existing), window_vals, np.maximum(existing, window_vals)
            )

    it_load = np.where(np.isnan(burst_level), base_load, burst_level)
    df["it_load_mw"] = np.clip(it_load, 8, 28)
    return df


# ---------------------------------------------------------------------------
# 4. Independent root parameters anomalies will mutate
# ---------------------------------------------------------------------------
def gen_system_roots(df):
    n = len(df)
    df["_chiller_age_factor"] = 1 - 0.08 * df["_t_frac"]           # 8%/yr linear decline
    df["_chiller_step_factor"] = 1.0                                 # set by month-8 anomaly
    df["_approach_root"] = 3.5 + 0.5 * df["_t_frac"] + rng.normal(0, 0.3, n)  # mild fouling drift
    df["_vibration_root"] = 1.0 + 0.5 * df["_t_frac"] + rng.normal(0, 0.15, n)  # mild bearing wear
    df["_flow_override"] = np.nan
    df["_oscillation_active"] = False
    df["_overload_active"] = False
    return df


# ---------------------------------------------------------------------------
# 5. Schedule + inject 8 anomalies (one per type, chronological, non-overlapping)
# ---------------------------------------------------------------------------
def schedule_anomalies(n, n_events, min_gap_steps=4 * 24 * 12, month8_window=None):
    starts = [None] * n_events
    # Force the refrigerant_leak_step event into month 8 (Aug) per spec
    idx_step = ANOMALY_TYPES.index("refrigerant_leak_step")
    starts[idx_step] = int(rng.integers(month8_window[0], month8_window[1]))

    placed = [starts[idx_step]]
    for i in range(n_events):
        if i == idx_step:
            continue
        attempts = 0
        while attempts < 5000:
            attempts += 1
            cand = int(rng.integers(4 * 24 * 5, n - 4 * 24 * 10))
            if all(abs(cand - s) > min_gap_steps for s in placed):
                starts[i] = cand
                placed.append(cand)
                break
    order = np.argsort(starts)
    return starts, order


def inject_anomalies(df):
    n = len(df)
    month8_mask = df["month"] == 8
    month8_idx = np.where(month8_mask.values)[0]
    month8_window = (month8_idx[0], month8_idx[-1])

    starts, _ = schedule_anomalies(n, len(ANOMALY_TYPES), month8_window=month8_window)
    df["is_anomaly"] = 0
    df["anomaly_type"] = ""
    log = []

    for a_type, start in zip(ANOMALY_TYPES, starts):
        if a_type == "cooling_tower_fouling":
            dur = rng.integers(4 * 4, 4 * 6 + 1)  # 4-6h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_approach_root"] += np.linspace(1.5, 4.0, end - start)

        elif a_type == "pump_bearing_wear":
            dur = rng.integers(4 * 2, 4 * 4 + 1)  # 2-4h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_vibration_root"] += rng.uniform(1.5, 2.5)

        elif a_type == "refrigerant_leak_step":
            dur = rng.integers(4 * 4, 4 * 6 + 1)  # 4-6h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_chiller_step_factor"] = np.linspace(1.0, 0.80, end - start)
            df.loc[end:, "_chiller_step_factor"] = 0.85  # permanent partial step from here on

        elif a_type == "crah_filter_clog":
            dur = rng.integers(4 * 3, 4 * 5 + 1)  # 3-5h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_approach_root"] += rng.uniform(0.5, 1.0)  # minor coupling
            df.attrs.setdefault("crah_clog_windows", []).append((start, end))

        elif a_type == "chiller_control_oscillation":
            dur = rng.integers(4 * 2, 4 * 4 + 1)  # 2-4h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_oscillation_active"] = True

        elif a_type == "flow_sensor_fault":
            dur = rng.integers(4 * 2, 4 * 3 + 1)  # 2-3h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_approach_root"] += rng.uniform(0.5, 1.0)
            df.attrs.setdefault("flow_fault_windows", []).append((start, end))

        elif a_type == "ai_thermal_overload":
            dur = rng.integers(4 * 4, 4 * 6 + 1)  # 4-6h
            end = min(start + dur, n)
            df.loc[start:end - 1, "it_load_mw"] = np.clip(
                df.loc[start:end - 1, "it_load_mw"] + rng.uniform(3, 6), 8, 28
            )
            df.loc[start:end - 1, "_overload_active"] = True

        elif a_type == "tower_fan_degradation":
            dur = rng.integers(4 * 3, 4 * 5 + 1)  # 3-5h
            end = min(start + dur, n)
            df.loc[start:end - 1, "_approach_root"] += np.linspace(1.0, 3.0, end - start)

        df.loc[start:end - 1, "is_anomaly"] = 1
        df.loc[start:end - 1, "anomaly_type"] = a_type
        log.append((a_type, df.loc[start, "timestamp"], df.loc[end - 1, "timestamp"], end - start))

    return df, log


# ---------------------------------------------------------------------------
# 6. Derive all dependent columns from the final root signals
# ---------------------------------------------------------------------------
def derive_system(df):
    n = len(df)

    df["chiller_inlet_temp_c"] = np.clip(
        7 + 0.15 * (df["it_load_mw"] - 12) + rng.normal(0, 0.3, n), 6, 14
    )
    df["chiller_outlet_temp_c"] = np.clip(
        df["chiller_inlet_temp_c"] + 5 + rng.normal(0, 0.3, n), 12, 22
    )

    # chiller_control_oscillation: inlet/outlet oscillate abnormally
    osc = df["_oscillation_active"].values
    if osc.any():
        t = np.arange(n)
        osc_wave = 1.5 * np.sin(2 * np.pi * t / 3)
        df.loc[osc, "chiller_inlet_temp_c"] = np.clip(
            df.loc[osc, "chiller_inlet_temp_c"] + osc_wave[osc], 6, 14
        )
        df.loc[osc, "chiller_outlet_temp_c"] = np.clip(
            df.loc[osc, "chiller_outlet_temp_c"] + osc_wave[osc], 12, 22
        )

    temp_factor = 1 - 0.01 * (df["outdoor_temp_c"] - 25)
    base_cop = 5.8 * df["_chiller_age_factor"] * df["_chiller_step_factor"] * temp_factor
    cop = base_cop + rng.normal(0, 0.15, n)
    # refrigerant_leak_step and chiller_control_oscillation also dent COP directly
    leak_mask = df["anomaly_type"] == "refrigerant_leak_step"
    overload_mask = df["anomaly_type"] == "ai_thermal_overload"
    clog_mask = df["anomaly_type"] == "crah_filter_clog"
    cop = cop - leak_mask.values * rng.uniform(0.3, 0.6)
    cop = cop - osc * rng.uniform(0.2, 0.4)
    cop = cop - overload_mask.values * rng.uniform(0.2, 0.4)  # higher heat load stresses the chiller
    cop = cop - clog_mask.values * rng.uniform(0.15, 0.3)      # restricted airflow raises lift
    df["chiller_cop"] = np.clip(cop, 2.8, 6.5)

    df["cooling_tower_approach_c"] = np.clip(df["_approach_root"], 2, 8)

    df["crah_delta_t_c"] = np.clip(
        10 + 0.2 * (df["it_load_mw"] - 12) + rng.normal(0, 0.5, n), 8, 18
    )
    overload = df["_overload_active"].values
    if overload.any():
        df.loc[overload, "crah_delta_t_c"] = np.clip(
            df.loc[overload, "crah_delta_t_c"] + rng.uniform(1.5, 3.0), 8, 18
        )
        df.loc[overload, "chiller_inlet_temp_c"] = np.clip(
            df.loc[overload, "chiller_inlet_temp_c"] + rng.uniform(1.0, 2.0), 6, 14
        )

    df["pump_vibration_mms"] = np.clip(df["_vibration_root"], 0.5, 4.5)

    flow_natural = (
        150 + 2.0 * (df["it_load_mw"] - 12) - 5 * (df["pump_vibration_mms"] - 1.0)
        + rng.normal(0, 5, n)
    )
    flow_final = flow_natural.copy()
    fault_windows = df.attrs.get("flow_fault_windows", [])
    for start, end in fault_windows:
        freeze_val = flow_natural.iloc[start - 1] if start > 0 else flow_natural.iloc[start]
        flow_final.iloc[start:end] = freeze_val + rng.normal(0, 0.5, end - start)
    df["pump_flow_rate_ls"] = np.clip(flow_final, 80, 220)

    conductivity = (
        350 + 60 * (df["cooling_tower_approach_c"] - 3.5) + 80 * df["_t_frac"]
        + rng.normal(0, 20, n)
    )
    df["water_conductivity_us"] = np.clip(conductivity, 200, 900)

    pue = 1.05 + 0.108 * (6.5 - df["chiller_cop"]) + rng.normal(0, 0.02, n)
    df["pue"] = np.clip(pue, 1.05, 1.45)

    return df


# ---------------------------------------------------------------------------
# 7. cooling_health_score: nonlinear, noisy composite of 6 normalized terms
# ---------------------------------------------------------------------------
def compute_chs(df):
    n = len(df)
    cop_score = (df["chiller_cop"] - 2.8) / (6.5 - 2.8)
    approach_score = 1 - (df["cooling_tower_approach_c"] - 2) / (8 - 2)
    vib_score = 1 - (df["pump_vibration_mms"] - 0.5) / (4.5 - 0.5)
    flow_score = (df["pump_flow_rate_ls"] - 80) / (220 - 80)
    cond_score = 1 - (df["water_conductivity_us"] - 200) / (900 - 200)
    pue_score = 1 - (df["pue"] - 1.05) / (1.45 - 1.05)
    inlet_norm = (df["chiller_inlet_temp_c"] - 6) / (14 - 6)
    crah_norm = (df["crah_delta_t_c"] - 8) / (18 - 8)
    thermal_score = 1 - (0.5 * inlet_norm + 0.5 * crah_norm)

    weights = dict(cop=0.20, approach=0.13, vib=0.13, flow=0.13,
                    cond=0.13, pue=0.14, thermal=0.14)
    composite = (
        weights["cop"] * cop_score + weights["approach"] * approach_score
        + weights["vib"] * vib_score + weights["flow"] * flow_score
        + weights["cond"] * cond_score + weights["pue"] * pue_score
        + weights["thermal"] * thermal_score
    )

    worst = pd.concat([cop_score, approach_score, vib_score, thermal_score], axis=1).min(axis=1)
    compounding_penalty = (1 - composite).clip(lower=0) * (1 - worst).clip(lower=0) * 0.35

    chs_unit = (composite - compounding_penalty).clip(0, 1)
    chs = 55 + chs_unit * 43 + rng.normal(0, 2.5, n)
    df["cooling_health_score"] = chs.clip(55, 98)
    return df


# ---------------------------------------------------------------------------
# 7b. Background anomaly layer -- isolated sensor-drift perturbations
# ---------------------------------------------------------------------------
def inject_background_anomalies(df):
    """
    Reconciles Section 1.2's "~3% prevalence" with Section 1.3's 8-event
    construction rule (per instructor guidance): adds a second, diffuse
    anomaly population on top of the 8 structured events already injected.

    ~2.7% of currently-normal rows are flagged is_anomaly=1. For each, 3
    randomly chosen telemetry columns get Gaussian noise at 1.5-2x that
    column's normal-population standard deviation, then are re-clipped to
    the spec range. This represents sensor drift / marginal exceedances,
    NOT a real physical event: cooling_health_score is deliberately left
    untouched, since the underlying system health hasn't actually changed,
    only the sensor reading has. anomaly_class distinguishes these from
    the "structured" population for Day 5's analysis of whether
    IsolationForest can separate the two.
    """
    n = len(df)
    df["anomaly_class"] = np.where(df["is_anomaly"] == 1, "structured", "none")

    normal_idx = df.index[df["is_anomaly"] == 0].to_numpy()
    n_background = int(round(BACKGROUND_ANOMALY_FRACTION * len(normal_idx)))
    background_idx = rng.choice(normal_idx, size=n_background, replace=False)

    # normal-population std, computed once before perturbation
    normal_std = {c: df.loc[df["is_anomaly"] == 0, c].std() for c in BACKGROUND_PERTURBATION_FEATURES}

    feature_perturb_counts = {c: 0 for c in BACKGROUND_PERTURBATION_FEATURES}
    for idx in background_idx:
        chosen = rng.choice(BACKGROUND_PERTURBATION_FEATURES, size=3, replace=False)
        for feat in chosen:
            scale = rng.uniform(1.5, 2.0)
            noise = rng.normal(0, scale * normal_std[feat])
            lo, hi = SPEC_RANGES[feat]
            df.loc[idx, feat] = np.clip(df.loc[idx, feat] + noise, lo, hi)
            feature_perturb_counts[feat] += 1

    df.loc[background_idx, "is_anomaly"] = 1
    df.loc[background_idx, "anomaly_type"] = "background_sensor_drift"
    df.loc[background_idx, "anomaly_class"] = "background"

    bg_log = {
        "n_background_rows": n_background,
        "n_normal_rows_pool": len(normal_idx),
        "feature_perturb_counts": feature_perturb_counts,
    }
    return df, bg_log


# ---------------------------------------------------------------------------
# 8. Orchestration
# ---------------------------------------------------------------------------
FINAL_COLUMNS = [
    "timestamp", "outdoor_temp_c", "outdoor_humidity_pct", "it_load_mw",
    "chiller_inlet_temp_c", "chiller_outlet_temp_c", "chiller_cop",
    "cooling_tower_approach_c", "crah_delta_t_c", "pump_vibration_mms",
    "pump_flow_rate_ls", "water_conductivity_us", "pue",
    "cooling_health_score", "is_anomaly", "anomaly_type", "anomaly_class",
]


def generate_dataset():
    df = build_time_index()
    df = gen_weather(df)
    df = gen_it_load(df)
    df = gen_system_roots(df)
    df, log = inject_anomalies(df)
    df = derive_system(df)
    df = compute_chs(df)
    df, bg_log = inject_background_anomalies(df)
    return df[FINAL_COLUMNS].copy(), log, bg_log


if __name__ == "__main__":
    out, log, bg_log = generate_dataset()
    out.to_csv("cooling_telemetry_doha_dc1.csv", index=False)

    print(f"Rows generated: {len(out):,}")
    print(f"Total anomaly rate: {out['is_anomaly'].mean():.3%}  "
          f"({out['is_anomaly'].sum()} of {len(out)} rows)")
    print(out["anomaly_class"].value_counts())

    print("\nStructured anomaly schedule (8 events):")
    for a_type, t0, t1, dur_steps in log:
        print(f"  {a_type:<28} {t0} -> {t1}  ({dur_steps / STEPS_PER_HOUR:.1f} hrs)")

    print(f"\nBackground anomaly layer: {bg_log['n_background_rows']} rows "
          f"sampled from {bg_log['n_normal_rows_pool']} remaining normal rows "
          f"({bg_log['n_background_rows'] / bg_log['n_normal_rows_pool']:.3%})")
    print("Feature perturbation counts (3 features chosen per background row):")
    for feat, count in sorted(bg_log["feature_perturb_counts"].items(), key=lambda x: -x[1]):
        print(f"  {feat:<26} {count}")

    print("\nFeature ranges (min / max) vs spec:")
    for col, (lo, hi) in SPEC_RANGES.items():
        actual_lo, actual_hi = out[col].min(), out[col].max()
        ok = "OK" if (actual_lo >= lo and actual_hi <= hi) else "OUT OF RANGE"
        print(f"  {col:<26} spec [{lo},{hi}]  actual [{actual_lo:.2f},{actual_hi:.2f}]  {ok}")

    print("\nCHS: normal vs anomalous rows")
    print(out.groupby("is_anomaly")["cooling_health_score"].describe()[["mean", "std", "min", "max"]])

    print("\nCHS by anomaly_class (this is the key check: background should ~= normal,")
    print("structured should be clearly lower, confirming the two populations differ")
    print("in exactly the way intended)")
    print(out.groupby("anomaly_class")["cooling_health_score"].describe()[["mean", "std", "min", "max"]])

    print("\nCHS by anomaly type (structured events):")
    print(out[out["anomaly_class"] == "structured"].groupby("anomaly_type")["cooling_health_score"]
          .agg(["mean", "min", "max", "count"]))

    print(f"\nNaNs present: {out.isna().sum().sum()}")
