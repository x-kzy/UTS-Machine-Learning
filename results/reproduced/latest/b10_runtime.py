import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

def build_completed_windows(sensor_frame, config, observed_until):
    """Agregasi [start,end); observed_until adalah batas waktu data telah tersedia."""
    analog, digital = config["analog"], config["digital"]
    required = analog + digital
    missing = sorted(set(required) - set(sensor_frame.columns))
    if missing:
        raise ValueError(f"Sensor wajib belum lengkap: {missing}")
    if not isinstance(sensor_frame.index, pd.DatetimeIndex):
        raise TypeError("Index sensor harus DatetimeIndex.")
    if sensor_frame.empty or not sensor_frame.index.is_unique or not sensor_frame.index.is_monotonic_increasing:
        raise ValueError("Sensor harus berisi timestamp unik, terurut, dan tidak kosong.")
    if sensor_frame.index.tz is not None:
        raise ValueError("Sumber tidak menyatakan zona waktu; gunakan waktu naive yang sama dengan sumber.")
    observed_until = pd.Timestamp(observed_until)
    if sensor_frame.index.max() >= observed_until:
        raise ValueError("Semua sampel harus lebih awal daripada observed_until.")
    source = sensor_frame[required].astype(float).replace([np.inf, -np.inf], np.nan).copy()
    for name in digital:
        source.loc[~source[name].isin([0., 1.]) & source[name].notna(), name] = np.nan
    delta = pd.Timedelta(config["window"])
    args = dict(rule=config["window"], label="right", closed="left", origin="start_day")
    a = source[analog].resample(**args).agg(["mean", "std"])
    a.columns = [f"{name}__{stat}" for name, stat in a.columns]
    d = source[digital].resample(**args).mean()
    d.columns = [f"{name}__duty" for name in digital]
    features = pd.concat([a, d], axis=1)
    features["pressure_gap_mean"] = (source.TP3 - source.Reservoirs).resample(**args).mean()
    load = pd.Series(np.where(source.Motor_current.notna(),
                              (source.Motor_current >= config["motor_load_ampere"]).astype(float), np.nan),
                     index=source.index)
    features["motor_load_fraction"] = load.resample(**args).mean()
    oil = source.Oil_temperature.resample(**args)
    features["oil_temperature_range"] = oil.max() - oil.min()
    features = features.replace([np.inf, -np.inf], np.nan)
    features.index.name = "window_end"
    meta = pd.DataFrame(index=features.index)
    meta["window_start"] = meta.index - delta
    meta["n_observations"] = source.TP2.resample(**args).size().astype(int)
    meta["coverage_ratio"] = (meta.n_observations * config["cadence_seconds"] / delta.total_seconds()).clip(upper=1.)
    times = source.index.to_series()
    bin_start = source.index.floor(config["window"])
    same_bin = pd.Series(bin_start == pd.Series(bin_start).shift().to_numpy(), index=source.index)
    internal_gap = times.diff().dt.total_seconds().where(same_bin)
    meta["max_internal_gap_seconds"] = internal_gap.resample(**args).max().fillna(0.)
    first = times.resample(**args).min()
    last = times.resample(**args).max()
    meta["first_sample"] = first
    meta["last_sample"] = last
    meta["left_gap_seconds"] = (first - meta.window_start).dt.total_seconds()
    meta["right_gap_seconds"] = (pd.Series(meta.index, index=meta.index) - last).dt.total_seconds()
    counts = source[required].resample(**args).count()
    fractions = 1. - counts.div(meta.n_observations.replace(0, np.nan), axis=0)
    meta["max_sensor_missing_fraction"] = fractions.max(axis=1)
    meta["completed"] = meta.index <= observed_until
    checks = {
        "insufficient_samples": meta.n_observations < config["min_observations"],
        "large_internal_gap": meta.max_internal_gap_seconds > config["max_gap_seconds"],
        "large_left_gap": meta.left_gap_seconds.gt(config["max_gap_seconds"]) | first.isna(),
        "large_right_gap": meta.right_gap_seconds.gt(config["max_gap_seconds"]) | last.isna(),
        "excess_sensor_missing": meta.max_sensor_missing_fraction.gt(config["max_sensor_missing_fraction"]) | fractions.isna().all(axis=1),
        "unfinished_window": ~meta.completed,
    }
    for reason, invalid in checks.items():
        meta[reason] = invalid
    meta["usable"] = ~meta[list(checks)].any(axis=1) & features.notna().any(axis=1)
    return features, meta


def anomaly_scores(pipeline, frame, cpu_threads=2):
    if frame.empty:
        return pd.Series(dtype=float, index=frame.index, name="score")
    with threadpool_limits(limits=cpu_threads):
        if isinstance(pipeline.named_steps["model"], KMeans):
            values = np.square(pipeline.transform(frame).min(axis=1))
        else:
            values = -pipeline.score_samples(frame)
    if not np.isfinite(values).all():
        raise ValueError("Skor anomali harus finite.")
    return pd.Series(values, index=frame.index, name="score")


def persistent_alarm(exceed, required=3, window="5min", state=None, return_state=False):
    """Alarm mulai pada jendela ke-required; reset saat flag 0 atau waktu terputus."""
    if required < 1:
        raise ValueError("required harus >=1.")
    if not isinstance(exceed.index, pd.DatetimeIndex) or not exceed.index.is_unique or not exceed.index.is_monotonic_increasing:
        raise ValueError("Timestamp alarm harus DatetimeIndex yang unik dan terurut.")
    if exceed.isna().any() or not exceed.isin([0, 1, False, True]).all():
        raise ValueError("Pelampauan harus biner dan tidak boleh kosong/NaN.")
    step = pd.Timedelta(window)
    old = {} if state is None else dict(state)
    previous = pd.Timestamp(old["last_timestamp"]) if old.get("last_timestamp") is not None else None
    run = int(old.get("run_length", 0))
    result = np.zeros(len(exceed), dtype=np.uint8)
    for i, (timestamp, flag) in enumerate(exceed.items()):
        if previous is not None and timestamp <= previous:
            raise ValueError("Batch baru harus lebih akhir daripada state terakhir.")
        if previous is not None and timestamp - previous != step:
            run = 0
        run = run + 1 if bool(flag) else 0
        result[i] = int(run >= required)
        previous = timestamp
    alarm = pd.Series(result, index=exceed.index, name="alarm")
    updated = {"last_timestamp": str(previous) if previous is not None else None, "run_length": run}
    return (alarm, updated) if return_state else alarm


def predict_bundle(bundle, frame, state=None, return_state=False):
    columns = bundle["feature_columns"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Fitur inferensi belum lengkap: {missing}")
    result = anomaly_scores(bundle["pipeline"], frame[columns], bundle.get("cpu_threads", 2)).to_frame()
    result["exceed"] = (result.score > bundle["threshold"]).astype("uint8")
    result["alarm"], updated = persistent_alarm(
        result.exceed, required=bundle.get("persistence_windows", 3),
        window=bundle.get("window", "5min"), state=state, return_state=True)
    return (result, updated) if return_state else result


def infer_raw_completed(sensor_frame, artifact, observed_until, state=None):
    features, quality = build_completed_windows(sensor_frame, artifact["feature_config"], observed_until)
    usable = features.loc[quality.usable]
    prediction, updated = predict_bundle(artifact, usable, state=state, return_state=True)
    return prediction, quality, updated
