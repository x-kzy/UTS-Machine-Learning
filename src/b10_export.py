"""Ekspor hasil run tanpa mengubah fitur, estimator, ambang, atau prediksi."""
from pathlib import Path
import hashlib
import json

TABLE_NAMES = [
    "schema_table", "source_discrepancies", "sensor_dictionary", "FAILURES", "split_plan",
    "evaluation_protocol", "train_summary", "digital_frequency", "train_corr", "iqr_table",
    "findings", "preprocessing_plan", "window_summary", "quality_reasons", "feature_definitions",
    "constant_feature_audit", "k_scan", "profile", "cluster_interpretation", "pca_variance",
    "model_settings", "event_coverage", "cv_results", "cv_summary", "validation_metrics",
    "development_blocks", "validation_events", "model_tradeoffs", "feature_ablation", "sensitivity",
    "test_metrics", "test_events", "error_slices", "validation_alarm_segments", "deviation_profile",
    "drift_diagnostics", "integrity_checks", "s13_progress", "s14_goals", "demo_prediction",
]

def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "get_params"):
        return repr(value)
    raise TypeError(type(value).__name__)

def export_run(namespace):
    import pandas as pd
    folder = Path(namespace["OUTPUT_DIR"])
    tables = folder / "tables"
    tables.mkdir(exist_ok=True)
    inventory = []
    for name in TABLE_NAMES:
        value = namespace.get(name)
        if isinstance(value, pd.DataFrame):
            target = tables / (name + ".csv")
            value.to_csv(target, index=True)
            inventory.append({"name": name, "rows": len(value), "columns": list(map(str, value.columns))})
    predictions = []
    for (partition, model), prediction in namespace["PREDICTIONS"].items():
        part = prediction.join(namespace["meta"][["y_report", "event_number", "n_observations"]]).copy()
        part.insert(0, "model", model)
        part.insert(0, "split", partition)
        predictions.append(part.reset_index())
    pd.concat(predictions, ignore_index=True).to_csv(folder / "scores_and_alarms.csv.gz", index=False, compression="gzip")
    namespace["demo_raw"].to_csv(folder / "demo_raw_sensor.csv", index=True)
    payloads = {
        "environment.json": namespace["VERSIONS"],
        "data_audit.json": namespace["AUDIT"],
        "experiment_log.json": namespace["EXPERIMENT_LOG"],
        "configuration.json": {
            "project_id": "B10", "run_id": namespace["RUN_ID"],
            "selected_model": namespace["SELECTED_MODEL"],
            "feature_columns": namespace["FEATURE_COLUMNS"],
            "feature_config": namespace["FEATURE_CONFIG"],
            "seed": namespace["SEED"], "base_k": namespace["BASE_K"],
            "quantile": namespace["ALERT_QUANTILE"], "persistence": namespace["PERSISTENCE_WINDOWS"],
            "thresholds": namespace["FROZEN_THRESHOLDS"],
            "selection_basis": "AP proxy validation, then event recall, F1, model name",
            "benchmark_previously_inspected": True,
            "repository_url": "https://github.com/x-kzy/UTS-Machine-Learning",
        },
        "export_inventory.json": inventory,
    }
    for name, payload in payloads.items():
        (folder / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    checksums = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.relative_to(folder).as_posix())
    (folder / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")
    print("Hasil run diekspor ke:", folder)
    return folder
