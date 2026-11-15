import json
from pathlib import Path

import pandas as pd


def comparison(evaluations, output):
    summaries = [json.loads((Path(p) / "summary.json").read_text()) for p in evaluations]
    if len({(s["cohort_sha256"], s["task"], s["split"]) for s in summaries}) != 1:
        raise ValueError("Cannot compare runs with different cohorts/tasks/splits")
    devices = {(s["efficiency"]["device"], s["efficiency"]["environment"]["platform"],
                s["efficiency"]["environment"]["gpu"], s["efficiency"]["environment"]["packages"]["torch"],
                s["efficiency"]["input_size"]) for s in summaries}
    rows = []
    for path, summary in zip(evaluations, summaries):
        row = {k: summary[k] for k in ("experiment", "seed", "task", "split", "fracture_positive_dice", "fracture_positive_iou", "negative_image_fp_rate")}
        row.update({k: summary["efficiency"][k] for k in ("parameters", "conv_flops", "latency_median_ms", "checkpoint_bytes", "device")})
        class_table = pd.read_csv(Path(path) / "class_summary.csv")
        for cls in class_table.itertuples(index=False, name=None):
            for key, value in zip(class_table.columns, cls):
                if key != "class":
                    row[f"{cls[0]}_{key}"] = value
        row["latency_comparable"] = len(devices) == 1
        rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output
