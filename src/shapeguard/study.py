"""Sequential, resumable study execution with real-artifact status tracking."""

import json
import time
from pathlib import Path

from shapeguard.config import load_config
from shapeguard.engine.checkpoint import load_checkpoint
from shapeguard.engine.evaluate import evaluate
from shapeguard.engine.train import train
from shapeguard.evaluation.statistics import paired_bootstrap
from shapeguard.reporting.figures import figures
from shapeguard.reporting.tables import comparison
from shapeguard.reproducibility import save_json

CORE = ["baseline", "ici", "geometry_square", "geometry", "boundary_fixed", "boundary_adaptive", "proposed"]
OPTIONAL = ["contrast_control", "full", "attention"]


def run_study(config_directory="configs/gpu", output="results/binary_study", include_optional=True, repeat_seeds=None):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "study_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"runs": [], "completed": False}
    configs = [Path(config_directory) / f"{name}.yaml" for name in CORE + (OPTIONAL if include_optional else [])]
    work = [(path, None) for path in configs]
    for seed in repeat_seeds or []:
        work.extend((Path(config_directory) / f"{name}.yaml", seed) for name in ("ici", "proposed"))
    protocol = {"configurations": [load_config(path) for path in configs], "repeat_seeds": list(repeat_seeds or []),
                "selection": "best validation fracture-positive Dice; no test-driven selection",
                "execution": "one GPU process, sequential experiments", "budget_note": "Fixed 16-epoch study budget selected after runtime preflight, before comparative test evaluation."}
    protocol_path = directory / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("Study protocol changed; use a new study output directory")
    save_json(protocol_path, protocol)
    evaluations = []
    for path, seed in work:
        config = load_config(path)
        if seed is not None:
            config["seed"] = seed
        run = Path(config["train"]["output"]) / config["name"] / f"seed_{config['seed']}"
        entry = {"experiment": config["name"], "seed": config["seed"], "status": "running", "config": str(path)}
        previous = next((r for r in status["runs"] if (r["experiment"], r["seed"]) == (entry["experiment"], entry["seed"])), None)
        status["active"] = entry
        status["completed"] = False
        save_json(status_path, status)
        start = time.perf_counter()
        try:
            last = run / "last.pt"
            complete = False
            if last.exists():
                state = load_checkpoint(last)
                complete = state["epoch"] + 1 >= config["train"]["epochs"] or state["stale"] >= config["train"]["patience"]
                if state["config"] != config:
                    raise ValueError(f"Existing study run configuration differs: {run}")
            if not complete:
                train(config, resume=last.exists())
            evaluated = run / "evaluation_test"
            if not (evaluated / "summary.json").exists():
                # Evaluate once at a time. No parallel GPU consumers are launched.
                evaluated = evaluate(run / "best.pt", "cuda" if config["device"] == "cuda" else config["device"], robustness=path.stem in {"ici", "proposed", "contrast_control", "full"} and seed is None)
            figures(run / "best.pt", evaluated)
            evaluations.append(str(evaluated))
            elapsed = time.perf_counter() - start
            entry.update(status="completed", seconds=previous["seconds"] if complete and previous else elapsed,
                         seconds_this_invocation=elapsed, reused_existing_run=complete, evaluation=str(evaluated))
            status["runs"] = [r for r in status["runs"] if (r["experiment"], r["seed"]) != (entry["experiment"], entry["seed"])] + [entry]
            save_json(status_path, status)
            print(json.dumps(entry), flush=True)
        except Exception as error:
            entry.update(status="failed", error=str(error))
            status["active"] = entry
            save_json(status_path, status)
            raise
    comparison(evaluations, directory / "comparison.csv")
    # The main seed-42 comparison, and repetitions where present, each pair same-seed runs.
    for seed in [42] + list(repeat_seeds or []):
        pairs = {
            "geometry_vs_ici": ("ici_2d_adaptation", "geometry"),
            "adaptive_geometry_vs_square": ("geometry_square_control", "geometry"),
            "adaptive_vs_fixed": ("boundary_fixed", "boundary_adaptive"),
            "proposed_vs_ici": ("ici_2d_adaptation", "proposed"),
            "consistency_vs_control": ("contrast_control", "full"),
        }
        for label, (a, b) in pairs.items():
            root = Path(load_config(configs[0])["train"]["output"])
            first = root / a / f"seed_{seed}" / "evaluation_test" / "fracture_metrics.csv"
            second = root / b / f"seed_{seed}" / "evaluation_test" / "fracture_metrics.csv"
            if first.exists() and second.exists():
                save_json(directory / f"{label}_seed_{seed}_bootstrap.json", paired_bootstrap(first, second, positive_only=True))
    status["completed"] = True
    status["active"] = None
    status["evaluations"] = evaluations
    save_json(status_path, status)
    return directory
