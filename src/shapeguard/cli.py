"""Command-line entry point. Heavy training imports are loaded only when needed."""

import argparse
import json
from pathlib import Path

from shapeguard.reproducibility import json_safe, save_json


def select_images(manifest, output, count=20):
    import numpy as np

    from shapeguard.data.manifest import read_manifest
    frame = read_manifest(manifest)
    if count < 6:
        raise ValueError("Select at least six images to cover both fracture classes in all splits")
    pool = frame[(frame.hardware == 0) & (frame.multiscan == 0)]
    limbs = pool[(pool.leg == 1) | (pool.hand == 1)]
    if len(limbs) >= count and limbs.groupby(["split", "fractured"]).ngroups == 6:
        pool = limbs
    rng = np.random.default_rng(42)
    selected = []
    for split, amount in (("train", count - 2 * max(2, round(count * 0.15))), ("val", max(2, round(count * 0.15))), ("test", max(2, round(count * 0.15)))):
        part = pool[pool.split == split]
        for positive, n in ((1, (amount + 1) // 2), (0, amount // 2)):
            group = part[part.fractured == positive].drop_duplicates("group_id")
            if len(group) < n:
                raise ValueError(f"Insufficient {split} images with fractured={positive}")
            ids = group.image_id.to_numpy().copy()
            rng.shuffle(ids)
            selected.extend(ids[:n].tolist())
    save_json(output, {"image_ids": selected, "seed": 42, "selection": "source-attribute and split-stratified pilot, before model prediction"})
    return selected


def audit_ready(manifest, size, task="binary"):
    from shapeguard.data.dataset import FractureDataset
    summary = {}
    for split in ("train", "val", "test"):
        data = FractureDataset(manifest, split, task=task, size=size)
        vanished, small, total = 0, 0, 0
        for i in range(len(data)):
            sample = data[i]
            vanished += sample["meta"]["vanished_instances"]
            for mask in sample["instances"]:
                total += 1
                small += int(mask.sum() <= 4)
        summary[split] = {"eligible_images": len(data), "positive_images": int(data.frame.fractured.sum()),
                          "instances": total, "vanished_after_resize": vanished, "instances_at_most_4_pixels": small}
        if not data.frame.fractured.sum():
            raise ValueError(f"Accepted {split} cohort has no positive fractures")
    save_json(Path(manifest).parent / f"readiness_{size}.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="ShapeGuard-Frac research CLI; not a clinical tool")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Inspect local Python/PyTorch/device environment")
    download = commands.add_parser("download")
    download.add_argument("--root", default="data/raw")
    download.add_argument("--version", type=int, default=7)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", required=True)
    prepare.add_argument("--output", default="data/prepared")
    split = commands.add_parser("split")
    split.add_argument("--manifest", default="data/prepared/manifest.csv")
    split.add_argument("--seed", type=int, default=42)
    select = commands.add_parser("select")
    select.add_argument("--manifest", default="data/prepared/manifest.csv")
    select.add_argument("--count", type=int, default=20)
    select.add_argument("--output", default="data/prepared/pilot.json")
    proposals = commands.add_parser("propose")
    proposals.add_argument("--manifest", default="data/prepared/manifest.csv")
    proposals.add_argument("--selection", default="data/prepared/pilot.json")
    proposals.add_argument("--method", choices=["threshold", "sam"], required=True)
    proposals.add_argument("--sam-checkpoint")
    proposals.add_argument("--prompts")
    proposals.add_argument("--sam-model", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    proposals.add_argument("--device", default="auto")
    editor = commands.add_parser("annotate")
    editor.add_argument("--manifest", default="data/prepared/manifest.csv")
    editor.add_argument("--selection", default="data/prepared/pilot.json")
    editor.add_argument("--port", type=int, default=8765)
    ready = commands.add_parser("audit-ready")
    ready.add_argument("--manifest", default="data/prepared/manifest.csv")
    ready.add_argument("--size", type=int, default=384)
    ready.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    eda = commands.add_parser("eda")
    eda.add_argument("--manifest", default="data/prepared/manifest.csv")
    eda.add_argument("--selection")
    training = commands.add_parser("train")
    training.add_argument("--config", required=True)
    training.add_argument("--resume", action="store_true")
    testing = commands.add_parser("evaluate")
    testing.add_argument("--checkpoint", required=True)
    testing.add_argument("--device", default="auto")
    testing.add_argument("--split", choices=["val", "test"], default="test")
    testing.add_argument("--robustness", action="store_true")
    plots = commands.add_parser("figures")
    plots.add_argument("--checkpoint", required=True)
    plots.add_argument("--evaluation", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--evaluations", nargs="+", required=True)
    compare.add_argument("--output", required=True)
    bootstrap = commands.add_parser("bootstrap")
    bootstrap.add_argument("--first", required=True)
    bootstrap.add_argument("--second", required=True)
    bootstrap.add_argument("--metric", default="dice")
    bootstrap.add_argument("--samples", type=int, default=2000)
    bootstrap.add_argument("--positive-only", action="store_true")
    bootstrap.add_argument("--output", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="auto")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--output", default="runs/smoke")
    smoke.add_argument("--device", default="cpu")
    study = commands.add_parser("study")
    study.add_argument("--configs", default="configs/gpu")
    study.add_argument("--output", default="results/binary_study")
    study.add_argument("--core-only", action="store_true")
    study.add_argument("--repeat-seeds", type=int, nargs="*", default=[])
    reports = commands.add_parser("deliverables")
    reports.add_argument("--study", default="results/binary_study")
    reports.add_argument("--manifest", default="data/prepared/manifest.csv")
    reports.add_argument("--output", default="deliverables")
    reports.add_argument("--author")
    reports.add_argument("--roll-number")
    reports.add_argument("--submission", default="submission.json")
    reports.add_argument("--report-only", action="store_true")
    poster = commands.add_parser("poster")
    poster.add_argument("--study", default="results/binary_study")
    poster.add_argument("--manifest", default="data/prepared/manifest.csv")
    poster.add_argument("--output", default="deliverables")
    poster.add_argument("--template")
    poster.add_argument("--submission", default="submission.json")
    exporting = commands.add_parser("export-metrics")
    exporting.add_argument("--study", default="results/binary_study")
    args = parser.parse_args()
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")
    if result is not None:
        print(json.dumps(json_safe(result), indent=2, default=str))


def dispatch(args):
    if args.command == "doctor":
        from shapeguard.reproducibility import choose_device, environment
        return environment(choose_device())
    if args.command == "download":
        from shapeguard.data.download import download_fracatlas
        return str(download_fracatlas(args.root, args.version))
    if args.command == "prepare":
        from shapeguard.data.audit import prepare
        return str(prepare(args.source, args.output))
    if args.command == "split":
        from shapeguard.data.splits import make_splits
        return make_splits(args.manifest, args.seed).to_dict("index")
    if args.command == "select":
        return select_images(args.manifest, args.output, args.count)
    if args.command == "propose":
        from shapeguard.annotation.proposals import generate
        return generate(args.manifest, args.selection, args.method, args.sam_checkpoint, args.prompts, args.sam_model, args.device)
    if args.command == "annotate":
        from shapeguard.annotation.server import serve
        return serve(args.manifest, args.selection, args.port)
    if args.command == "audit-ready":
        return audit_ready(args.manifest, args.size, args.task)
    if args.command == "eda":
        from shapeguard.data.eda import eda
        return str(eda(args.manifest, args.selection))
    if args.command == "train":
        from shapeguard.config import load_config
        from shapeguard.engine.train import train
        return str(train(load_config(args.config), args.resume))
    if args.command == "evaluate":
        from shapeguard.engine.evaluate import evaluate
        return str(evaluate(args.checkpoint, args.device, args.split, args.robustness))
    if args.command == "figures":
        from shapeguard.reporting.figures import figures
        return str(figures(args.checkpoint, args.evaluation))
    if args.command == "compare":
        from shapeguard.reporting.tables import comparison
        return str(comparison(args.evaluations, args.output))
    if args.command == "bootstrap":
        from shapeguard.evaluation.statistics import paired_bootstrap
        result = paired_bootstrap(args.first, args.second, args.metric, args.samples, positive_only=args.positive_only)
        save_json(args.output, result)
        return result
    if args.command == "predict":
        from shapeguard.engine.predict import predict
        return str(predict(args.checkpoint, args.image, args.output, args.device))
    if args.command == "smoke":
        from shapeguard.smoke import smoke
        return smoke(args.output, args.device)
    if args.command == "study":
        from shapeguard.study import run_study
        return str(run_study(args.configs, args.output, not args.core_only, args.repeat_seeds))
    if args.command == "deliverables":
        from shapeguard.reporting.deliverables import deliverables
        metadata_path = Path(args.submission)
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        if metadata.get("ppt_guidelines_pending") and not args.report_only:
            raise ValueError("Revised PPT guidelines are pending. Use --report-only until the presentation specification is confirmed.")
        author = args.author or metadata.get("author")
        roll = args.roll_number or metadata.get("roll_number")
        if roll and not all(character.isalnum() or character in "_-" for character in roll):
            raise ValueError("Roll number must be safe for a filename (letters, numbers, underscores or hyphens)")
        poster_mode = metadata.get("deliverable") == "poster"
        result = deliverables(args.study, args.manifest, args.output, author, roll, args.report_only or poster_mode)
        if poster_mode and not args.report_only:
            from shapeguard.reporting.poster import poster
            result.update(poster(args.study, args.manifest, args.output, metadata["poster_template"], author, roll, metadata.get("email", ""), metadata.get("affiliation", "")))
        return result
    if args.command == "poster":
        from shapeguard.reporting.poster import poster
        metadata = json.loads(Path(args.submission).read_text())
        return poster(args.study, args.manifest, args.output, args.template or metadata["poster_template"], metadata["author"], metadata["roll_number"], metadata.get("email", ""), metadata.get("affiliation", ""))
    if args.command == "export-metrics":
        from shapeguard.reporting.export import export_metrics
        return str(export_metrics(args.study))
