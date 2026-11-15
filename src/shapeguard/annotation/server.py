"""Local-only native-resolution bone-mask editor with explicit acceptance and freezing."""

import base64
import io
import json
import os
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from PIL import Image

from shapeguard.data.annotations import combine_masks
from shapeguard.data.manifest import read_manifest
from shapeguard.reproducibility import sha256


def save_annotation(manifest, image_id, png, accepted=False):
    path = Path(manifest).resolve()
    if (path.parent / "annotations_frozen.json").exists():
        raise ValueError("Annotations are frozen for experiments. Create a separately versioned cohort to change them.")
    raw = pd.read_csv(path).fillna("")
    frame = read_manifest(path)
    rows = frame[frame.image_id == image_id]
    if len(rows) != 1:
        raise ValueError("Unknown image ID")
    row = rows.iloc[0]
    if row.bone_status == "accepted":
        raise ValueError("This annotation is already accepted. Preserve it; create a new dataset version to revise it.")
    with Image.open(io.BytesIO(png)) as im:
        if im.size != (int(row.width), int(row.height)):
            raise ValueError("Annotation must match native image dimensions")
        bone = (np.asarray(im.convert("L")) > 127).astype(np.uint8)
    if accepted and (not bone.any() or bone.mean() > 0.95):
        raise ValueError("Empty/nearly full-image bone mask: correct the annotation before accepting")
    bone_dir, multi_dir = path.parent / "bone_masks", path.parent / "multiclass_masks"
    bone_dir.mkdir(exist_ok=True)
    multi_dir.mkdir(exist_ok=True)
    bone_path = bone_dir / f"{image_id}.png"
    mask_path = multi_dir / f"{image_id}.png"
    Image.fromarray(bone).save(bone_path)
    fracture = np.asarray(Image.open(row.fracture_path)) > 0
    Image.fromarray(combine_masks(bone, fracture)).save(mask_path)
    status = "accepted" if accepted else "draft"
    index = raw.index[raw.image_id == image_id][0]
    raw.loc[index, "bone_path"] = os.path.relpath(bone_path, path.parent)
    raw.loc[index, "mask_path"] = os.path.relpath(mask_path, path.parent)
    raw.loc[index, "bone_status"] = status
    temporary = path.with_suffix(".partial")
    raw.to_csv(temporary, index=False)
    temporary.replace(path)
    record = {
        "image_id": image_id, "status": status, "source": "student/AI-assisted annotation",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bone_sha256": sha256(bone_path), "multiclass_sha256": sha256(mask_path),
        "bone_fraction": float(bone.mean()),
        "fracture_outside_bone_pixels": int((fracture & ~bone.astype(bool)).sum()),
        "note": "Original fracture regions preserved with fracture-class precedence; no clinical validation.",
    }
    # Append history so draft provenance is not lost after subsequent corrections.
    history = bone_dir / f"{image_id}.history.jsonl"
    with history.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    return record


def serve(manifest, selection, port=8765):
    manifest = Path(manifest).resolve()
    ids = json.loads(Path(selection).read_text())["image_ids"]
    frame = read_manifest(manifest)
    if not set(ids).issubset(set(frame.image_id)):
        raise ValueError("Selection contains unknown images")
    html = Path(__file__).with_name("editor.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                return self.respond(200, html, "text/html; charset=utf-8")
            if parsed.path == "/images":
                current = read_manifest(manifest)
                records = current[current.image_id.isin(ids)][["image_id", "width", "height", "bone_status", "split"]].to_dict("records")
                return self.respond(200, json.dumps(records).encode())
            if parsed.path in {"/image", "/bone", "/fracture"}:
                image_id = parse_qs(parsed.query).get("id", [""])[0]
                if image_id not in ids:
                    return self.respond(404, b'{}')
                row = read_manifest(manifest).set_index("image_id").loc[image_id]
                if parsed.path == "/image":
                    im = Image.open(row.image_path).convert("L")
                elif parsed.path == "/fracture":
                    im = Image.fromarray((np.asarray(Image.open(row.fracture_path)) > 0).astype(np.uint8) * 255)
                else:
                    proposal = manifest.parent / "proposals" / f"{image_id}.png"
                    bone = Path(row.bone_path) if row.bone_path else proposal
                    im = Image.open(bone).convert("L") if bone.exists() else Image.new("L", (int(row.width), int(row.height)))
                    im = Image.fromarray((np.asarray(im) > 0).astype(np.uint8) * 255)
                buffer = io.BytesIO()
                im.save(buffer, format="PNG")
                return self.respond(200, buffer.getvalue(), "image/png")
            self.respond(404, b'{}')

        def do_POST(self):
            if self.path != "/save":
                return self.respond(404, b'{}')
            # Reject cross-origin browser writes to the local annotation service.
            origin = self.headers.get("Origin")
            if origin not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self.respond(403, b'{"error":"Invalid origin"}')
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 25_000_000:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if payload["image_id"] not in ids:
                    raise ValueError("Image not in selected cohort")
                encoded = payload["png"].split(",", 1)[1]
                result = save_annotation(manifest, payload["image_id"], base64.b64decode(encoded, validate=True), payload.get("accepted") is True)
                self.respond(200, json.dumps(result).encode())
            except (ValueError, KeyError, IndexError, OSError) as error:
                self.respond(400, json.dumps({"error": str(error)}).encode())

    print(f"Bone-mask editor: http://127.0.0.1:{port} (Ctrl-C to stop)", flush=True)
    server = HTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
