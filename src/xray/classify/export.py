"""Export the fine-tuned classifier for CPU serving within a 512 MB instance.

Measured on this machine: `import torch` alone is ~200 MB RSS and the fp32 DistilBERT ~660 MB,
over Render's free-tier memory. Serving uses ONNX Runtime with an int8 dynamically quantized
graph and the `tokenizers` library instead, so torch/transformers are not needed at runtime.
Quantization can change predictions, so `parity` re-scores the real test split with the int8
model and reports agreement with the fp32 model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from xray.classify.predict import ONNX_FILE
from xray.classify.train import REPORT_DIR_NAME, evaluate, load_train_config
from xray.config import CONFIG_DIR, MODEL_DIR
from xray.store import Store

ONNX_DIR_NAME = "seniority-onnx"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_onnx(model_dir: Path = MODEL_DIR) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    src = model_dir / "seniority-distilbert"
    out = model_dir / ONNX_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(src).eval()
    tok = AutoTokenizer.from_pretrained(src)
    enc = tok("example", return_tensors="pt", truncation=True, max_length=16)
    fp32 = out / "model.fp32.onnx"
    torch.onnx.export(
        model,
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"},
                      "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )  # fmt: skip
    quantize_dynamic(str(fp32), str(out / ONNX_FILE), weight_type=QuantType.QInt8)
    fp32.unlink()
    tok.backend_tokenizer.save(str(out / "tokenizer.json"))
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    max_length = load_train_config(CONFIG_DIR).max_length
    (out / "labels.json").write_text(
        json.dumps({"labels": labels, "max_length": max_length}, indent=1), encoding="utf-8"
    )
    if (src / "README.md").exists():
        shutil.copy(src / "README.md", out / "README.md")
    return out


def parity(store: Store | None = None, model_dir: Path = MODEL_DIR) -> dict[str, Any]:
    """Re-score the real test split with the int8 ONNX model; compare with fp32 predictions."""
    import duckdb

    from xray.classify.predict import OnnxSeniorityPredictor
    from xray.classify.train import assign_splits, build_dataset

    store = store or Store()
    rows, _ = build_dataset(store)
    assign_splits(rows, load_train_config())
    test = [r for r in rows if r["split"] == "test"]
    onnx = OnnxSeniorityPredictor.load(model_dir / ONNX_DIR_NAME)
    preds = {r["posting_key"]: onnx.predict_prepared(r["text"])["prediction"] for r in test}
    path = store.root / REPORT_DIR_NAME / "seniority_test_predictions.parquet"
    query = f"SELECT posting_key, transformer_pred FROM '{path.as_posix()}'"
    fp32 = dict(duckdb.sql(query).fetchall())
    labels = onnx.labels
    agree = sum(preds[k] == fp32[k] for k in preds)
    return {
        "test_postings": len(test),
        "agreement_with_fp32": round(agree / len(test), 4),
        "disagreements": len(test) - agree,
        "int8_test_metrics": evaluate([r["target"] for r in test],
                                      [preds[r["posting_key"]] for r in test], labels, seed=42),
        "onnx_sha256": sha256(model_dir / ONNX_DIR_NAME / ONNX_FILE),
    }  # fmt: skip


def package(model_dir: Path = MODEL_DIR) -> tuple[Path, str]:
    """Zip the serving bundle (int8 ONNX + tokenizer + labels + card) for a release asset."""
    src = model_dir / ONNX_DIR_NAME
    out = model_dir / f"{ONNX_DIR_NAME}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in sorted(src.iterdir()):
            z.write(f, arcname=f.name)
    return out, sha256(out)
