"""Load the seniority model once and classify pasted job descriptions.

Input goes through the same preparation as training: role sections (requirements first) with
every labeling-rule pattern removed, so the model sees what it was trained on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xray.classify.labels import load_label_rules, model_input
from xray.config import CONFIG_DIR, MODEL_DIR

ONNX_FILE = "model.int8.onnx"
SCOPE_NOTE = (
    "trained only to separate senior from below_senior (mid/junior/intern); intern and junior "
    "had too few real labelled postings to model separately"
)


@dataclass
class SeniorityPredictor:
    kind: str  # "distilbert" | "tfidf_logreg"
    model: Any
    tokenizer: Any
    labels: list[str]
    max_length: int
    config_dir: Path

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR, config_dir: Path = CONFIG_DIR,
             prefer: str = "distilbert") -> SeniorityPredictor:  # fmt: skip
        bert = model_dir / "seniority-distilbert"
        if prefer == "distilbert" and (bert / "config.json").exists():
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model = AutoModelForSequenceClassification.from_pretrained(bert).eval()
            labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
            tok = AutoTokenizer.from_pretrained(bert)
            return cls("distilbert", model, tok, labels, 256, config_dir)
        import joblib

        pipe = joblib.load(model_dir / "seniority-tfidf-logreg.joblib")
        return cls("tfidf_logreg", pipe, None, list(pipe.classes_), 0, config_dir)

    def predict(self, description: str) -> dict[str, Any]:
        text = model_input(description, load_label_rules(self.config_dir))
        if not text:
            return {"error": "no usable text after preparation", "model": self.kind}
        return self.predict_prepared(text)

    def predict_prepared(self, text: str) -> dict[str, Any]:
        """Classify text that already went through `model_input` (training-time inputs)."""
        if self.kind == "distilbert":
            import torch

            enc = self.tokenizer(text, truncation=True, max_length=self.max_length,
                                 return_tensors="pt")  # fmt: skip
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, -1)[0].tolist()
        else:
            probs = self.model.predict_proba([text])[0].tolist()
        dist = dict(zip(self.labels, [round(p, 4) for p in probs], strict=True))
        return {
            "model": self.kind,
            "prediction": max(dist, key=dist.get),
            "probabilities": dist,
            "input_chars_used": len(text),
            "scope_note": SCOPE_NOTE,
        }


@dataclass
class OnnxSeniorityPredictor:
    """Serving path: int8 ONNX + `tokenizers`, no torch. Load once, reuse for every request."""

    session: Any
    tokenizer: Any
    labels: list[str]
    config_dir: Path
    kind: str = "distilbert-int8-onnx"

    @classmethod
    def load(cls, bundle_dir: Path, config_dir: Path = CONFIG_DIR) -> OnnxSeniorityPredictor:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        meta = json.loads((bundle_dir / "labels.json").read_text("utf-8"))
        tok = Tokenizer.from_file(str(bundle_dir / "tokenizer.json"))
        tok.enable_truncation(max_length=int(meta["max_length"]))
        tok.no_padding()
        session = ort.InferenceSession(str(bundle_dir / ONNX_FILE),
                                       providers=["CPUExecutionProvider"])  # fmt: skip
        return cls(session, tok, list(meta["labels"]), config_dir)

    def predict(self, description: str) -> dict[str, Any]:
        text = model_input(description, load_label_rules(self.config_dir))
        if not text:
            return {"error": "no usable text after preparation", "model": self.kind}
        return self.predict_prepared(text)

    def predict_prepared(self, text: str) -> dict[str, Any]:
        import numpy as np

        enc = self.tokenizer.encode(text)
        feed = {
            "input_ids": np.array([enc.ids], dtype=np.int64),
            "attention_mask": np.array([enc.attention_mask], dtype=np.int64),
        }
        logits = self.session.run(["logits"], feed)[0][0]
        e = np.exp(logits - logits.max())
        probs = (e / e.sum()).tolist()
        dist = dict(zip(self.labels, [round(float(p), 4) for p in probs], strict=True))
        return {
            "model": self.kind,
            "prediction": max(dist, key=dist.get),
            "probabilities": dist,
            "input_chars_used": len(text),
            "input_tokens_used": len(enc.ids),
            "scope_note": SCOPE_NOTE,
        }
