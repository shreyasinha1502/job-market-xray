"""Load the seniority model once and classify pasted job descriptions.

Input goes through the same preparation as training: role sections (requirements first) with
every labeling-rule pattern removed, so the model sees what it was trained on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xray.classify.labels import load_label_rules, model_input
from xray.classify.train import MODEL_DIR
from xray.config import CONFIG_DIR


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
            "scope_note": "trained only to separate senior from below_senior (mid/junior/intern); "
            "intern and junior had too few real labelled postings to model separately",
        }
