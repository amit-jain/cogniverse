"""Multi-head query classifier: one encoder pass, one softmax label per head.

A checkpoint directory holds the fine-tuned encoder and tokenizer under
``encoder/``, the head weights in ``heads.safetensors`` and the head label
vocabularies in ``heads.json``. ``cogniverse_finetuning`` trains and writes
it; the inference server in
``cogniverse_cli.modal_inference.servers.query_classifier`` and the in-process
loader in ``cogniverse_core.common.models.query_classifier`` read it. Both
answer ``{head: {"label": str, "score": float}}``.

torch, transformers and safetensors are imported only where a model is built,
so the validators load without them.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

HEADS_FILE = "heads.json"
HEAD_WEIGHTS_FILE = "heads.safetensors"
ENCODER_DIR = "encoder"

Predictions = Dict[str, Dict[str, Any]]


def validate_heads(heads: Any) -> Dict[str, List[str]]:
    """``heads`` as ``{name: [label, ...]}``; raises on any other shape."""
    if not isinstance(heads, dict) or not heads:
        raise ValueError("heads must be a non-empty object of label lists")
    for name, labels in heads.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"head name {name!r} must be a non-empty string")
        if (
            not isinstance(labels, list)
            or len(labels) < 2
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError(
                f"head {name!r} must list at least two distinct label strings"
            )
    return {name: list(labels) for name, labels in heads.items()}


def validate_predictions(
    data: Any, heads: Mapping[str, List[str]], source: str
) -> Predictions:
    """``data`` as one ``{"label", "score"}`` per head of ``heads``.

    Raises when a head is missing or extra, a label is outside its head's
    vocabulary, or a score is not a probability.
    """
    if not isinstance(data, dict):
        raise ValueError(f"{source} predictions must be an object")
    if set(data) != set(heads):
        raise ValueError(
            f"{source} predictions cover heads {sorted(data)}, expected {sorted(heads)}"
        )
    out: Predictions = {}
    for name, labels in heads.items():
        pred = data[name]
        if not isinstance(pred, dict) or set(pred) != {"label", "score"}:
            raise ValueError(f"{source} head {name!r} must be {{label, score}}")
        label, score = pred["label"], pred["score"]
        if label not in labels:
            raise ValueError(
                f"{source} head {name!r} label {label!r} is not one of {labels}"
            )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{source} head {name!r} score must be a number")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError(f"{source} head {name!r} score {score} is not in [0, 1]")
        out[name] = {"label": label, "score": float(score)}
    return out


def _build_head_module(hidden_size: int, heads: Mapping[str, List[str]]) -> Any:
    import torch

    return torch.nn.ModuleDict(
        {
            name: torch.nn.Linear(hidden_size, len(labels))
            for name, labels in heads.items()
        }
    )


def mean_pool(hidden: Any, attention_mask: Any) -> Any:
    """Mean of the token states the attention mask keeps."""
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


class QueryClassifier:
    """An encoder with one linear head per decision, served on one device."""

    def __init__(
        self,
        encoder: Any,
        tokenizer: Any,
        heads: Mapping[str, List[str]],
        *,
        head_module: Any = None,
        max_length: int = 128,
        base_model: str = "",
        device: str = "cpu",
    ) -> None:
        self.heads = validate_heads(dict(heads))
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.head_module = head_module or _build_head_module(
            encoder.config.hidden_size, self.heads
        )
        self.max_length = max_length
        self.base_model = base_model
        self.device = device
        self.encoder.to(device)
        self.head_module.to(device)

    def logits(self, texts: List[str]) -> Dict[str, Any]:
        """Per-head logits for a batch, with gradients (the training path)."""
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        hidden = self.encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state
        pooled = mean_pool(hidden, batch["attention_mask"])
        return {name: head(pooled) for name, head in self.head_module.items()}

    def classify(self, text: str) -> Predictions:
        """The top label and its probability for each head."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        import torch

        self.encoder.eval()
        self.head_module.eval()
        with torch.no_grad():
            logits = self.logits([text])
        out: Predictions = {}
        for name, labels in self.heads.items():
            probs = torch.softmax(logits[name][0], dim=-1)
            index = int(torch.argmax(probs))
            out[name] = {"label": labels[index], "score": float(probs[index])}
        return out

    def save(self, path: str | Path) -> Path:
        """Write the checkpoint directory ``load`` reads back."""
        from safetensors.torch import save_file

        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self.encoder.save_pretrained(root / ENCODER_DIR)
        self.tokenizer.save_pretrained(root / ENCODER_DIR)
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.head_module.state_dict().items()
        }
        save_file(state, str(root / HEAD_WEIGHTS_FILE))
        (root / HEADS_FILE).write_text(
            json.dumps(
                {
                    "heads": self.heads,
                    "max_length": self.max_length,
                    "base_model": self.base_model,
                },
                indent=2,
            )
        )
        return root

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "QueryClassifier":
        """Read a checkpoint directory written by :meth:`save`."""
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        root = Path(path)
        spec_file = root / HEADS_FILE
        if not spec_file.is_file():
            raise FileNotFoundError(
                f"{root} is not a query classifier: no {HEADS_FILE}"
            )
        spec = json.loads(spec_file.read_text())
        heads = validate_heads(spec.get("heads"))
        encoder = AutoModel.from_pretrained(root / ENCODER_DIR)
        tokenizer = AutoTokenizer.from_pretrained(root / ENCODER_DIR)
        head_module = _build_head_module(encoder.config.hidden_size, heads)
        head_module.load_state_dict(load_file(str(root / HEAD_WEIGHTS_FILE)))
        return cls(
            encoder,
            tokenizer,
            heads,
            head_module=head_module,
            max_length=int(spec.get("max_length", 128)),
            base_model=str(spec.get("base_model", "")),
            device=device,
        )
