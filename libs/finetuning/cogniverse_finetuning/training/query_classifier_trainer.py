"""Fine-tune a multi-head query classifier on labelled queries.

One encoder (mmBERT-small by default) with a linear head per decision,
trained on the sum of the per-head cross-entropies. An example may label only
some heads; the others take no loss from it. The result is saved as a
``cogniverse_foundation.query_classifier`` checkpoint directory.
"""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from cogniverse_finetuning.dataset.gateway_labels import QueryLabelExample
from cogniverse_foundation.query_classifier import QueryClassifier, validate_heads

DEFAULT_BASE_MODEL = "jhu-clsp/mmBERT-small"
_IGNORE = -100


@dataclass(frozen=True)
class QueryClassifierTrainingConfig:
    base_model: str = DEFAULT_BASE_MODEL
    epochs: int = 3
    learning_rate: float = 5e-5
    batch_size: int = 16
    max_length: int = 128
    seed: int = 0
    device: str = "cpu"


def _targets(
    examples: Sequence[QueryLabelExample], heads: Mapping[str, List[str]]
) -> Dict[str, List[int]]:
    targets: Dict[str, List[int]] = {head: [] for head in heads}
    for example in examples:
        unknown = set(example.labels) - set(heads)
        if unknown:
            raise ValueError(
                f"example {example.span_id} labels unknown heads {sorted(unknown)}"
            )
        for head, labels in heads.items():
            label = example.labels.get(head)
            if label is None:
                targets[head].append(_IGNORE)
            elif label in labels:
                targets[head].append(labels.index(label))
            else:
                raise ValueError(
                    f"example {example.span_id} labels {head}={label!r}, "
                    f"not one of {labels}"
                )
    for head, column in targets.items():
        if all(value == _IGNORE for value in column):
            raise ValueError(f"no example labels head {head!r}")
    return targets


def train_query_classifier(
    examples: Sequence[QueryLabelExample],
    heads: Mapping[str, List[str]],
    output_dir: str | Path,
    config: QueryClassifierTrainingConfig = QueryClassifierTrainingConfig(),
) -> QueryClassifier:
    """Train on ``examples`` and save the checkpoint to ``output_dir``."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    heads = validate_heads(dict(heads))
    if not examples:
        raise ValueError("no training examples")
    targets = _targets(examples, heads)

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    classifier = QueryClassifier(
        AutoModel.from_pretrained(config.base_model),
        AutoTokenizer.from_pretrained(config.base_model),
        heads,
        max_length=config.max_length,
        base_model=config.base_model,
        device=config.device,
    )
    params = list(classifier.encoder.parameters()) + list(
        classifier.head_module.parameters()
    )
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=_IGNORE)

    order = list(range(len(examples)))
    classifier.encoder.train()
    classifier.head_module.train()
    for _ in range(config.epochs):
        random.shuffle(order)
        for start in range(0, len(order), config.batch_size):
            batch = order[start : start + config.batch_size]
            logits = classifier.logits([examples[i].text for i in batch])
            losses = []
            for head in heads:
                target = torch.tensor(
                    [targets[head][i] for i in batch], device=config.device
                )
                if bool((target != _IGNORE).any()):
                    losses.append(loss_fn(logits[head], target))
            loss = torch.stack(losses).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    classifier.save(output_dir)
    return classifier


def head_accuracy(
    classifier: QueryClassifier, examples: Sequence[QueryLabelExample]
) -> Dict[str, float]:
    """Per head, the share of the examples labelling it that it gets right."""
    hits: Dict[str, int] = {head: 0 for head in classifier.heads}
    totals: Dict[str, int] = {head: 0 for head in classifier.heads}
    for example in examples:
        predictions = classifier.classify(example.text)
        for head, label in example.labels.items():
            totals[head] += 1
            hits[head] += predictions[head]["label"] == label
    return {head: hits[head] / totals[head] for head in totals if totals[head]}
