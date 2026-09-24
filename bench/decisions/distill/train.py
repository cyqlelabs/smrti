"""Fitting the student: a shared 6-layer multilingual encoder and one linear
head per registry task, trained to reproduce the teacher's probabilities.

Runs in the CUDA environment (``make distill-env``). Soft targets
throughout — binary cross-entropy against the teacher's noul probabilities,
KL divergence against its choice distributions — so the student inherits
the teacher's calibration and not only its argmax. Held-out is a stable
tenth of every task's states, split by id, and the checkpoint kept is the
one with the best mean held-out agreement.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from smrti.decisions.student import registry
from smrti.decisions.student.runtime import CONFIG_NAME, FORMAT, MAX_LEN

from .corpus import STATES
from .label import LABELS, read_rows
from .sources import DATA

logger = logging.getLogger("distill.train")

ENCODER = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
OUT = Path(DATA) / "distill" / "out"
HELD_OUT_SHARE = 10  # one in ten, by id


def is_held_out(state_id: str) -> bool:
    return int(hashlib.sha256(state_id.encode()).hexdigest()[:8], 16) % HELD_OUT_SHARE == 0


def load_examples(tasks: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(train, held-out) examples: text, task, target vector, lang."""
    train: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    for task in tasks:
        spec = registry.tasks()[task]
        states = {r["id"]: r for r in read_rows(STATES / f"{task}.jsonl")}
        n = 0
        for row in read_rows(LABELS / f"{task}.jsonl"):
            state = states.get(row["id"])
            if state is None:
                continue
            target = [float(row["targets"].get(k, 0.0)) for k in spec.keys]
            ex = {"id": row["id"], "task": task, "lang": row.get("lang") or "en",
                  "text": registry.encoder_text(task, state["state"]), "target": target}
            (held if is_held_out(row["id"]) else train).append(ex)
            n += 1
        logger.info("%s: %d labelled examples", task, n)
    return train, held


def head_layout(tasks: list[str]) -> list[dict[str, Any]]:
    layout, offset = [], 0
    for task in tasks:
        spec = registry.tasks()[task]
        layout.append({"task": task, "kind": spec.kind, "keys": list(spec.keys), "offset": offset,
                       "width": spec.width, "temperature": 1.0})
        offset += spec.width
    return layout


def build_model(tasks: list[str], encoder: str = ENCODER):
    import torch
    from torch import nn
    from transformers import AutoModel

    class StudentModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder)
            hidden = self.encoder.config.hidden_size
            self.dropout = nn.Dropout(0.1)
            self.heads = nn.ModuleDict({t: nn.Linear(hidden, registry.tasks()[t].width) for t in tasks})
            self.tasks = list(tasks)

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            h = self.dropout(h)
            return torch.cat([self.heads[t](h) for t in self.tasks], dim=-1)

    return StudentModel()


def _loss(kind: str, logits, target):
    import torch.nn.functional as F

    if kind == registry.NOULS:
        return F.binary_cross_entropy_with_logits(logits, target)
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def _agreement(kind: str, logits, target) -> tuple[int, int]:
    """(agreeing, total): a noul agrees when both sides fall on the same side
    of 0.5; a choice when the argmax matches."""
    import torch

    if kind == registry.NOULS:
        agree = ((logits > 0) == (target > 0.5)).sum().item()
        return int(agree), int(target.numel())
    agree = (logits.argmax(-1) == target.argmax(-1)).sum().item()
    return int(agree), int(target.shape[0])


def evaluate(model, tokenizer, held: list[dict[str, Any]], tasks: list[str], layout: dict[str, dict[str, Any]],
             device, batch_size: int = 64) -> dict[str, dict[str, Any]]:
    import torch

    model.eval()
    report: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for task in tasks:
            spec = registry.tasks()[task]
            rows = [e for e in held if e["task"] == task]
            agree = total = 0
            loss_sum = 0.0
            by_lang: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                enc = tokenizer([e["text"] for e in batch], truncation=True, max_length=MAX_LEN,
                                padding=True, return_tensors="pt").to(device)
                target = torch.tensor([e["target"] for e in batch], device=device)
                out = model(enc["input_ids"], enc["attention_mask"])
                lo = layout[task]["offset"]
                logits = out[:, lo : lo + spec.width]
                loss_sum += _loss(spec.kind, logits, target).item() * len(batch)
                a, t = _agreement(spec.kind, logits, target)
                agree += a
                total += t
                for e, l_, tg in zip(batch, logits, target):
                    a1, t1 = _agreement(spec.kind, l_.unsqueeze(0), tg.unsqueeze(0))
                    by_lang[e["lang"]][0] += a1
                    by_lang[e["lang"]][1] += t1
            report[task] = {
                "n": len(rows), "agreement": (agree / total) if total else None,
                "loss": (loss_sum / len(rows)) if rows else None,
                "by_lang": {l: round(a / t, 3) for l, (a, t) in sorted(by_lang.items()) if t},
            }
    model.train()
    return report


def fit_temperatures(model, tokenizer, held, tasks, layout, device) -> dict[str, float]:
    """One temperature per head, the value that makes the student's
    probabilities on held-out data closest to the teacher's."""
    import torch

    model.eval()
    temps: dict[str, float] = {}
    with torch.no_grad():
        for task in tasks:
            spec = registry.tasks()[task]
            rows = [e for e in held if e["task"] == task]
            if not rows:
                temps[task] = 1.0
                continue
            logits_all, targets_all = [], []
            for i in range(0, len(rows), 64):
                batch = rows[i : i + 64]
                enc = tokenizer([e["text"] for e in batch], truncation=True, max_length=MAX_LEN,
                                padding=True, return_tensors="pt").to(device)
                lo = layout[task]["offset"]
                logits_all.append(model(enc["input_ids"], enc["attention_mask"])[:, lo : lo + spec.width].float().cpu())
                targets_all.append(torch.tensor([e["target"] for e in batch]))
            logits, target = torch.cat(logits_all), torch.cat(targets_all)
            best, best_t = None, 1.0
            for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0, 2.5]:
                value = _loss(spec.kind, logits / t, target).item()
                if best is None or value < best:
                    best, best_t = value, t
            temps[task] = best_t
    model.train()
    return temps


def train(tasks: list[str], *, epochs: int = 3, batch_size: int = 32, lr: float = 4e-5, head_lr: float = 1e-3,
          encoder: str = ENCODER, out: Path = OUT, seed: int = 0, device: str | None = None) -> Path:
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_rows, held = load_examples(tasks)
    if not train_rows:
        raise SystemExit("no labelled examples; run `label` first")
    layout_list = head_layout(tasks)
    layout = {h["task"]: h for h in layout_list}
    tokenizer = AutoTokenizer.from_pretrained(encoder)
    model = build_model(tasks, encoder).to(device)

    # Batches are single-task, drawn in proportion to each task's share of
    # the corpus, so every head sees its whole data each epoch.
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in train_rows:
        by_task[e["task"]].append(e)
    steps_per_epoch = sum(math.ceil(len(v) / batch_size) for v in by_task.values())
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": lr},
        {"params": model.heads.parameters(), "lr": head_lr},
    ], weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps_per_epoch * epochs),
                                                steps_per_epoch * epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best_score, best_report = -1.0, None
    checkpoint = out / "checkpoint"
    logger.info("training %d examples over %d tasks on %s, %d steps/epoch",
                len(train_rows), len(tasks), device, steps_per_epoch)
    for epoch in range(epochs):
        batches: list[tuple[str, list[dict[str, Any]]]] = []
        for task, rows in by_task.items():
            random.shuffle(rows)
            batches += [(task, rows[i : i + batch_size]) for i in range(0, len(rows), batch_size)]
        random.shuffle(batches)
        started, running = time.monotonic(), 0.0
        for step, (task, batch) in enumerate(batches, 1):
            spec = registry.tasks()[task]
            enc = tokenizer([e["text"] for e in batch], truncation=True, max_length=MAX_LEN,
                            padding=True, return_tensors="pt").to(device)
            target = torch.tensor([e["target"] for e in batch], device=device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                out_logits = model(enc["input_ids"], enc["attention_mask"])
            lo = layout[task]["offset"]
            loss = _loss(spec.kind, out_logits[:, lo : lo + spec.width].float(), target)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 200 == 0:
                logger.info("epoch %d step %d/%d loss %.4f (%.1f steps/s)", epoch + 1, step, len(batches),
                            running / 200, step / (time.monotonic() - started))
                running = 0.0
        report = evaluate(model, tokenizer, held, tasks, layout, device)
        score = sum(r["agreement"] or 0.0 for r in report.values()) / len(report)
        logger.info("epoch %d held-out: %s", epoch + 1,
                    {t: round(r["agreement"] or 0, 3) for t, r in report.items()})
        if score > best_score:
            best_score, best_report = score, report
            checkpoint.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(checkpoint / "encoder")
            tokenizer.save_pretrained(checkpoint / "encoder")
            torch.save(model.heads.state_dict(), checkpoint / "heads.pt")
            temps = fit_temperatures(model, tokenizer, held, tasks, layout, device)
            for h in layout_list:
                h["temperature"] = temps[h["task"]]
            (checkpoint / CONFIG_NAME).write_text(json.dumps({
                "format": FORMAT, "model_name": f"smrti-student-{time.strftime('%Y%m%d')}",
                "encoder": "model.onnx", "tokenizer": "tokenizer.json", "max_len": MAX_LEN,
                "source_encoder": encoder, "heads": layout_list,
                "held_out": {t: {k: v for k, v in r.items()} for t, r in report.items()},
            }, indent=2, ensure_ascii=False))
            logger.info("checkpoint saved (mean agreement %.3f)", score)
    (out / "train_report.json").write_text(json.dumps(best_report, indent=2))
    return checkpoint
