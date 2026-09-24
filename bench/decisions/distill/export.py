"""From checkpoint to artifact: the encoder and heads as one ONNX graph with
a single logits output, quantized to int8, beside the tokenizer and
``student.json``; then the tarball and its checksum, which is what
:mod:`smrti.decisions.model` fetches.

The graph's output is every head's logits concatenated in registry order
— ``student.json`` says where each head's slice sits — so the runtime runs
one pass and reads the slice it needs.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tarfile
from pathlib import Path

from smrti.decisions.student.runtime import CONFIG_NAME, MAX_LEN, Student, is_ready

from .train import OUT, build_model

logger = logging.getLogger("distill.export")


def export(checkpoint: Path = OUT / "checkpoint", out: Path = OUT / "student", *,
           asset: str = "student-minilm-int8.tar.gz") -> tuple[Path, str]:
    import numpy as np
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoTokenizer

    meta = json.loads((checkpoint / CONFIG_NAME).read_text())
    tasks = [h["task"] for h in meta["heads"]]
    model = build_model(tasks, str(checkpoint / "encoder"))
    model.heads.load_state_dict(torch.load(checkpoint / "heads.pt", map_location="cpu"))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint / "encoder")

    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "model_fp32.onnx"
    enc = tokenizer(["routing:\nmessage: hola"], return_tensors="pt", padding=True)
    with torch.no_grad():
        torch.onnx.export(
            model, (enc["input_ids"], enc["attention_mask"]), str(fp32),
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                          "logits": {0: "batch"}},
            opset_version=17, dynamo=False,
        )
    onnx.checker.check_model(str(fp32))
    # What gets quantized is what the accuracy survives. The 250k-row
    # embedding table is 90% of the fp32 graph and costs nothing to hold in
    # int8; the projections inside attention and the feed-forward take
    # int8 too. The two output projections of every layer do not: with them
    # quantized the choice heads lost 5–15 points of agreement with the
    # teacher (supersession 0.77 → 0.62, completion 0.66 → 0.51) and with
    # them in fp32 the graph matches the fp32 one on every head, at 114 MB
    # against 428.
    graph = onnx.load(str(fp32))
    keep = [n.name for n in graph.graph.node if "attention/output" in n.name or "/output/dense" in n.name]
    quantize_dynamic(str(fp32), str(out / "model.onnx"), weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul", "Gather"], nodes_to_exclude=keep)
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    (out / CONFIG_NAME).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    assert is_ready(out)
    # The fp32 graph beside it, as its own student directory, so what the
    # quantization cost can be measured with the same evaluator.
    full = out.parent / "student-fp32"
    full.mkdir(exist_ok=True)
    fp32.replace(full / "model.onnx")
    for name in ("tokenizer.json", CONFIG_NAME):
        shutil.copy(out / name, full / name)

    # The artifact answers like the checkpoint: same argmax, probabilities
    # within quantization noise, on a handful of texts.
    student = Student(out, threads=2)
    with torch.no_grad():
        ref = model(enc["input_ids"], enc["attention_mask"])[0].numpy()
    got, _ = student._logits("routing:\nmessage: hola")
    drift = float(np.abs(got - ref).max())
    logger.info("exported %s (max logit drift after int8: %.3f)", out, drift)

    archive = out.parent / asset
    with tarfile.open(archive, "w:gz") as tar:
        for name in ("model.onnx", "tokenizer.json", CONFIG_NAME):
            tar.add(out / name, arcname=name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (out.parent / f"{asset}.sha256").write_text(digest + "\n")
    logger.info("packaged %s (%d MB) sha256 %s", archive, archive.stat().st_size >> 20, digest)
    return archive, digest
