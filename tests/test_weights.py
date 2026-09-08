"""externalized() converts an ONNX model once and serves the mapped copy after."""
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from smrti.core.weights import externalized, release_heap


@pytest.fixture
def inline_model(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    weights = np.arange(512, dtype=np.float32).reshape(1, 512)
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        "add",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 512])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 512])],
        initializer=[numpy_helper.from_array(weights, "w")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    # make_model stamps the installed onnx library's newest IR version, which a
    # runtime a release or two older refuses; the test is about the conversion.
    model.ir_version = 9
    src_dir = tmp_path / "model"
    src_dir.mkdir()
    onnx.save(model, src_dir / "model.onnx")
    (src_dir / "tokenizer.json").write_text("{}")
    return src_dir / "model.onnx"


def test_externalized_moves_tensors_to_a_sidecar_and_runs(inline_model):
    mapped = externalized(inline_model)

    assert mapped != inline_model
    assert (mapped.parent / "model.onnx.data").exists()
    session = ort.InferenceSession(str(mapped), providers=["CPUExecutionProvider"])
    (out,) = session.run(None, {"x": np.ones((1, 512), dtype=np.float32)})
    assert out[0, 511] == pytest.approx(512.0)


def test_externalized_converts_once(inline_model):
    first = externalized(inline_model)
    written = first.stat().st_mtime_ns

    assert externalized(inline_model) == first
    assert first.stat().st_mtime_ns == written


def test_externalized_returns_a_mapped_source_as_is(inline_model):
    mapped = externalized(inline_model)

    assert externalized(mapped) == mapped


def test_externalized_copies_siblings_for_directory_loaders(inline_model):
    mapped = externalized(inline_model, with_siblings=True)

    assert (mapped.parent / "tokenizer.json").read_text() == "{}"


def test_release_heap_is_safe_to_call():
    release_heap()
