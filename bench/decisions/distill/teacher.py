"""The teacher on a GPU, for labelling: EdgeJev's own server with its
session moved to the CUDA execution provider.

EdgeJev picks CPU or CoreML and nothing else, and the int8 graph it ships
has no CUDA kernels for its quantized matmuls; the fp32 build of the same
checkpoint (``edgejev build --precision fp32``) runs on CUDA whole. A
labelling run of two hundred thousand questions is hours on the CPU server
and minutes here, and the answers are the same model's.

    python -m bench.decisions.distill.teacher --model data/distill/teacher/laya-fp32 --port 8732
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from http.server import ThreadingHTTPServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="an fp32 EdgeJev model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8732)
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    args = parser.parse_args(argv)

    import onnxruntime as ort
    from edgejev import Agent

    # The CUDA and cuDNN libraries come from the nvidia wheels torch pulled
    # in; onnxruntime loads them from site-packages when told to.
    ort.preload_dlls()
    from edgejev.serve import make_handler

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("this onnxruntime has no CUDA provider; install onnxruntime-gpu", file=sys.stderr)
        return 2
    agent = Agent(args.model, threads=2)
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    agent.sess = ort.InferenceSession(
        os.path.join(args.model, agent.cfg["onnx_file"]), options,
        providers=[("CUDAExecutionProvider", {"device_id": args.device}), "CPUExecutionProvider"],
    )
    agent.provider_note = f"CUDA:{args.device}"
    # One session, one request at a time on the GPU; the HTTP threads only
    # queue behind it.
    lock = threading.Lock()
    original = agent.system_one

    def serialized(state, questions):
        with lock:
            return original(state, questions)

    agent.system_one = serialized
    agent.predict = serialized
    server = ThreadingHTTPServer((args.host, args.port), make_handler(agent, None))
    print(f"teacher on http://{args.host}:{args.port}/v1/systemone ({agent.provider_note})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
