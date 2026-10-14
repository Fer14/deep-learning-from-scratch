import importlib.util
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits


def load_data():
    digits = load_digits()
    x = torch.tensor(digits.data, dtype=torch.float32)
    y = torch.tensor(digits.target)
    order = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    return x[order], y[order]


class DigitClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor(1 / 16.0))
        self.hidden = nn.Linear(64, 128)
        self.out = nn.Linear(128, 10)

    def forward(self, pixels):
        x = pixels * self.scale
        return torch.softmax(self.out(F.relu(self.hidden(x))), dim=-1)


def train(model, x, y, epochs=30):
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(epochs):
        for i in range(0, 1400, 64):
            probs = model(x[i:i + 64])
            loss = F.nll_loss(torch.log(probs + 1e-9), y[i:i + 64])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


def export_artifact(model, directory):
    spec = {
        "format_version": 1,
        "input": {"name": "pixels", "shape": [None, 64], "dtype": "float32", "range": [0, 16]},
        "layers": [
            {"op": "scale", "param": "scale"},
            {"op": "linear", "weight": "hidden.weight", "bias": "hidden.bias"},
            {"op": "relu"},
            {"op": "linear", "weight": "out.weight", "bias": "out.bias"},
            {"op": "softmax"},
        ],
        "labels": [str(d) for d in range(10)],
    }
    arrays = {name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    np.savez(directory / "weights.npz", **arrays)
    (directory / "model.json").write_text(json.dumps(spec, indent=2))
    return directory


class NumpyRuntime:
    def __init__(self, directory):
        self.spec = json.loads((directory / "model.json").read_text())
        self.params = dict(np.load(directory / "weights.npz"))
        self.labels = self.spec["labels"]

    def run(self, x):
        x = np.asarray(x, dtype=np.float32)
        for layer in self.spec["layers"]:
            if layer["op"] == "scale":
                x = x * self.params[layer["param"]]
            elif layer["op"] == "linear":
                x = x @ self.params[layer["weight"]].T + self.params[layer["bias"]]
            elif layer["op"] == "relu":
                x = np.maximum(x, 0)
            elif layer["op"] == "softmax":
                e = np.exp(x - x.max(axis=-1, keepdims=True))
                x = e / e.sum(axis=-1, keepdims=True)
        return x


def make_handler(runtime):
    class PredictHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, payload):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"status": "ok", "format_version": runtime.spec["format_version"]})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                return self.reply(404, {"error": "not found"})
            try:
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                pixels = np.asarray(request["pixels"], dtype=np.float32)
                if pixels.ndim == 1:
                    pixels = pixels[None]
                if pixels.shape[1:] != (64,):
                    raise ValueError(f"expected 64 pixels per image, got shape {list(pixels.shape)}")
            except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
                return self.reply(400, {"error": str(error)})
            probs = runtime.run(pixels)
            self.reply(200, {"predictions": [
                {"label": runtime.labels[int(p.argmax())], "confidence": round(float(p.max()), 4)} for p in probs]})

    return PredictHandler


def post_json(url, payload):
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    torch.manual_seed(0)
    x, y = load_data()
    model = train(DigitClassifier(), x, y)
    x_test, y_test = x[1400:], y[1400:]
    with torch.no_grad():
        reference = model(x_test)
    print(f"Trained DigitClassifier (raw 0-16 pixels in, probabilities out): test accuracy "
          f"{(reference.argmax(1) == y_test).float().mean():.4f}\n")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        print("1) Export from scratch: an architecture spec (JSON) + weights (npz)")
        artifact = export_artifact(model, tmp)
        files = {p.name: p.stat().st_size for p in sorted(artifact.iterdir())}
        print(f"  files: {files}")
        runtime = NumpyRuntime(artifact)
        numpy_out = runtime.run(x_test.numpy())
        print(f"  NumPy runtime (no PyTorch needed) max |diff| vs PyTorch: {np.abs(numpy_out - reference.numpy()).max():.2e}")

        print("\n2) Serve it over HTTP from scratch")
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(runtime))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        with urllib.request.urlopen(base + "/health", timeout=5) as response:
            print(f"  GET  /health  -> {response.status} {json.loads(response.read())}")
        status, body = post_json(base + "/predict", {"pixels": x_test[:3].tolist()})
        print(f"  POST /predict (3 images) -> {status} {body}  true labels {y_test[:3].tolist()}")
        status, body = post_json(base + "/predict", {"pixels": [1.0, 2.0, 3.0]})
        print(f"  POST /predict (bad input) -> {status} {body}")
        latencies = []
        for i in range(100):
            start = time.perf_counter()
            post_json(base + "/predict", {"pixels": x_test[i].tolist()})
            latencies.append(time.perf_counter() - start)
        latencies = np.array(latencies) * 1e3
        print(f"  100 sequential requests: p50 {np.percentile(latencies, 50):.2f} ms, p99 {np.percentile(latencies, 99):.2f} ms")
        server.shutdown()
        server.server_close()

        print("\n3) Use It: torch.export, a full graph with a dynamic batch dimension")
        batch = torch.export.Dim("batch", min=1, max=1024)
        exported = torch.export.export(model, (x_test[:2],), dynamic_shapes={"pixels": {0: batch}})
        ops = [node.target.__name__ for node in exported.graph.nodes if node.op == "call_function"]
        print(f"  graph ops: {ops}")
        torch.export.save(exported, tmp / "model.pt2")
        loaded = torch.export.load(tmp / "model.pt2").module()
        with torch.no_grad():
            out = loaded(x_test)
        print(f"  saved {(tmp / 'model.pt2').stat().st_size} bytes, reloaded, batch {len(x_test)}: max |diff| {(out - reference).abs().max():.2e}")

        print("\n4) Use It: TorchScript (legacy, deprecated in favor of torch.export)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scripted = torch.jit.script(model)
            scripted.save(str(tmp / "model_script.pt"))
            reloaded = torch.jit.load(str(tmp / "model_script.pt"))
            with torch.no_grad():
                out = reloaded(x_test)
        print(f"  saved {(tmp / 'model_script.pt').stat().st_size} bytes, reloaded: max |diff| {(out - reference).abs().max():.2e}")

        print("\n5) ONNX")
        if importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxscript"):
            onnx_program = torch.onnx.export(model, (x_test[:2],), dynamo=True)
            onnx_program.save(str(tmp / "model.onnx"))
            print(f"  exported model.onnx ({(tmp / 'model.onnx').stat().st_size} bytes)")
        else:
            print("  the onnx/onnxscript packages are not installed; skipping (torch.onnx.export(model, args, dynamo=True))")


if __name__ == "__main__":
    main()
