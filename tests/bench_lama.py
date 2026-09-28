"""LaMa inpainting speed: CPU vs CoreML (Apple GPU / Neural Engine)."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import onnxruntime as ort

from mt import config

x = np.random.rand(1, 3, 512, 512).astype(np.float32)
m = np.zeros((1, 1, 512, 512), np.float32)
m[:, :, 200:300, 150:350] = 1
for name, prov in [("CPU", ["CPUExecutionProvider"]),
                   ("CoreML (all units)", [("CoreMLExecutionProvider", {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}), "CPUExecutionProvider"]),
                   ("CoreML (GPU)", [("CoreMLExecutionProvider", {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "MLProgram"}), "CPUExecutionProvider"])]:
    try:
        t0 = time.time()
        s = ort.InferenceSession(str(config.LAMA_ONNX), providers=prov)
        load = time.time() - t0
        names = [i.name for i in s.get_inputs()]
        feed = {names[0]: x, names[1]: m} if "mask" not in names[0].lower() else {names[1]: x, names[0]: m}
        s.run(None, feed)  # first run (compile)
        t0 = time.time()
        for _ in range(3):
            s.run(None, feed)
        print(f"{name:20s} load {load:5.1f}s  per run {(time.time() - t0) / 3:5.2f}s")
    except Exception as e:
        print(f"{name:20s} failed: {str(e)[:150]}")
