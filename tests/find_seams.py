import numpy as np
from pathlib import Path
from PIL import Image
from mt.detect import TextDetector
d = TextDetector()
fs = sorted(Path("out/real/ep1").glob("*.jpg"))
for a, b in zip(fs, fs[1:]):
    A, B = (np.asarray(Image.open(f).convert("RGB")) for f in (a, b))
    if A.shape[1] != B.shape[1]:
        continue
    H = A.shape[0]
    bl, _ = d.detect(np.vstack([A, B]))
    for x in bl:
        if x.y1 < H - 2 < H + 2 < x.y2:
            print(a.stem, b.stem, (x.x1, x.y1 - H, x.x2, x.y2 - H), flush=True)
print("done")
