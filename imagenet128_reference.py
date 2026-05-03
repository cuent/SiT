from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np

root = Path("/workspace/datasets/blurred_imagenet_128/train")
out = Path("references/blurred_imagenet_128_train_50k_seed0.npz")
out.parent.mkdir(exist_ok=True)

paths = sorted(
    p for p in root.rglob("*")
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
)

rng = np.random.default_rng(0)
idx = rng.choice(len(paths), size=50_000, replace=False)
paths = [paths[i] for i in idx]

images = []
for p in tqdm(paths, desc="Building real reference npz"):
    im = Image.open(p).convert("RGB")
    if im.size != (128, 128):
        im = im.resize((128, 128), Image.BICUBIC)
    images.append(np.asarray(im, dtype=np.uint8))

arr = np.stack(images, axis=0)
np.savez(out, arr_0=arr)
print(f"saved {out} {arr.shape} {arr.dtype}")