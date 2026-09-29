"""Export the full XFeat sparse detector (CNN + NMS + top-k + descriptor sampling) to ONNX with static
output shapes, and check it against XFeat.detectAndCompute.

Graph:   image (B,1,H,W), threshold (1,)  ->  keypoints (B,K,2), scores (B,K), descriptors (B,K,64)
         with K = --k-max fixed at export time. Padding rows have score -1.
Outside: keep = scores > 0, slice to the configured k (<= K, TopK output is sorted), rescale the
         keypoints from network-input pixels to original-frame pixels.

Run from the repo root:
    .venv/bin/python onnx_conversion/export_xfeat_onnx.py --video <video0.mp4>
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from modules.model import XFeatModel  # noqa: E402
from modules.xfeat import XFeat  # noqa: E402  (reference for the checks only)

INPUT_NAMES = ["image", "threshold"]
OUTPUT_NAMES = ["keypoints", "scores", "descriptors"]


class XFeatModelONNX(XFeatModel):
    """XFeatModel with `_unfold2d` rewritten as reshape/permute.

    The original uses Tensor.unfold, which the TorchScript exporter can't export with dynamic H/W
    ("ONNX export of operator Unfold, input size not accessible"). This version produces the identical
    tensor (asserted in load_net) using only ops that export cleanly.
    """

    def _unfold2d(self, x, ws=2):
        B, C, H, W = x.shape
        x = x.reshape(B, C, H // ws, ws, W // ws, ws)
        return x.permute(0, 1, 3, 5, 2, 4).reshape(B, C * ws * ws, H // ws, W // ws)


class XFeatONNX(nn.Module):
    """forward() is exactly the graph that ends up in the ONNX file.

    Inputs
      - image: (B, 1, H, W) float32 gray in [0, 1], H and W multiples of 32. Resizing and gray
        conversion happen outside, so no preprocessing of the input tensor here.
      - threshold: (1,) float32 tensor. A tensor input stays tunable at run time; a Python float
        (or a default argument) would be frozen into the graph as a constant during tracing.

    Export-time constants (recorded in the ONNX metadata)
      - k_max: number of rows in every output. Requirement: H*W >= k_max.
      - nms_kernel: max-pool window (XFeat uses 5).
    """

    def __init__(self, net: XFeatModel, k_max: int, nms_kernel: int = 5):
        super().__init__()
        self.net = net
        self.k_max = k_max
        self.nms_kernel = nms_kernel

    @staticmethod
    def _normgrid(pos: torch.Tensor, H, W) -> torch.Tensor:
        """Pixel coords (..., 2) as (x, y) -> grid_sample coords in [-1, 1].

        Same formula as InterpolateSparse2d.normgrid (interpolator.py:19): it divides by (W-1, H-1)
        although grid_sample is then called with align_corners=False. That mismatch is a quirk of
        XFeat, but copying it is what makes the results identical.
        """
        x = 2.0 * (pos[..., 0] / (W - 1)) - 1.0
        y = 2.0 * (pos[..., 1] / (H - 1)) - 1.0
        return torch.stack([x, y], dim=-1)

    def forward(self, image: torch.Tensor, threshold: torch.Tensor):
        B, _, H, W = image.shape

        # 1. Network: feats (B,64,H/8,W/8), kpt_logits (B,65,H/8,W/8), reliability (B,1,H/8,W/8).
        feats, kpt_logits, reliability = self.net(image)

        # 2. Dense L2 normalisation of feats over the channel dim (xfeat.py:70).
        feats = F.normalize(feats, dim=1)

        # 3. Keypoint probability at full res (get_kpts_heatmap, xfeat.py:243-247):
        #    softmax over 65 channels -> drop dustbin -> 8x8 depth-to-space -> (B,1,H,W).
        prob = F.softmax(kpt_logits, dim=1)[:, :64]
        Hc, Wc = prob.shape[-2:]
        prob = prob.permute(0, 2, 3, 1).reshape(B, Hc, Wc, 8, 8)
        prob = prob.permute(0, 1, 3, 2, 4).reshape(B, 1, Hc * 8, Wc * 8)

        # 4. NMS without nonzero() (xfeat.py:249-252): a dense boolean peak map.
        local_max = F.max_pool2d(prob, self.nms_kernel, stride=1, padding=self.nms_kernel // 2)
        peak = (prob == local_max) & (prob > threshold)

        # 5. Scores at every pixel, sampled exactly like xfeat.py:79 samples them at the peaks:
        #    keypoint prob with 'nearest', reliability (1/8 res) with 'bilinear', both through
        #    the normgrid quirk. For the prob this is the pixel's own value except on the last
        #    row/column, where the quirk samples outside the image (0) and XFeat drops the point.
        xs = torch.arange(W, dtype=torch.float32)
        ys = torch.arange(H, dtype=torch.float32)
        pix = torch.stack([xs[None, :].expand(H, W), ys[:, None].expand(H, W)], dim=-1)  # (H, W, 2)
        grid = self._normgrid(pix, H, W)[None].expand(B, H, W, 2)
        prob_s = F.grid_sample(prob, grid, mode="nearest", align_corners=False)
        rel_s = F.grid_sample(reliability, grid, mode="bilinear", align_corners=False)

        # 6. Score map: prob * reliability where peak, -1 elsewhere (-1 = XFeat's padding marker, xfeat.py:80).
        score_map = torch.where(peak, prob_s * rel_s, torch.full_like(prob, -1.0))

        # 7. Top-k instead of nonzero + argsort (xfeat.py:83-87). Output is sorted, descending.
        scores, flat_idx = torch.topk(score_map.reshape(B, H * W), k=self.k_max, dim=1)
        kx = flat_idx % W
        ky = torch.div(flat_idx, W, rounding_mode="floor")
        keypoints = torch.stack([kx, ky], dim=-1).to(torch.float32)  # (B, K, 2), (x, y)

        # 8. Descriptors (xfeat.py:90-94): bicubic sampling of the 1/8-res feats at the keypoints,
        #    then L2-normalise again (mixing unit vectors makes them shorter).
        kgrid = self._normgrid(keypoints, H, W)[:, :, None, :]  # (B, K, 1, 2)
        desc = F.grid_sample(feats, kgrid, mode="bicubic", align_corners=False)  # (B, 64, K, 1)
        descriptors = F.normalize(desc[..., 0].permute(0, 2, 1), dim=-1)  # (B, K, 64)

        return keypoints, scores, descriptors


def load_net(weights: Path) -> XFeatModelONNX:
    """CPU weights, eval mode (BatchNorm must use the trained statistics). xfeat.pt also contains
    fine_matcher weights; they load fine and don't end up in the graph because forward() never uses them."""
    net = XFeatModelONNX()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    net.eval()
    x = torch.rand(2, 1, 64, 96)
    assert torch.equal(XFeatModel._unfold2d(net, x, 8), net._unfold2d(x, 8)), "unfold rewrite differs"
    return net


def load_test_images(hw: tuple[int, int], video: Path | None, frame: int) -> dict[str, torch.Tensor]:
    """(1, 1, H, W) float32 test inputs in [0, 1]: noise (thousands of peaks), flat (no peaks), and
    optionally a real video frame (luma, INTER_AREA resize, as in benchmark_xfeat_vs_orb.py)."""
    h, w = hw
    images = {
        "noise": torch.rand(1, 1, h, w, generator=torch.Generator().manual_seed(0)),
        "flat": torch.full((1, 1, h, w), 0.5),
    }
    if video is not None:
        import cv2

        cap = cv2.VideoCapture(str(video))
        for _ in range(frame + 1):
            ok, bgr = cap.read()
            assert ok, f"could not read frame {frame} of {video}"
        gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_AREA)
        images[f"frame{frame}"] = torch.from_numpy(gray).float().div(255.0)[None, None]
    return images


def compare(name: str, keypoints, scores, descriptors, ref: dict) -> None:
    """Compare one image's (K,2)/(K,)/(K,64) outputs with XFeat's result dict as sets of keypoints."""
    keypoints, scores, descriptors = (torch.as_tensor(t) for t in (keypoints, scores, descriptors))
    keep = scores > 0
    kpts, scores, descriptors = keypoints[keep], scores[keep], descriptors[keep]

    # Keypoints sit on integer pixels in both versions, so exact (x, y) tuples work as keys.
    ref_index = {tuple(p): i for i, p in enumerate(ref["keypoints"].round().long().tolist())}
    ours = [tuple(p) for p in kpts.round().long().tolist()]
    shared = [(i, ref_index[p]) for i, p in enumerate(ours) if p in ref_index]

    line = (f"  {name:10s} ours={len(ours):5d} ref={len(ref_index):5d} shared={len(shared):5d} "
            f"only_ours={len(ours) - len(shared)} only_ref={len(ref_index) - len(shared)}")
    if shared:
        oi, ri = map(list, zip(*shared))
        line += (f"  max|d score|={(scores[oi] - ref['scores'][ri]).abs().max():.2e}"
                 f"  max|d desc|={(descriptors[oi] - ref['descriptors'][ri]).abs().max():.2e}")
    print(line)


def export(model: XFeatONNX, out: Path, opset: int, dummy_hw: tuple[int, int]) -> None:
    image = torch.rand(1, 1, *dummy_hw)
    threshold = torch.tensor([0.05], dtype=torch.float32)
    torch.onnx.export(
        model,
        (image, threshold),
        str(out),
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "keypoints": {0: "batch"},
            "scores": {0: "batch"},
            "descriptors": {0: "batch"},
        },
        opset_version=opset,
    )

    proto = onnx.load(str(out))
    commit = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"k_max": model.k_max, "nms_kernel": model.nms_kernel, "opset": opset, "torch": torch.__version__, "xfeat_commit": commit}
    onnx.helper.set_model_props(proto, {k: str(v) for k, v in meta.items()})
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) {meta}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, default=REPO_ROOT / "weights" / "xfeat.pt")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "onnx_conversion" / "xfeat.onnx")
    ap.add_argument("--opset", type=int, default=17, help="GridSample needs >= 16.")
    ap.add_argument("--k-max", type=int, default=4096, help="Rows per output; slice to k at run time.")
    ap.add_argument("--nms-kernel", type=int, default=5)
    ap.add_argument("--dummy-hw", type=int, nargs=2, default=(640, 576), metavar=("H", "W"))
    ap.add_argument("--check-hw", type=int, nargs=2, default=(512, 608), metavar=("H", "W"),
                    help="Second resolution for the checks, to prove the dynamic axes work.")
    ap.add_argument("--threshold", type=float, default=0.05, help="Run-time value used by the checks.")
    ap.add_argument("--video", type=Path, default=None, help="Optional video for a real-frame check.")
    ap.add_argument("--frame", type=int, default=18)
    args = ap.parse_args()

    net = load_net(args.weights)
    model = XFeatONNX(net, args.k_max, args.nms_kernel).eval()
    reference = XFeat(weights=str(args.weights), top_k=args.k_max, detection_threshold=args.threshold)
    threshold = torch.tensor([args.threshold], dtype=torch.float32)
    tests = {hw: load_test_images(hw, args.video, args.frame) for hw in (tuple(args.dummy_hw), tuple(args.check_hw))}
    refs = {(hw, name): reference.detectAndCompute(img, top_k=args.k_max, detection_threshold=args.threshold)[0]
            for hw, images in tests.items() for name, img in images.items()}

    print("1) PyTorch wrapper vs XFeat.detectAndCompute")
    for hw, images in tests.items():
        print(f" {hw[0]}x{hw[1]}")
        for name, img in images.items():
            with torch.no_grad():
                k, s, d = model(img, threshold)
            compare(name, k[0], s[0], d[0], refs[hw, name])

    print("2) export")
    export(model, args.out, args.opset, tuple(args.dummy_hw))

    print("3) ONNX Runtime vs XFeat.detectAndCompute")
    session = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    for hw, images in tests.items():
        print(f" {hw[0]}x{hw[1]}")
        for name, img in images.items():
            k, s, d = session.run(None, {"image": img.numpy(), "threshold": threshold.numpy()})
            compare(name, k[0], s[0], d[0], refs[hw, name])


if __name__ == "__main__":
    main()
