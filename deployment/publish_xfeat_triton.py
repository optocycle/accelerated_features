"""Package an exported XFeat ONNX file as a Triton model and log it to MLflow (triton flavor), the same way
optocycle/RAFT-Stereo does it (export_to_onnx.py + raft.pbtxt + triton_flavor.py).

Logged artifact (run artifact path `models`):
    model/config.pbtxt     from xfeat.pbtxt, K taken from the ONNX metadata (k_max)
    model/1/model.onnx

Deploy it to Triton with the MLflow Triton plugin (oc_ml/inference-server), after registering the run's model:
    mlflow deployments create -t triton --flavor triton --name xfeat_tile_k128 -m models:/<registered name>/<version>

MLflow credentials come from the environment or a .env file in the repo root (see .env.example).

Run from the repo root, for a file that export_xfeat_onnx.py already wrote:
    poetry run python deployment/publish_xfeat_triton.py weights/xfeat_tiled_k128.onnx --triton-hw 192 256
or export and publish in one go with `export_xfeat_onnx.py --publish`.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import onnx

HERE = Path(__file__).resolve().parent
CONFIG_TEMPLATE = HERE / "xfeat.pbtxt"


def triton_config(k_max: int, hw: tuple[int, int]) -> str:
    """config.pbtxt for a model with k_max output rows. hw = (-1, -1) keeps height and width dynamic."""
    h, w = hw
    assert all(d == -1 or (d > 0 and d % 32 == 0) for d in hw), f"--triton-hw must be -1 or multiples of 32, got {hw}"
    assert h == -1 or w == -1 or h * w >= k_max, f"TopK needs H*W >= k_max ({h}x{w} < {k_max})"
    config = CONFIG_TEMPLATE.read_text()
    return config.replace("<H>", str(h)).replace("<W>", str(w)).replace("<K>", str(k_max))


def build_triton_model(onnx_path: Path, out_dir: Path, hw: tuple[int, int]) -> Path:
    """Write a Triton model directory <out_dir>/model (config.pbtxt + 1/model.onnx) and return its path."""
    meta = {p.key: p.value for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props}
    config = triton_config(int(meta["k_max"]), hw)
    model_dir = out_dir / "model"
    (model_dir / "1").mkdir(parents=True)
    shutil.copyfile(onnx_path, model_dir / "1" / "model.onnx")
    (model_dir / "config.pbtxt").write_text(config)
    return model_dir


def publish(onnx_path: Path, hw: tuple[int, int], experiment: str, registered_model_name: str | None,
            params: dict | None = None) -> None:
    """Log onnx_path as a Triton model to a new run of `experiment`, with the ONNX metadata and `params` as run params."""
    import mlflow
    from dotenv import load_dotenv

    from triton_flavor import log_model

    load_dotenv()
    mlflow.set_experiment(experiment)  # creates the experiment if it doesn't exist
    meta = {p.key: p.value for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props}
    with tempfile.TemporaryDirectory() as tmp, mlflow.start_run() as run:
        model_dir = build_triton_model(onnx_path, Path(tmp), hw)
        log_model(
            str(model_dir),
            artifact_path="models",
            registered_model_name=registered_model_name,
            await_registration_for=10,
        )
        mlflow.log_params({**meta, "triton_h": hw[0], "triton_w": hw[1], "onnx_file": onnx_path.name, **(params or {})})
        print(f"logged {onnx_path} to experiment {experiment!r}, run {run.info.run_id}"
              + (f", registered as {registered_model_name!r}" if registered_model_name else ""))


def add_publish_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--triton-hw", type=int, nargs=2, default=(-1, -1), metavar=("H", "W"),
                    help="Input size in config.pbtxt; -1 -1 accepts any multiple of 32. Tiled mode: the tile network size.")
    ap.add_argument("--experiment", default="xfeat", help="MLflow experiment.")
    ap.add_argument("--registered-model-name", default=None,
                    help="Also register the model under this name (otherwise register the run's model in the MLflow UI).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path, help="ONNX file written by export_xfeat_onnx.py.")
    add_publish_args(ap)
    ap.add_argument("--dry-run", type=Path, default=None, metavar="DIR",
                    help="Only write the Triton model directory to DIR/model, don't log anything.")
    args = ap.parse_args()

    hw = tuple(args.triton_hw)
    if args.dry_run is not None:
        print(f"wrote {build_triton_model(args.onnx, args.dry_run, hw)}")
    else:
        publish(args.onnx, hw, args.experiment, args.registered_model_name)


if __name__ == "__main__":
    main()
