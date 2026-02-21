#!/usr/bin/env python3
"""
Flask HTTP wrapper for the diffusion policy that matches your ROS-side GR00T-style request pattern.

Why this exists
- Your ROS controller uses requests.post(model_url, files=..., data=...) (multipart/form-data).
- Your diffusion policy server logic is currently TCP/msgpack.
- This wrapper exposes an HTTP /predict endpoint that:
    1) Validates required camera views
    2) Builds the 44D observation.state using your Kuavo layout
    3) Maps incoming camera keys -> model keys via kuavo_real_env.yaml (obs_key_map)
    4) Runs policy.select_action()
    5) Returns GR00T-like JSON action trajectories so your ROS controller doesn't change.

Endpoint
POST /predict
multipart/form-data:
  Files (images):
    ego_view: JPEG
    left_wrist_view: JPEG
    right_wrist_view: JPEG
  Fields:
    task_description: string (accepted but not used by diffusion policy unless you later add language)
    left_arm:  "a0,a1,...,a6"    (7)
    left_hand: "h0,...,h5"       (6)
    right_arm: "b0,...,b6"       (7)
    right_hand:"k0,...,k5"       (6)

Behavior:
- If ANY required camera is missing => returns HTTP 400 and does NOT run inference.
- Model is loaded once on startup for speed.

Run:
  python kuavo_deploy/diffusion_http_server.py --config configs/deploy/kuavo_real_env.yaml --host 0.0.0.0 --port 8000

Then set in ROS:
  model_url = http://<model_host>:8000/predict
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import yaml
import cv2
from flask import Flask, request, jsonify
from torchvision.transforms.functional import to_tensor

from configs.deploy.config_inference import load_inference_config
from configs.deploy.config_kuavo_env import load_kuavo_env_config

from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy


def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_tuple_hw(cfg: Dict[str, Any], key: str, default_hw: Tuple[int, int]) -> Tuple[int, int]:
    v = cfg.get(key, list(default_hw))
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        return default_hw
    return int(v[0]), int(v[1])  # (H, W)


def decode_jpeg_bytes(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image bytes")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(img_rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    return cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)


def select_checkpoint_dir(infer_cfg, repo_root: Path) -> Path:
    base_run = repo_root / "outputs" / "train" / infer_cfg.task / infer_cfg.method / infer_cfg.timestamp
    best_dir = base_run / "best"
    epoch_dir = base_run / f"epoch{infer_cfg.epoch}"

    def _looks_like_ckpt_dir(p: Path) -> bool:
        return (p / "config.json").exists() and (p / "model.safetensors").exists()

    if _looks_like_ckpt_dir(best_dir):
        return best_dir
    if _looks_like_ckpt_dir(base_run):
        return base_run
    if _looks_like_ckpt_dir(epoch_dir):
        return epoch_dir

    raise FileNotFoundError(
        "Model checkpoint directory not found.\n"
        f"Tried:\n  - {best_dir}\n  - {base_run}\n  - {epoch_dir}\n"
        "Expected: config.json + model.safetensors."
    )


def load_policy(pretrained_path: Path, policy_type: str, device: torch.device):
    if policy_type == "diffusion":
        policy = CustomDiffusionPolicyWrapper.from_pretrained(pretrained_path, strict=True)
    elif policy_type == "act":
        policy = ACTPolicy.from_pretrained(pretrained_path, strict=True)
    else:
        raise ValueError(f"Unsupported policy_type: {policy_type}")
    policy.eval().to(device)
    policy.reset()
    return policy


def parse_csv_floats(s: str, n: int, name: str) -> np.ndarray:
    parts = [p.strip() for p in (s or "").split(",") if p.strip() != ""]
    if len(parts) != n:
        raise ValueError(f"{name} must have {n} values, got {len(parts)}")
    return np.asarray([float(x) for x in parts], dtype=np.float32)


def build_state44(left_arm7: np.ndarray, left_hand6: np.ndarray, right_arm7: np.ndarray, right_hand6: np.ndarray) -> np.ndarray:
    state = np.zeros((44,), dtype=np.float32)
    state[0:7] = left_arm7
    state[7:13] = left_hand6
    state[22:29] = right_arm7
    state[29:35] = right_hand6
    return state


def split_action44(action44: np.ndarray):
    left_arm = action44[0:7]
    left_hand = action44[7:13]
    right_arm = action44[22:29]
    right_hand = action44[29:35]
    return left_arm, right_arm, left_hand, right_hand


def repeat_traj(vec: np.ndarray, T: int) -> np.ndarray:
    return np.tile(vec.reshape(1, -1), (T, 1))


def create_app(config_path: str) -> Flask:
    infer_cfg = load_inference_config(config_path)
    _ = load_kuavo_env_config(config_path)

    user_cfg = read_yaml(config_path)

    required_cams = user_cfg.get("required_http_camera_keys", ["ego_view", "left_wrist_view", "right_wrist_view"])
    if not (isinstance(required_cams, list) and all(isinstance(x, str) for x in required_cams)):
        required_cams = ["ego_view", "left_wrist_view", "right_wrist_view"]

    obs_key_map = user_cfg.get("obs_key_map") or {
        "ego_view": "observation.images.ego_view",
        "left_wrist_view": "observation.images.left_wrist_view",
        "right_wrist_view": "observation.images.right_wrist_view",
    }

    state_key = user_cfg.get("state_key_in", "observation.state")
    model_image_size = get_tuple_hw(user_cfg, "model_image_size", (256, 256))

    device = torch.device(infer_cfg.device)

    repo_root = Path(__file__).resolve().parent.parent
    pretrained_path = select_checkpoint_dir(infer_cfg, repo_root=repo_root)
    policy = load_policy(pretrained_path, infer_cfg.policy_type, device)

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"ok": True, "policy_type": infer_cfg.policy_type, "device": str(device), "checkpoint": str(pretrained_path)})

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            missing = [k for k in required_cams if k not in request.files]
            if missing:
                return jsonify({
                    "success": False,
                    "error": "missing_camera",
                    "missing": missing,
                    "required": required_cams
                }), 400

            left_arm7 = parse_csv_floats(request.form.get("left_arm", ""), 7, "left_arm")
            right_arm7 = parse_csv_floats(request.form.get("right_arm", ""), 7, "right_arm")
            left_hand6 = parse_csv_floats(request.form.get("left_hand", ""), 6, "left_hand")
            right_hand6 = parse_csv_floats(request.form.get("right_hand", ""), 6, "right_hand")

            state44 = build_state44(left_arm7, left_hand6, right_arm7, right_hand6)

            observation: Dict[str, torch.Tensor] = {}

            for cam_key in required_cams:
                f = request.files[cam_key]
                img_bytes = f.read()
                rgb = decode_jpeg_bytes(img_bytes)
                rgb = resize_rgb(rgb, model_image_size)

                model_key = obs_key_map.get(cam_key, cam_key)
                if not model_key.startswith("observation.images."):
                    model_key = f"observation.images.{cam_key}"

                observation[model_key] = to_tensor(rgb).unsqueeze(0).to(device, non_blocking=True)

            observation[state_key] = torch.from_numpy(state44).float().unsqueeze(0).to(device, non_blocking=True)

            with torch.inference_mode():
                act_t = policy.select_action(observation)

            action44 = act_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            left_arm, right_arm, left_hand, right_hand = split_action44(action44)

            T = int(user_cfg.get("return_horizon_steps", 16))
            resp = {
                "success": True,
                "action": {
                    "left_arm": repeat_traj(left_arm, T).tolist(),
                    "right_arm": repeat_traj(right_arm, T).tolist(),
                    "left_hand": repeat_traj(left_hand, T).tolist(),
                    "right_hand": repeat_traj(right_hand, T).tolist(),
                },
                "meta": {
                    "task_description": request.form.get("task_description", ""),
                    "used_cameras": required_cams,
                    "checkpoint": str(pretrained_path),
                }
            }
            return jsonify(resp)

        except ValueError as e:
            return jsonify({"success": False, "error": "bad_request", "message": str(e)}), 400
        except Exception as e:
            return jsonify({"success": False, "error": "server_error", "message": str(e)}), 500

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML (e.g., kuavo_real_env.yaml)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = create_app(args.config)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
