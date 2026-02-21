#!/usr/bin/env python3
"""
ROS-free TCP inference server for Kuavo diffusion policy (LeRobot-based).

Design goals
- Centralize configuration in ONE place: your kuavo_real_env.yaml (no separate "TCP-only" config class).
- Load trained policy using Config_Inference (task/method/timestamp[/epoch]).
- Provide a configurable mapping layer so robot camera names can map to the keys the policy was trained on.
- Keep "delta support" purely as a configurable OUTPUT MODE (default: absolute),
  because most diffusion trainings here are on absolute joint targets.

Wire protocol (msgpack + 4-byte big-endian length prefix)
Client -> Server (dict):
{
  "state": [float, ...],                        # e.g., length 44
  "images": { "head_cam_h": <jpeg_bytes>, ... } # JPEG bytes recommended
}

Server -> Client (dict):
{
  "action": [float, ...],
  "mode": "absolute" | "delta" | "delta_to_absolute"
}

YAML keys (all inside the same kuavo_real_env.yaml):
  # Required by existing repo loaders (inference)
  task, method, timestamp, epoch, policy_type, device, use_delta (optional/legacy)

  # Additional keys used by this TCP server (still centralized in same YAML):
  obs_key_map:                 # robot cam name -> model obs key
    head_cam_h: "observation.images.ego_view"
    wrist_cam_l: "observation.images.left_wrist_view"
    wrist_cam_r: "observation.images.right_wrist_view"
  state_key_in: "observation.state"
  action_key_out: "action"
  model_image_size: [256, 256]         # (H, W) expected by the model (training)
  missing_camera_policy: "error"       # or "duplicate:head_cam_h"
  required_model_image_keys:           # optional override
    - "observation.images.ego_view"
    - "observation.images.left_wrist_view"
    - "observation.images.right_wrist_view"

  # Output mode (THIS is your "delta as an option")
  # - "absolute" (default): return model output as absolute targets
  # - "delta": return model output as delta (no integration)
  # - "delta_to_absolute": return state + delta (server integrates)
  action_mode: "absolute"

  # Only used if action_mode is delta or delta_to_absolute
  delta_clip: 0.1    # set null to disable

Checkpoint selection
- Prefer: outputs/train/{task}/{method}/{timestamp}/best
- Fallback: outputs/train/{task}/{method}/{timestamp} (run folder)
- Fallback: outputs/train/{task}/{method}/{timestamp}/epoch{epoch}

Run:
  python kuavo_deploy/tcp_policy_server.py --config configs/deploy/kuavo_real_env.yaml --port 5555
"""

import argparse
import socket
import struct
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import msgpack
import numpy as np
import torch
import yaml
import cv2
from torchvision.transforms.functional import to_tensor

# Repo config loaders (provided by your repo)
from configs.deploy.config_inference import load_inference_config
from configs.deploy.config_kuavo_env import load_kuavo_env_config  # loaded for consistency / validation

# Policy wrappers (provided by your repo)
from kuavo_train.wrapper.policy.diffusion.DiffusionPolicyWrapper import CustomDiffusionPolicyWrapper
from lerobot.policies.act.modeling_act import ACTPolicy


# -----------------------
# YAML helpers
# -----------------------
def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_tuple_hw(cfg: Dict[str, Any], key: str, default_hw: Tuple[int, int]) -> Tuple[int, int]:
    v = cfg.get(key, list(default_hw))
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        return default_hw
    return int(v[0]), int(v[1])  # (H, W)


# -----------------------
# Msgpack TCP utilities
# -----------------------
def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Client disconnected")
        buf += chunk
    return buf


def recv_msg(conn: socket.socket) -> Dict[str, Any]:
    header = recv_exact(conn, 4)
    (n,) = struct.unpack("!I", header)
    payload = recv_exact(conn, n)
    return msgpack.unpackb(payload, raw=False)


def send_msg(conn: socket.socket, obj: Dict[str, Any]) -> None:
    out = msgpack.packb(obj, use_bin_type=True)
    conn.sendall(struct.pack("!I", len(out)) + out)


# -----------------------
# Preprocessing
# -----------------------
def decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Decode JPEG bytes -> RGB uint8 (H, W, 3)."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode JPEG image bytes")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def resize_rgb(img_rgb: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Resize RGB uint8 image to (H, W)."""
    th, tw = target_hw
    return cv2.resize(img_rgb, (tw, th), interpolation=cv2.INTER_AREA)


def build_model_observation(
    request: Dict[str, Any],
    device: torch.device,
    required_model_image_keys: Tuple[str, ...],
    obs_key_map: Dict[str, str],
    state_key: str,
    model_image_size: Tuple[int, int],
    missing_camera_policy: str,
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """
    Build the torch observation dict expected by policy.select_action().

    request:
      {
        "state": [...],
        "images": {"head_cam_h": <jpeg_bytes>, ...}
      }

    Returns:
      observation_torch: dict[str, torch.Tensor] (batched: leading dim = 1)
      state_np: np.ndarray state vector (used for delta_to_absolute mode)
    """
    if "state" not in request:
        raise KeyError("Missing required field 'state' in request")
    if "images" not in request:
        raise KeyError("Missing required field 'images' in request")

    state_np = np.asarray(request["state"], dtype=np.float32)
    images_in = request["images"]
    if not isinstance(images_in, dict):
        raise TypeError("'images' must be a dict of {camera_name: jpeg_bytes}")

    # Decode all provided images so we can:
    # - map robot camera names -> model keys
    # - optionally duplicate a provided camera to fill missing required model keys
    robot_cam_to_rgb: Dict[str, np.ndarray] = {}
    model_images_np: Dict[str, np.ndarray] = {}

    for robot_cam, blob in images_in.items():
        if not isinstance(blob, (bytes, bytearray)):
            raise TypeError(f"images['{robot_cam}'] must be bytes (JPEG recommended)")
        rgb = decode_jpeg(blob)
        rgb = resize_rgb(rgb, model_image_size)
        robot_cam_to_rgb[robot_cam] = rgb

        # If this robot cam is mapped to a model key, store it.
        if robot_cam in obs_key_map:
            model_key = obs_key_map[robot_cam]
            model_images_np[model_key] = rgb
        # Also allow clients to directly send model-key names as keys.
        elif robot_cam.startswith("observation.images."):
            model_images_np[robot_cam] = rgb

    # Ensure all required model keys exist (fill missing if configured)
    for req_key in required_model_image_keys:
        if req_key in model_images_np:
            continue

        if isinstance(missing_camera_policy, str) and missing_camera_policy.startswith("duplicate:"):
            src = missing_camera_policy.split(":", 1)[1].strip()
            if src not in robot_cam_to_rgb:
                raise KeyError(
                    f"Missing required model image key '{req_key}'. "
                    f"missing_camera_policy requested duplicate from '{src}', but it was not provided."
                )
            model_images_np[req_key] = robot_cam_to_rgb[src]
        else:
            raise KeyError(
                f"Missing required model image key '{req_key}'. "
                f"Provided robot cameras: {sorted(list(robot_cam_to_rgb.keys()))}. "
                f"Either provide/match it via obs_key_map, or set missing_camera_policy: 'duplicate:<camera>'."
            )

    observation: Dict[str, torch.Tensor] = {}

    # Images -> torch (float32, [0,1], CHW, batched)
    for k, img_rgb in model_images_np.items():
        observation[k] = to_tensor(img_rgb).unsqueeze(0).to(device, non_blocking=True)

    # State -> torch (float32, batched)
    observation[state_key] = torch.from_numpy(state_np).float().unsqueeze(0).to(device, non_blocking=True)

    return observation, state_np


# -----------------------
# Policy loading + infer
# -----------------------
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


def select_checkpoint_dir(infer_cfg, repo_root: Path) -> Path:
    """
    Pick a checkpoint directory that contains config.json and model.safetensors.

    Preference order:
      1) .../{timestamp}/best
      2) .../{timestamp} (run folder)
      3) .../{timestamp}/epoch{epoch}
    """
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
        "Expected to find both: config.json and model.safetensors in one of those directories.\n"
        "Check task/method/timestamp/epoch in your YAML, and confirm where training saved outputs."
    )


def apply_action_mode(
    action_np: np.ndarray,
    state_np: np.ndarray,
    action_mode: str,
    delta_clip: Optional[float],
) -> Tuple[np.ndarray, str]:
    """
    action_mode:
      - "absolute" (default): return action as-is
      - "delta": return delta action (optionally clipped)
      - "delta_to_absolute": return state + delta (optionally clipped)
    """
    if action_mode is None:
        action_mode = "absolute"
    action_mode = str(action_mode).lower().strip()

    if action_mode == "absolute":
        return action_np, "absolute"

    if action_mode in ("delta", "delta_to_absolute"):
        if delta_clip is not None:
            dc = float(delta_clip)
            action_np = np.clip(action_np, -dc, dc)
        if action_mode == "delta":
            return action_np, "delta"
        return (state_np + action_np), "delta_to_absolute"

    raise ValueError(f"action_mode must be one of: absolute | delta | delta_to_absolute. Got: {action_mode}")


# -----------------------
# Server
# -----------------------
def serve(config_path: str, host: str, port: int):
    infer_cfg = load_inference_config(config_path)
    _ = load_kuavo_env_config(config_path)  # keep repo's YAML parsing consistent

    user_cfg = read_yaml(config_path)

    obs_key_map = user_cfg.get("obs_key_map") or {}
    state_key = user_cfg.get("state_key_in", "observation.state")
    model_image_size = get_tuple_hw(user_cfg, "model_image_size", (256, 256))
    missing_camera_policy = user_cfg.get("missing_camera_policy", "error")

    required_keys = user_cfg.get("required_model_image_keys", None)
    if isinstance(required_keys, list) and all(isinstance(x, str) for x in required_keys) and len(required_keys) > 0:
        required_model_image_keys = tuple(required_keys)
    else:
        required_model_image_keys = (
            "observation.images.ego_view",
            "observation.images.left_wrist_view",
            "observation.images.right_wrist_view",
        )

    # Delta as an OPTION: controlled by YAML; default absolute.
    action_mode = user_cfg.get("action_mode", "absolute")
    delta_clip = user_cfg.get("delta_clip", 0.1)
    delta_clip_val = None if delta_clip is None else float(delta_clip)

    device = torch.device(infer_cfg.device)

    # Repo root = parent of this script's folder (kuavo_deploy/..)
    repo_root = Path(__file__).resolve().parent.parent
    pretrained_path = select_checkpoint_dir(infer_cfg, repo_root=repo_root)
    policy = load_policy(pretrained_path, infer_cfg.policy_type, device)

    print(f"TCP policy server listening on {host}:{port}")
    print(f"Loaded policy from: {pretrained_path}")
    print(f"policy_type={infer_cfg.policy_type} device={infer_cfg.device}")
    print(f"action_mode={action_mode} delta_clip={delta_clip_val}")
    print(f"model_image_size={model_image_size}")
    print(f"required_model_image_keys={required_model_image_keys}")
    if obs_key_map:
        print(f"obs_key_map keys={list(obs_key_map.keys())}")

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen(1)

    while True:
        conn, addr = s.accept()
        print(f"Client connected: {addr}")
        try:
            policy.reset()
            while True:
                req = recv_msg(conn)

                observation, state_np = build_model_observation(
                    request=req,
                    device=device,
                    required_model_image_keys=required_model_image_keys,
                    obs_key_map=obs_key_map,
                    state_key=state_key,
                    model_image_size=model_image_size,
                    missing_camera_policy=missing_camera_policy,
                )

                with torch.inference_mode():
                    act_t = policy.select_action(observation)

                action_np = act_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

                action_out, mode = apply_action_mode(
                    action_np=action_np,
                    state_np=state_np,
                    action_mode=action_mode,
                    delta_clip=delta_clip_val,
                )

                send_msg(conn, {"action": action_out.tolist(), "mode": mode})

        except Exception as e:
            print(f"Client disconnected / error: {e}")
        finally:
            conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML (e.g., kuavo_real_env.yaml)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()
    serve(config_path=args.config, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
