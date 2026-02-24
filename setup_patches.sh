#!/bin/bash
# Run once after cloning/pulling on any new machine

VIDEO_UTILS=third_party/diffusion_policy/third_party/lerobot/src/lerobot/datasets/video_utils.py
CONVERT_STATS=third_party/diffusion_policy/third_party/lerobot/src/lerobot/datasets/v21/convert_stats.py

sed -i 's/if importlib.util.find_spec("torchcodec"):/if False:  # torchcodec disabled - no FFmpeg/' $VIDEO_UTILS
sed -i 's/    if backend == "torchcodec":/    if False:  # torchcodec disabled/' $VIDEO_UTILS
sed -i 's/        ep_stats\[key\] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)/        if isinstance(ep_ft_data, np.ndarray) and not np.issubdtype(ep_ft_data.dtype, np.number):\n            continue\n        ep_stats[key] = get_feature_stats(ep_ft_data, axis=axes_to_reduce, keepdims=keepdims)/' $CONVERT_STATS

echo "Patches applied."
