import os
os.environ["AMDGPU_TARGETS"] = "gfx1032"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

import torch  # import AFTER setting env vars

# For AMD/ROCm, cuda calls are aliased to HIP — check this way:
print("ROCm available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("GPU not detected — check ROCm install")