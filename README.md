## Installation

Prerequisites:
- CUDA version >=11.8 (this is required if you want to perform a full installation of this repo and perform RT-1 or Octo inference)
- An NVIDIA GPU (ideally RTX; for non-RTX GPUs, such as 1080Ti and A100, environments that involve ray tracing will be slow). Currently TPU is not supported as SAPIEN requires a GPU to run.

For GlassVLA testing, also complete the following:
- Install the Grounded-SAM-2 repository first(https://github.com/IDEA-Research/Grounded-SAM-2).
- Update SAM2 and GroundingDINO paths in [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py) to match your local machine:
	- [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py#L25): `grounded_sam2_root` (Grounded-SAM-2 repo root path)
	- [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py#L768): `grounding_dino_config` (GroundingDINO config file path)
	- [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py#L769): `grounding_dino_checkpoint` (GroundingDINO checkpoint `.pth` path)
	- [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py#L770): `sam2_checkpoint` (SAM2 checkpoint `.pt` path)
	- [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py#L772): `sam2_config` (SAM2 Hydra config module path)

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

If you'd like to perform evaluations on provided agents (e.g., RT-1, Octo, OpenVLA), or add new robots and environments, follow the full installation instructions below.

Environment export yml reference (for `GlassVLA`):
- [env_exports/simpler_gs2_mlp/environment-history-20260401.yml](env_exports/simpler_gs2_mlp/environment-history-20260401.yml)
- [env_exports/simpler_gs2_mlp/environment-full-20260401.yml](env_exports/simpler_gs2_mlp/environment-full-20260401.yml)

Run the GlassVLA test script from [scripts/run_glassvla-4b-sam2-text-10k.sh](scripts/run_glassvla-4b-sam2-text-10k.sh).

Example command:
```bash
bash scripts/run_glassvla-4b-sam2-text-10k.sh
```

## GlassVLA Checkpoint Setup

For GlassVLA evaluation, please use the following checkpoint and config sources:

- GlassVLA checkpoint: download from Hugging Face `chenglongy/GlassVLA-4b-224-fractal`.
- SAM2 checkpoint and config: use the default checkpoint and default config provided by the Grounded-SAM-2 repository.
- GroundingDINO checkpoint: download from Hugging Face `chenglongy/groundingdino_GlassVLA`.
- GroundingDINO config: follow the default Grounded-SAM-2 GroundingDINO config setting.

After downloading, update the corresponding paths in [simpler_env/utils/image_simplification.py](simpler_env/utils/image_simplification.py) as described in the Installation section above.
