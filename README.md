# viam-sequence-to-lerobot

Convert a [Viam](https://www.viam.com) sequence dataset export into a
[LeRobot](https://github.com/huggingface/lerobot) v3 dataset for fine-tuning a
VLA policy (SmolVLA, pi0, ACT, ...).

## Before: capture and export from Viam

1. **Capture** on your machine: each camera via `GetImages` and the arm via
   `JointPositions`, all at the same rate (e.g. 10 Hz). One recorded
   demonstration = one sequence.
2. **Create sequences** over each demonstration's time range (part, resources,
   start/end) and add them to a **sequence dataset**.
3. **Export** it:

   ```sh
   viam dataset export --dataset-id <id> --destination <export_dir>
   ```

   This writes the parquet zip and a `binary_data/` directory of images —
   exactly what the converter reads.

## Convert

```sh
uv venv --python 3.11 && uv pip install -e .
viam-seq-to-lerobot <export_dir> \
    --task "open the box" \
    --camera webcam-teleop --camera realsense-cam-teleop \
    --repo-id viam/open-box \
    --output-root ~/datasets/open-box-lerobot
```

Semantics (run `--help` for all flags):

- Each sequence becomes one episode; the **first** `--camera` defines the
  frame clock, and joint readings / other cameras are matched to it by nearest
  timestamp (`--tolerance-s`, default 50 ms).
- `observation.state` = joint angles; `action` = joint angles at the next
  frame (next-state-as-action). Viam-native units are kept (degrees).
- Sequences missing a listed camera and episodes shorter than `--min-frames`
  are skipped, with reasons logged.
- Frames are encoded as MP4 video; the output loads with `LeRobotDataset`
  as-is.

## After: fine-tune and deploy

```sh
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.push_to_hub=false \
    --dataset.repo_id=viam/open-box \
    --dataset.root=~/datasets/open-box-lerobot \
    --rename_map='{"observation.images.webcam_teleop": "observation.images.camera1",
                   "observation.images.realsense_cam_teleop": "observation.images.camera2"}'
```

- `smolvla_base` expects camera keys `camera1/2/3` — map yours with
  `--rename_map`; unused slots are fine.
- Train on a CUDA GPU for real runs; at rollout, lower `n_action_steps`
  (e.g. 5–10) so the policy replans frequently.
- The inference client must mirror the dataset contract: build the state from
  `JointPositions` (degrees) exactly as captured, and send the policy's output
  to `MoveToJointPositions` (degrees).

## Development

```sh
uv pip install -e '.[dev]'
pytest
```
