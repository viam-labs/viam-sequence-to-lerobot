# viam-sequence-to-lerobot

Convert a [Viam](https://www.viam.com) sequence dataset export into a
[LeRobot](https://github.com/huggingface/lerobot) v3 dataset for fine-tuning a
VLA policy (SmolVLA, pi0, ACT, ...).

## Before: capture and export from Viam

1. **Capture** on your machine: each camera via `GetImages` and the arm via
   `JointPositions` (plus `EndPosition` if you want `--action-space delta-ee`),
   all at the same rate (e.g. 10 Hz). One recorded demonstration = one
   sequence.
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
- `--action-space joints` (default): `observation.state` = joint angles;
  `action` = joint angles at the next frame (next-state-as-action).
  Viam-native units are kept (degrees).
- `--action-space delta-ee`, from the arm's `EndPosition` readings.
  `observation.state` is 9 dims: `[x, y, z]` in millimeters, then
  `[r00, r01, r02, r10, r11, r12]` — the first two rows of the pose's 3×3
  rotation matrix. `action` is 6 dims: `[dx, dy, dz]` in millimeters plus
  `[drx, dry, drz]`, the body-frame rotation `R_t⁻¹·R_{t+1}` as an axis-angle
  vector in radians. Sequences with no usable `EndPosition` are skipped, and so
  is any frame whose delta would span a gap in the alignment.

  Rotation takes six dims because no three-number encoding is continuous
  everywhere, and this arm holds its tool within 0.03 rad of π — exactly where
  an axis-angle state flips sign under smooth motion. Actions keep three
  because a per-tick rotation is ~0.02 rad, far from that cut.
- Sequences missing a listed camera and episodes shorter than `--min-frames`
  are skipped, with reasons logged.
- Frames are encoded as MP4 video; the output loads with `LeRobotDataset`
  as-is.

## After: fine-tune and deploy

```sh
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --policy.input_features=null \
    --policy.push_to_hub=false \
    --dataset.repo_id=viam/open-box \
    --dataset.root=~/datasets/open-box-lerobot
```

- **`--policy.input_features=null` matters if you have fewer than three
  cameras.** Without it, the checkpoint declares whatever `smolvla_base`
  declares — `camera1/2/3` — no matter how many cameras your dataset has, so a
  2-camera dataset produces a checkpoint claiming three. `null` tells lerobot to
  derive the features from your dataset instead, and your camera keys keep their
  real names (`observation.images.webcam_teleop`), which read better than
  `camera1` anyway. Details in [Camera keys](#camera-keys) below.
- Train on a CUDA GPU for real runs; at rollout, lower `n_action_steps`
  (e.g. 5–10) so the policy replans frequently.
- The inference client must mirror the dataset contract. For `joints`
  datasets: build the state from `JointPositions` (degrees) exactly as
  captured, and send the policy's output to `MoveToJointPositions` (degrees).
  For `delta-ee` datasets, use or copy the helpers in
  `viam_sequence_to_lerobot.pose` so the encoding cannot drift apart from the
  converter's:

  ```python
  from viam_sequence_to_lerobot.pose import (
      orientation_vector, pose_state, state_compose, state_rotation,
  )

  state = pose_state({"pose": pose_fields})        # live EndPosition -> 9 dims
  delta = policy(state, images, task)              # 6 dims
  target = state_compose(state, delta)             # 9 dims
  arm.move_to_position(Pose(*target[:3], **orientation_vector(state_rotation(target))))
  ```

  Two ways to get this silently wrong: transposing the rotation (the state
  holds matrix *rows*), and left-multiplying the delta, which applies it in the
  world frame instead of the body frame. `state_compose` does both correctly;
  call it rather than reimplementing it.

### Training on HF Jobs

`scripts/train_job.py` submits the same command to
[HF Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) so the flags
live in a TOML file instead of your shell history. Copy
`scripts/train_job.toml` per experiment; `[job]` keys become `hf jobs run`
flags and `[train]` keys become `lerobot-train` flags.

```sh
scripts/train_job.py my-run.toml                 # fresh run
scripts/train_job.py my-run.toml --steps=50000   # any --key=value overrides the file
scripts/train_job.py my-run.toml --resume        # continue from the latest Hub checkpoint
scripts/train_job.py my-run.toml --dry-run       # print the command only
```

`--resume` needs `save_checkpoint_to_hub = true` on the original run: lerobot
pushes `checkpoints/<step>/` into `policy.repo_id`, and the resumed pod pulls
the highest step from there and continues (pass `--steps=N` to extend the run).

### Camera keys

`smolvla_base` declares three image features (`observation.images.camera1/2/3`),
and the older advice here was to map yours onto them with `--rename_map` and
leave unused slots alone. That works for training, but it bakes a wrong
declaration into the checkpoint you ship, for two reasons:

- `lerobot/configs/train.py:285` refuses a `--rename_map` unless a pretrained
  checkpoint is given, so the base's `input_features` passes through verbatim —
  including camera slots your dataset never filled.
- `lerobot/policies/factory.py:395` **skips** the camera-consistency check
  entirely when a `rename_map` is set, so nothing warns you about the mismatch.

Training itself is unharmed — `modeling_smolvla.py:340-346` builds each batch
from whatever keys are present and drops the rest — but anything reading
`config.json` afterwards will try to supply a camera that does not exist, and
there is no safe filler: black, mid-gray, and a duplicate of another camera each
shift the predicted action chunk by several degrees versus omitting the key.

`--policy.input_features=null` avoids all of it. `input_features` is documented
as accepting `None` "in order to infer those values from the dataset"
(`lerobot/configs/policies.py:58`), and dropping `--rename_map` re-enables the
consistency check. Weights are unaffected either way: SmolVLA runs every camera
through one shared vision tower, so the slot names are labels with no parameters
behind them.

Note that `null` must be spelled exactly that — `--policy.input_features='{}'`
does **not** work, because draccus merges an empty dict into the pretrained
config's rather than replacing it, leaving all three inherited features in place.

One knock-on effect worth knowing: the derived features carry your cameras'
**native** resolutions, not the base model's `[3, 256, 256]`. For a 1080p feed
that means `config.json` declares `[3, 1920, 1080]`, and an inference client
sizing its payloads from `input_features` will send full-resolution frames. That
is closer to what training actually saw — SmolVLA letterboxes to 512x512
internally either way — but keep the transport on JPEG; a raw 1080p frame
base64-encodes to roughly 8 MB and will hit gRPC message limits.

Use `--rename_map` only when you deliberately want the base model's key names —
for example to stay drop-in compatible with an existing inference client. If you
already have a checkpoint declaring an unused camera, either delete the key from
its `config.json` (weight-compatible, byte-identical output) or tell the
inference side to ignore it.

## Development

```sh
uv pip install -e '.[dev]'
pytest
```
