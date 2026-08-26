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
- The inference client must mirror the dataset contract: build the state from
  `JointPositions` (degrees) exactly as captured, and send the policy's output
  to `MoveToJointPositions` (degrees).

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
