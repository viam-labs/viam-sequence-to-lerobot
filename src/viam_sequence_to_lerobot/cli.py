"""Command-line entry point: viam-seq-to-lerobot."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .convert import ConversionConfig, convert

logger = logging.getLogger(__name__)

EPILOG = """\
example:
  viam-seq-to-lerobot ~/Downloads/workshop-2-sequences \\
      --task "open the box" \\
      --repo-id viam/open-box \\
      --output-root ~/datasets/open-box-lerobot

Each Viam sequence becomes one episode. The camera stream provides the frame
clock; arm readings are matched to each frame by nearest timestamp. With
--action-space joints (default), observation.state is the joint angles at each
frame and action is the joint angles at the next frame. With --action-space
delta-ee, observation.state is the absolute end-effector pose as [x,y,z] in mm
plus the first two rows of its rotation matrix (9 dims), and action is the
body-frame delta to the next frame's pose as [dx,dy,dz] in mm plus an
axis-angle rotation in radians (6 dims). Camera frames are encoded as MP4
video. The result loads with LeRobotDataset and trains with lerobot-train
as-is.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="viam-seq-to-lerobot",
        description="Convert a Viam sequence dataset export into a LeRobot v3 dataset "
        "for fine-tuning a VLA policy (SmolVLA, pi0, ACT, ...).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "export_dir",
        type=Path,
        help="Viam sequence export directory: the folder containing "
        "sequences.parquet, tabular_data.parquet, binary_data.parquet "
        "(directly or in a .zip) and the binary_data/ image files",
    )
    parser.add_argument(
        "--task",
        required=True,
        metavar="TEXT",
        help='natural-language instruction stored with every frame and used to '
        'condition the policy, e.g. "open the box"',
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        metavar="DIR",
        default=None,
        help="directory to write the LeRobot dataset to "
        "(default: <export_dir>-lerobot next to the export)",
    )
    parser.add_argument(
        "--repo-id",
        metavar="NAMESPACE/NAME",
        default=None,
        help="dataset name in Hugging Face Hub convention, used when loading "
        "(lerobot-train --dataset.repo_id=...) and as the target repo if the "
        "dataset is ever pushed to the Hub "
        "(default: viam/<export dir name>)",
    )
    parser.add_argument(
        "--camera",
        metavar="COMPONENT",
        action="append",
        dest="cameras",
        default=None,
        help="name of a Viam camera component to include as "
        "observation.images.<name>; repeat the flag for multiple cameras. "
        "The first camera defines the frame clock; sequences missing any "
        "listed camera are skipped (default: webcam-teleop)",
    )
    parser.add_argument(
        "--arm",
        metavar="COMPONENT",
        default="xarm",
        help="name of the Viam arm component whose readings become "
        "observation.state and action (default: %(default)s)",
    )
    parser.add_argument(
        "--action-space",
        choices=("joints", "delta-ee"),
        default="joints",
        help="joints: state/action are JointPositions angles (action = next "
        "frame's joints). delta-ee: state is the absolute EndPosition pose as "
        "[x,y,z] plus two rotation-matrix rows (9 dims) and action is the "
        "body-frame delta as [dx,dy,dz] plus an axis-angle rotation (6 dims); "
        "at inference, compose the delta onto the live EndPosition and call "
        "MoveToPosition (default: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="capture rate of the export in frames per second; stored in the "
        "dataset metadata and used to time video playback during training "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--tolerance-s",
        type=float,
        metavar="SECONDS",
        default=0.05,
        help="maximum clock offset allowed when matching a joint reading to a "
        "camera frame; frames with no reading within this window are dropped "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        metavar="N",
        default=10,
        help="skip sequences that yield fewer than N usable frames, e.g. "
        "aborted or accidental recordings (default: %(default)s)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        metavar="N",
        default=None,
        help="downscale every camera frame to NxN before encoding. VLA "
        "backbones resize to a square anyway (EVO1/InternVL3 uses 448), so "
        "this mostly removes wasted encode time, disk, and per-epoch decode "
        "cost. Omit to keep the captured resolution",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="log per-frame alignment decisions and other debug detail",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    export_dir = args.export_dir.expanduser().resolve()
    output_root = args.output_root or export_dir.with_name(export_dir.name + "-lerobot")
    repo_id = args.repo_id or f"viam/{export_dir.name}"

    config = ConversionConfig(
        export_dir=export_dir,
        output_root=output_root,
        repo_id=repo_id,
        task=args.task,
        camera_components=tuple(args.cameras) if args.cameras else ("webcam-teleop",),
        arm_component=args.arm,
        action_space=args.action_space,
        fps=args.fps,
        tolerance_s=args.tolerance_s,
        min_frames=args.min_frames,
        image_size=args.image_size,
    )
    try:
        summary = convert(config)
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if summary.skipped:
        logger.info("Skipped sequences:")
        for sequence_id, reason in summary.skipped:
            logger.info("  %s: %s", sequence_id, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
