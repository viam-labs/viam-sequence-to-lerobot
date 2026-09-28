#!/usr/bin/env python3
"""Export a Viam sequence dataset, convert it, and push it to the Hub, all on HF Jobs.

    scripts/prepare_job.py                                  # from scripts/prepare_job.toml
    scripts/prepare_job.py my.toml --then-train my-run.toml # also submit training after the push
    scripts/prepare_job.py --dry-run                        # print the command only

The data never touches this machine: the pod downloads from Viam and uploads to the Hub.
Export VIAM_API_KEY_ID and VIAM_API_KEY locally first; bare secret names are read from
your environment. --then-train builds the train_job.py command here and runs it in the
pod after the push succeeds, so training starts on the new data with nothing left to
wait on locally. Requires Python 3.11+.
"""

import argparse
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

from train_job import build_command, job_command

DEFAULT_CONFIG = Path(__file__).with_name("prepare_job.toml")
VIAM_CLI = "https://storage.googleapis.com/packages.viam.com/apps/viam-cli/viam-cli-stable-linux-amd64"
PACKAGE = "git+https://github.com/viam-labs/viam-sequence-to-lerobot"

# LeRobotDataset loads the v3.0 tag, and push_to_hub moves it; pin the old data first.
# exist_ok keeps a rerun from re-pointing the archive at data an earlier run already pushed.
ARCHIVE = """
import sys
from huggingface_hub import HfApi
repo, tag = sys.argv[1:]
api = HfApi()
if api.repo_exists(repo, repo_type="dataset"):
    api.create_tag(repo, tag=tag, revision="v3.0", repo_type="dataset", exist_ok=True)
"""

PUSH = """
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
LeRobotDataset(sys.argv[1], root=sys.argv[2]).push_to_hub()
"""


def cli_flags(table: dict) -> list[str]:
    """--key=value per entry; a list repeats the flag (--camera=a --camera=b)."""
    return [f"--{k}={v}" for k, vals in table.items() for v in (vals if isinstance(vals, list) else [vals])]


def build_script(cfg: dict, train_cmd: list[str] | None) -> str:
    prep = cfg.get("prepare", {})
    repo = cfg["convert"]["repo-id"]
    lines = [
        "set -euo pipefail",  # no -x: it would print the expanded Viam key into the job logs
        f"curl -fsSL {VIAM_CLI} -o /usr/local/bin/viam && chmod +x /usr/local/bin/viam",
        f"pip install -q uv && uv pip install --system -q {shlex.quote(PACKAGE + '@' + prep.get('ref', 'main'))}",
        'viam login api-key --key-id "$VIAM_API_KEY_ID" --key "$VIAM_API_KEY"',
    ]
    if tag := prep.get("archive_tag"):
        lines.append(shlex.join(["python", "-c", ARCHIVE, repo, tag]))
    lines += [
        shlex.join(["viam", "dataset", "export", "--destination=export", *cli_flags(cfg["export"])]),
        shlex.join(["viam-seq-to-lerobot", "export", "--output-root=lerobot", *cli_flags(cfg["convert"])]),
        shlex.join(["python", "-c", PUSH, repo, "lerobot"]),
    ]
    if train_cmd:
        lines.append(shlex.join(train_cmd))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--then-train", type=Path, metavar="TRAIN_TOML",
                    help="train_job.py config to submit from the pod once the push succeeds")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = tomllib.loads(args.config.read_text())
    train_cmd = None
    if args.then_train:
        train_cfg = tomllib.loads(args.then_train.read_text())
        # Detached, or this pod would idle (and bill) until training finished.
        train_cfg["job"] = {**train_cfg["job"], "detach": True}
        train_cmd = build_command(train_cfg, None, [])
    cmd = job_command(cfg["job"]) + ["bash", "-c", build_script(cfg, train_cmd)]
    print(shlex.join(cmd))
    if not args.dry_run:
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
