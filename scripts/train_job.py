#!/usr/bin/env python3
"""Submit a lerobot-train run to HF Jobs from a TOML config.

    scripts/train_job.py                      # fresh run from scripts/train_job.toml
    scripts/train_job.py my.toml --steps=50000 # override any lerobot-train flag
    scripts/train_job.py --resume             # continue from the latest Hub checkpoint
    scripts/train_job.py --resume other/repo  # ...or from another repo's checkpoints
    scripts/train_job.py --dry-run            # print the command only

Resume works because the pod runs `lerobot-train --resume=true --config_path=<repo>`,
which downloads `checkpoints/<latest step>/` from that repo (written by
save_checkpoint_to_hub) and continues training there. Requires Python 3.11+.
"""

import argparse
import json
import shlex
import subprocess
import sys
import tomllib
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).with_name("train_job.toml")


def flag(key: str, value) -> list[str]:
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, (dict, list)):
        value = json.dumps(value)
    return [f"--{key}={value}"]


def build_command(cfg: dict, resume: str | None, overrides: list[str]) -> list[str]:
    job = dict(cfg["job"])
    train = dict(cfg["train"])
    for tok in overrides:  # --key=value wins over the config file
        if not tok.startswith("--") or "=" not in tok:
            sys.exit(f"overrides must look like --key=value, got {tok!r}")
        key, value = tok[2:].split("=", 1)
        train[key] = value

    if resume is not None:
        # Mirrors lerobot.jobs.hf._build_resume_job: the checkpoint's own config names the
        # policy and its repo, so a --policy.path here would win over --resume.
        source = resume or train["policy.repo_id"]
        train.pop("policy.path", None)
        train.pop("policy.repo_id", None)
        train = {"resume": True, "config_path": source, **train}

    cmd = ["hf", "jobs", "run"]
    for key, value in job.items():
        if key == "image":
            continue
        if isinstance(value, bool):
            cmd += [f"--{key}"] if value else []
        elif isinstance(value, list):
            for v in value:
                cmd += [f"--{key}", str(v)]
        else:
            cmd += [f"--{key}", str(value)]
    cmd += [job["image"], "lerobot-train"]
    for key, value in train.items():
        cmd += flag(key, value)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--resume", nargs="?", const="", metavar="REPO_ID",
                    help="resume from the latest Hub checkpoint (default: policy.repo_id)")
    ap.add_argument("--dry-run", action="store_true")
    args, overrides = ap.parse_known_args()

    cfg = tomllib.loads(args.config.read_text())
    cmd = build_command(cfg, args.resume, overrides)
    print(shlex.join(cmd))
    if not args.dry_run:
        sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
