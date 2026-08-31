from __future__ import annotations

from viam_sequence_to_lerobot.cli import main
from viam_sequence_to_lerobot.convert import ConversionSummary


def run_main(synthetic_export, tmp_path, monkeypatch, *extra_args):
    captured = {}

    def fake_convert(config):
        captured["config"] = config
        return ConversionSummary()

    monkeypatch.setattr("viam_sequence_to_lerobot.cli.convert", fake_convert)
    rc = main(
        [
            str(synthetic_export),
            "--task",
            "open the box",
            "--output-root",
            str(tmp_path / "out"),
            *extra_args,
        ]
    )
    assert rc == 0
    return captured["config"]


def test_action_space_defaults_to_joints(synthetic_export, tmp_path, monkeypatch):
    config = run_main(synthetic_export, tmp_path, monkeypatch)
    assert config.action_space == "joints"


def test_action_space_delta_ee(synthetic_export, tmp_path, monkeypatch):
    config = run_main(synthetic_export, tmp_path, monkeypatch, "--action-space", "delta-ee")
    assert config.action_space == "delta-ee"
