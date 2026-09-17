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


def test_task_is_optional_and_prefix_defaults(synthetic_export, tmp_path, monkeypatch):
    config = run_main(synthetic_export, tmp_path, monkeypatch)
    assert config.task is None
    assert config.task_prefix == "cmd:"


def test_task_and_prefix_flags(synthetic_export, tmp_path, monkeypatch):
    config = run_main(
        synthetic_export, tmp_path, monkeypatch, "--task", "open the box", "--task-prefix", "task:"
    )
    assert config.task == "open the box"
    assert config.task_prefix == "task:"


def test_empty_task_prefix_is_a_clean_error(synthetic_export, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("viam_sequence_to_lerobot.cli.convert", lambda config: ConversionSummary())
    rc = main([str(synthetic_export), "--task-prefix", ""])
    assert rc == 1
    assert "--task-prefix" in caplog.text
