import shutil

import pytest

from xray.config import CONFIG_DIR, ConfigError, enforce_no_synthetic_gates, load_sources


def test_real_sources_config_validates():
    cfg = load_sources()
    assert cfg.provider == "adzuna"
    assert cfg.countries and cfg.roles_to_track


def test_real_config_passes_synthetic_gates():
    enforce_no_synthetic_gates()


@pytest.mark.parametrize(
    ("fname", "line", "replacement"),
    [
        ("project.yaml", "allow_synthetic: false", "allow_synthetic: true"),
        ("labeling.yaml", "allow_synthetic_labels: false", "allow_synthetic_labels: true"),
        ("project.yaml", "allow_synthetic: false", "# gate removed"),
    ],
)
def test_flipped_or_missing_gate_refuses_to_run(tmp_path, fname, line, replacement):
    cfg_dir = tmp_path / "config"
    shutil.copytree(CONFIG_DIR, cfg_dir)
    path = cfg_dir / fname
    text = path.read_text(encoding="utf-8")
    assert line in text
    path.write_text(text.replace(line, replacement), encoding="utf-8")
    with pytest.raises(ConfigError, match="hard gate"):
        enforce_no_synthetic_gates(cfg_dir)
