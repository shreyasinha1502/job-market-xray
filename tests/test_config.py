import shutil

import pytest

from xray.config import (
    CONFIG_DIR,
    ConfigError,
    enforce_no_synthetic_gates,
    load_panel,
    load_regions,
    load_sources,
)


def test_real_sources_config_validates():
    cfg = load_sources()
    assert cfg.providers and cfg.countries and cfg.roles_to_track
    panel = load_panel(cfg)
    assert panel and all(b.source in cfg.providers for b in panel)
    assert set(load_regions(cfg)) == set(cfg.countries)


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
