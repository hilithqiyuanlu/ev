from ev import __version__
from ev.config import load_settings


def test_version():
    assert __version__


def test_default_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("EV_DATA_DIR", str(tmp_path / "data"))
    s = load_settings()
    assert s.audio.sample_rate == 16000
    assert s.audio.channels == 1
    assert s.data_dir == tmp_path / "data"
    s.ensure_dirs()
    assert s.models_dir.is_dir()
    assert s.archive_dir.is_dir()
