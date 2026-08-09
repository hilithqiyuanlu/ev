import hashlib
import io
import tarfile

import ev.model_download as download_module
from ev.config import ModelSettings
from ev.model_download import ModelAsset, ModelDownloader
from ev.models import verify_models


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _model_archive() -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, content in (("config.yaml", b"model: vad"), ("model.pt", b"weights")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def test_download_sha_verify_extract_and_atomic_install(tmp_path, monkeypatch):
    body = _model_archive()
    asset = ModelAsset(
        key="vad",
        directory="vad-model",
        filename="vad-model.tar.gz",
        url="https://example.test/vad-model.tar.gz",
        size=len(body),
        sha256=hashlib.sha256(body).hexdigest(),
    )
    settings = ModelSettings(
        root=tmp_path,
        vad="vad-model",
        asr_streaming="stream",
        asr_final="final",
        speaker="speaker",
    )
    monkeypatch.setattr(download_module, "load_manifest", lambda: ("test", (asset,)))
    events = []
    downloader = ModelDownloader(
        settings,
        lambda kind, payload: events.append((kind, payload)),
        opener=lambda request, timeout: Response(body),
    )
    downloader.download_all()
    check = next(item for item in verify_models(settings) if item.key == "vad")
    assert check.ok
    assert (tmp_path / "vad-model" / "model.pt").read_bytes() == b"weights"
    assert any(kind == "download_progress" for kind, _ in events)


def test_safe_extract_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    try:
        ModelDownloader._safe_extract(archive, tmp_path / "target")
    except RuntimeError as exc:
        assert "不安全路径" in str(exc)
    else:
        raise AssertionError("unsafe archive should fail")
