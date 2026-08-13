import hashlib
import io
import json
import tarfile
from pathlib import Path

from ev.model_registry import ModelRegistry


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


def test_download_manifest_verify_extract_and_install(tmp_path, monkeypatch):
    """通过 Registry.install_all_from_manifest 下载、校验、安装。"""
    body = _model_archive()
    asset = {
        "key": "vad",
        "directory": "ev-fsmn-vad-zh-16k",
        "filename": "ev-fsmn-vad-zh-16k.tar.gz",
        "url": "https://example.test/vad-model.tar.gz",
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    manifest = {"version": "test", "assets": [asset]}

    # Mock 资源文件加载（importlib.resources.files 是 install_all_from_manifest 内的局部导入）
    monkeypatch.setattr(
        "importlib.resources.files",
        lambda pkg: _FakeResources(manifest),
    )

    events = []
    registry = ModelRegistry(
        models_root=tmp_path,
        emit=lambda kind, payload: events.append((kind, payload)),
    )
    # 注入 mock opener
    original_urlopen = __import__("urllib.request").request.urlopen

    class _MockOpener:
        def __init__(self, body_bytes):
            self.body = body_bytes

        def __enter__(self):
            return Response(self.body)

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _MockOpener(body),
    )

    registry.install_all_from_manifest()

    # 验证文件已安装
    assert (tmp_path / "ev-fsmn-vad-zh-16k" / "model.pt").read_bytes() == b"weights"
    assert any(kind == "download_progress" for kind, _ in events)

    # 验证已注册到 registry
    installed = registry.list_installed()
    assert "fsmn-vad" in installed
    assert installed["fsmn-vad"].local_path == str(tmp_path / "ev-fsmn-vad-zh-16k")


def test_safe_extract_rejects_parent_traversal(tmp_path):
    """_safe_extract 应拒绝包含 ../ 路径的压缩包。"""
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("../outside")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    try:
        ModelRegistry._safe_extract(archive, tmp_path / "target")
    except RuntimeError as exc:
        assert "不安全路径" in str(exc)
    else:
        raise AssertionError("unsafe archive should fail")


class _FakeResources:
    def __init__(self, manifest):
        self._manifest = manifest

    def joinpath(self, _name):
        return _FakeResourceFile(self._manifest)


class _FakeResourceFile:
    def __init__(self, manifest):
        self._manifest = manifest

    def read_text(self, encoding=None):
        return json.dumps(self._manifest)
