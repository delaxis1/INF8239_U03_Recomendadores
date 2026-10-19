"""Protección de la procedencia: ZIP pinneado, restauración y traversal."""

import io, zipfile
import pytest
from scripts import run_exercise05 as exp


def setup_local(monkeypatch, tmp_path, member="ml-latest-small/ratings.csv"):
    monkeypatch.setattr(exp, "ROOT", tmp_path)
    monkeypatch.setattr(exp, "MOVIELENS_DIR", tmp_path / "data/raw/ml-latest-small")
    (tmp_path / "docs").mkdir()
    (tmp_path / "data/raw").mkdir(parents=True)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(member, "original")
    (tmp_path / "data/raw/ml-latest-small.zip").write_bytes(buf.getvalue())


def test_restore_csv_from_verified_zip(monkeypatch, tmp_path):
    setup_local(monkeypatch, tmp_path)
    exp.download()
    f = tmp_path / "data/raw/ml-latest-small/ratings.csv"
    f.write_text("editado")
    exp.download()
    assert f.read_text() == "original"


def test_changed_zip_hash_is_rejected(monkeypatch, tmp_path):
    setup_local(monkeypatch, tmp_path)
    exp.download()
    (tmp_path / "data/raw/ml-latest-small.zip").write_bytes(b"cambio")
    with pytest.raises(ValueError, match="cambió"):
        exp.download()


def test_unsafe_zip_path_is_rejected(monkeypatch, tmp_path):
    setup_local(monkeypatch, tmp_path, "../../outside.txt")
    with pytest.raises(ValueError, match="ZIP inseguro"):
        exp.download()
    assert not (tmp_path / "outside.txt").exists()
