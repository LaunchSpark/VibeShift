from scripts import install_pocketbase


def test_windows_amd64_asset_name(monkeypatch):
    monkeypatch.setattr(install_pocketbase.platform, "system", lambda: "Windows")
    monkeypatch.setattr(install_pocketbase.platform, "machine", lambda: "AMD64")
    assert install_pocketbase.asset_name("0.40.5") == "pocketbase_0.40.5_windows_amd64.zip"


def test_unsupported_architecture_is_reported(monkeypatch):
    monkeypatch.setattr(install_pocketbase.platform, "system", lambda: "Windows")
    monkeypatch.setattr(install_pocketbase.platform, "machine", lambda: "ARMv7")
    try:
        install_pocketbase.asset_name("0.40.5")
    except SystemExit as exc:
        assert "unsupported platform: Windows ARMv7" in str(exc)
    else:
        raise AssertionError("expected unsupported architecture to fail")
