"""Unit tests for MediaLocator: URI dispatch, canonicalization, tenant isolation."""

import pytest

from cogniverse_core.common.media import (
    MediaCacheConfig,
    MediaConfig,
    MediaLocator,
)


@pytest.fixture
def cache_root(tmp_path):
    return tmp_path / "cache"


@pytest.fixture
def locator(cache_root):
    return MediaLocator(
        tenant_id="acme",
        config=MediaConfig(),
        cache_root=cache_root,
    )


class TestCanonicalUri:
    def test_passthrough_when_already_uri(self, locator):
        assert locator.to_canonical_uri("s3://bucket/key.mp4") == "s3://bucket/key.mp4"
        assert locator.to_canonical_uri("https://x.com/v.mp4") == "https://x.com/v.mp4"
        assert locator.to_canonical_uri("file:///abs/v.mp4") == "file:///abs/v.mp4"

    def test_default_file_scheme_for_absolute_path(self, locator):
        assert locator.to_canonical_uri("/abs/v.mp4") == "file:///abs/v.mp4"

    def test_default_file_scheme_for_relative_path_resolves(self, locator, tmp_path):
        result = locator.to_canonical_uri("data/v.mp4")
        assert result.startswith("file://")
        assert result.endswith("/data/v.mp4") or result.endswith("\\data\\v.mp4")

    def test_uri_prefix_with_relative_path(self, cache_root):
        loc = MediaLocator(
            tenant_id="t",
            config=MediaConfig(uri_prefix="s3://corpus/"),
            cache_root=cache_root,
        )
        assert loc.to_canonical_uri("subdir/v.mp4") == "s3://corpus/subdir/v.mp4"
        assert loc.to_canonical_uri("v.mp4") == "s3://corpus/v.mp4"

    def test_uri_prefix_strips_trailing_slash(self, cache_root):
        loc = MediaLocator(
            tenant_id="t",
            config=MediaConfig(uri_prefix="s3://corpus"),
            cache_root=cache_root,
        )
        assert loc.to_canonical_uri("v.mp4") == "s3://corpus/v.mp4"

    def test_uri_prefix_absolute_path_uses_basename(self, cache_root):
        loc = MediaLocator(
            tenant_id="t",
            config=MediaConfig(uri_prefix="s3://corpus/"),
            cache_root=cache_root,
        )
        assert loc.to_canonical_uri("/abs/path/v.mp4") == "s3://corpus/v.mp4"


class TestLocalizeFile:
    def test_file_scheme_returns_path(self, locator, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"video")

        assert locator.localize(f"file://{f}") == f

    def test_bare_path_returns_path(self, locator, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"video")

        assert locator.localize(str(f)) == f

    def test_missing_file_raises(self, locator, tmp_path):
        with pytest.raises(FileNotFoundError):
            locator.localize(f"file://{tmp_path}/missing.mp4")

    def test_no_copy_made(self, locator, cache_root, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"video")

        result = locator.localize(f"file://{f}")

        assert result == f
        assert not list((cache_root / "acme" / "media").rglob("v.mp4"))


class TestLocalizePvc:
    def test_pvc_translates_to_mount_root(self, tmp_path, cache_root):
        mount = tmp_path / "mnt"
        (mount / "media-corpus" / "videos").mkdir(parents=True)
        f = mount / "media-corpus" / "videos" / "v.mp4"
        f.write_bytes(b"video")

        loc = MediaLocator(
            tenant_id="t",
            config=MediaConfig(pvc_mount_root=str(mount)),
            cache_root=cache_root,
        )

        assert loc.localize("pvc://media-corpus/videos/v.mp4") == f

    def test_pvc_missing_volume_raises(self, locator):
        with pytest.raises(ValueError, match="missing volume"):
            locator.localize("pvc:///path/v.mp4")

    def test_pvc_missing_file_raises(self, tmp_path, cache_root):
        loc = MediaLocator(
            tenant_id="t",
            config=MediaConfig(pvc_mount_root=str(tmp_path / "mnt")),
            cache_root=cache_root,
        )
        with pytest.raises(FileNotFoundError):
            loc.localize("pvc://corpus/missing.mp4")


class TestUnsupportedScheme:
    def test_unknown_scheme_raises(self, locator):
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            locator.localize("ftp://example.com/v.mp4")


class TestStat:
    def test_stat_local(self, locator, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x" * 42)

        stat = locator.stat(str(f))
        assert stat.size == 42
        assert stat.last_modified is not None

    def test_exists_local_true(self, locator, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"x")
        assert locator.exists(str(f)) is True

    def test_exists_local_false(self, locator, tmp_path):
        assert locator.exists(str(tmp_path / "missing.mp4")) is False


class TestList:
    def test_list_local_yields_video_uris(self, locator, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.mkv").write_bytes(b"")
        (tmp_path / "ignore.txt").write_bytes(b"")

        results = list(locator.list(f"file://{tmp_path}"))

        assert any(r.endswith("/a.mp4") for r in results)
        assert any(r.endswith("/b.mkv") for r in results)
        assert not any(r.endswith(".txt") for r in results)
        assert all(r.startswith("file://") for r in results)

    def test_list_pvc_yields_pvc_uris(self, tmp_path, cache_root):
        mount = tmp_path / "mnt"
        (mount / "corpus" / "vids").mkdir(parents=True)
        (mount / "corpus" / "vids" / "a.mp4").write_bytes(b"")

        loc = MediaLocator(
            tenant_id="t",
            config=MediaConfig(pvc_mount_root=str(mount)),
            cache_root=cache_root,
        )

        results = list(loc.list("pvc://corpus/vids"))

        assert results == ["pvc://corpus/vids/a.mp4"]

    def test_list_extension_filter(self, locator, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"")
        (tmp_path / "b.mkv").write_bytes(b"")

        results = list(locator.list(f"file://{tmp_path}", extensions=(".mkv",)))

        assert len(results) == 1
        assert results[0].endswith("/b.mkv")


class TestTenantIsolation:
    def test_separate_tenants_separate_cache_dirs(self, cache_root):
        loc_a = MediaLocator(
            tenant_id="acme", config=MediaConfig(), cache_root=cache_root
        )
        loc_b = MediaLocator(
            tenant_id="other", config=MediaConfig(), cache_root=cache_root
        )

        assert loc_a.cache.base_dir != loc_b.cache.base_dir
        assert "acme" in str(loc_a.cache.base_dir)
        assert "other" in str(loc_b.cache.base_dir)

    def test_org_tenant_format(self, cache_root):
        loc = MediaLocator(
            tenant_id="acme:prod", config=MediaConfig(), cache_root=cache_root
        )
        path_str = str(loc.cache.base_dir)
        assert "acme" in path_str
        assert "prod" in path_str


class TestOpen:
    def test_open_local_returns_file_handle(self, locator, tmp_path):
        f = tmp_path / "v.mp4"
        f.write_bytes(b"hello")

        with locator.open(str(f)) as fh:
            assert fh.read() == b"hello"


class TestConfigFromDict:
    def test_loads_minimal_config(self):
        cfg = MediaConfig.from_dict({})
        assert cfg.default_uri_scheme == "file"
        assert cfg.uri_prefix == ""

    def test_loads_full_config(self):
        cfg = MediaConfig.from_dict(
            {
                "default_uri_scheme": "s3",
                "uri_prefix": "s3://corpus/",
                "pvc_mount_root": "/data",
                "cache": {"max_bytes_gb": 10, "ttl_days": 1},
                "backends": {
                    "s3": {"endpoint_url": "http://minio:9000", "anon": True},
                    "http": {"timeout_s": 30},
                },
            }
        )
        assert cfg.default_uri_scheme == "s3"
        assert cfg.uri_prefix == "s3://corpus/"
        assert cfg.pvc_mount_root == "/data"
        assert cfg.cache.max_bytes_gb == 10
        assert cfg.s3.endpoint_url == "http://minio:9000"
        assert cfg.s3.anon is True
        assert cfg.http.timeout_s == 30


class TestCacheBaseDirSelection:
    def test_explicit_cache_root_overrides_config(self, tmp_path):
        cfg = MediaConfig(cache=MediaCacheConfig(base_dir="/should/not/be/used"))
        loc = MediaLocator(tenant_id="t", config=cfg, cache_root=tmp_path / "explicit")
        assert str(tmp_path / "explicit") in str(loc.cache.base_dir)

    def test_config_base_dir_used_when_no_explicit(self, tmp_path):
        configured = tmp_path / "configured"
        cfg = MediaConfig(cache=MediaCacheConfig(base_dir=str(configured)))
        loc = MediaLocator(tenant_id="t", config=cfg)
        assert str(configured) in str(loc.cache.base_dir)
