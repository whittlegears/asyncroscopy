import numpy as np

from asyncroscopy.data import tiled_client


class FakeNode:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeClient(dict):
    pass


def test_open_client_uses_env_api_key(monkeypatch):
    seen = {}
    monkeypatch.setattr(tiled_client, "from_uri", lambda uri, api_key=None: seen.update(uri=uri, api_key=api_key))
    monkeypatch.setenv(tiled_client.TILED_API_KEY_ENV, "from-env")

    tiled_client.open_client("http://tiled:9091")
    assert seen == {"uri": "http://tiled:9091", "api_key": "from-env"}

    tiled_client.open_client("http://tiled:9091", "explicit")
    assert seen["api_key"] == "explicit"


def test_default_api_key_falls_back(monkeypatch):
    monkeypatch.delenv(tiled_client.TILED_API_KEY_ENV, raising=False)
    assert tiled_client.default_api_key() == tiled_client.DEFAULT_TILED_API_KEY


def test_key_timestamp_parses_writer_naming():
    stamp = tiled_client.key_timestamp("stem_image_HAADF_BF-S_20260904T103906954339.h5")
    assert stamp is not None and (stamp.year, stamp.month, stamp.day, stamp.microsecond) == (2026, 9, 4, 954339)
    assert tiled_client.key_timestamp("stem_image_20260904T103906954339_HAADF.tiff") is not None
    assert tiled_client.key_timestamp("notes.txt") is None


def test_list_acquisitions_sorts_newest_first_and_filters():
    client = FakeClient(
        {
            "stem_image_HAADF_20260904T100000000000.h5": FakeNode({"stage_x": 1.0}),
            "spectrum_eds_20260904T110000000000.h5": FakeNode({"elements": np.array(["Au"])}),
            "stem_image_HAADF_20260904T120000000000.h5": FakeNode({"stage_x": 2.0}),
            "unstamped.h5": FakeNode({}),
        }
    )

    listed = tiled_client.list_acquisitions(client)
    assert [item["key"][:10] for item in listed] == ["stem_image", "spectrum_e", "stem_image", "unstamped."]
    assert listed[0]["metadata"] == {"stage_x": 2.0}
    assert listed[1]["metadata"] == {"elements": ["Au"]}
    assert listed[-1]["timestamp"] is None

    images = tiled_client.list_acquisitions(client, acquisition_type="stem_image", limit=1)
    assert [item["key"] for item in images] == ["stem_image_HAADF_20260904T120000000000.h5"]

    recent = tiled_client.list_acquisitions(client, since="2026-09-04T10:30:00", with_metadata=False)
    assert [item["key"][:8] for item in recent] == ["stem_ima", "spectrum"]
    assert "metadata" not in recent[0]
