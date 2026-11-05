import json, pytest
import pos_processor as pp
from tests.make_fixture import build


def test_extract():
    rows = pp.extract_table(pp.parse_pos(json.dumps(build())), "ZStage")
    assert [(r["X"], r["Y"]) for r in rows] == [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)]


def test_apply_roundtrip_preserves_everything():
    d = build()
    data = pp.parse_pos(json.dumps(d))
    new, upd, add = pp.apply_z(data, [1.5, 2.5, 3.5], "ZStage")
    assert (upd, add) == (3, 0)
    checks = pp.validate_roundtrip(data, pp.dumps(new), [1.5, 2.5, 3.5], "ZStage")
    assert all(checks.values())
    assert data == d


def test_missing_z_requires_confirmation():
    data = pp.parse_pos(json.dumps(build(with_z=False)))
    with pytest.raises(pp.PosFormatError):
        pp.apply_z(data, [1, 2, 3], "ZStage")
    assert pp.apply_z(data, [1, 2, 3], "ZStage", allow_add=True)[2] == 3


def test_unsupported_files():
    for bad in ["not json", "{}", json.dumps({"format": pp.FORMAT_NAME, "map": {}})]:
        with pytest.raises(pp.PosFormatError):
            pp.parse_pos(bad)


def test_xy_stage_rejected_as_z():
    data = pp.parse_pos(json.dumps(build()))
    with pytest.raises(pp.PosFormatError):
        pp.apply_z(data, [1, 2, 3], "XYStage")
