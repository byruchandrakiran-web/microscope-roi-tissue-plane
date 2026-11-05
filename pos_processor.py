"""Micro-Manager 2.x position-list (.pos) handling.

The parser is strict and schema-driven. It targets the Micro-Manager 2.0 "Property Map"
JSON layout (map -> StagePositions -> array -> [ {DefaultXYStage, DefaultZStage, Label,
DevicePositions -> array -> [ {Device, Position_um -> array} ... ]} ... ]).
Anything that does not match raises PosFormatError instead of producing a questionable file.
Existing fields are never removed; only Z Position_um values are changed (or a Z device entry
is added when the user has explicitly named the Z stage).
"""
import copy
import json

FORMAT_NAME = "Micro-Manager Property Map"


class PosFormatError(ValueError):
    pass


def parse_pos(text):
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PosFormatError(f"The file is not valid JSON ({e}). Only Micro-Manager 2.x JSON position "
                             "lists are supported.")
    if not isinstance(data, dict) or data.get("format") != FORMAT_NAME:
        raise PosFormatError(f'Missing the "format": "{FORMAT_NAME}" header. This does not look like a '
                             "Micro-Manager 2.x position list.")
    try:
        sps = data["map"]["StagePositions"]["array"]
    except (KeyError, TypeError):
        raise PosFormatError('Could not find map → StagePositions → array in the file.')
    if not isinstance(sps, list) or not sps:
        raise PosFormatError("The position list is empty.")
    for i, sp in enumerate(sps, 1):
        dp = sp.get("DevicePositions") if isinstance(sp, dict) else None
        if not isinstance(dp, dict) or not isinstance(dp.get("array"), list):
            raise PosFormatError(f"Position {i} has no DevicePositions array; structure not recognised.")
        for e in dp["array"]:
            if _dev_name(e) is None or not isinstance(_dev_pos(e), list):
                raise PosFormatError(f"Position {i} has a DevicePositions entry with an unrecognised layout.")
    return data


def _stage_positions(data):
    return data["map"]["StagePositions"]["array"]


def _scalar(sp, key):
    v = sp.get(key)
    return v.get("scalar") if isinstance(v, dict) else None


def _dev_name(entry):
    d = entry.get("Device") if isinstance(entry, dict) else None
    return d.get("scalar") if isinstance(d, dict) else None


def _dev_pos(entry):
    p = entry.get("Position_um") if isinstance(entry, dict) else None
    return p.get("array") if isinstance(p, dict) else None


def _entries(sp):
    return sp["DevicePositions"]["array"]


def _find_xy(sp, idx):
    default = _scalar(sp, "DefaultXYStage")
    two_d = [e for e in _entries(sp) if len(_dev_pos(e)) == 2]
    for e in two_d:
        if _dev_name(e) == default:
            return e
    if two_d:
        return two_d[0]
    raise PosFormatError(f"Position {idx} has no 2-axis (XY) stage entry.")


def _find_dev(sp, name):
    for e in _entries(sp):
        if _dev_name(e) == name:
            return e
    return None


def z_device_candidates(data):
    """Single-axis devices found in the file (plus DefaultZStage values)."""
    names = []
    for sp in _stage_positions(data):
        dz = _scalar(sp, "DefaultZStage")
        if dz and dz not in names:
            names.append(dz)
        for e in _entries(sp):
            if len(_dev_pos(e)) == 1 and _dev_name(e) not in names:
                names.append(_dev_name(e))
    return names


def extract_table(data, z_device=None):
    rows = []
    for i, sp in enumerate(_stage_positions(data), 1):
        xy = _find_xy(sp, i)
        z = _find_dev(sp, z_device) if z_device else None
        zv = None
        if z is not None and len(_dev_pos(z)) == 1:
            zv = float(_dev_pos(z)[0])
        rows.append({"Position": i, "Label": _scalar(sp, "Label") or f"Pos{i}", "XY Stage": _dev_name(xy),
                     "X": float(_dev_pos(xy)[0]), "Y": float(_dev_pos(xy)[1]), "Original Z": zv})
    return rows


def apply_z(data, z_values, z_device, allow_add=False):
    """Return (new_data, n_updated, n_added). Original object is untouched."""
    if not z_device or not str(z_device).strip():
        raise PosFormatError("A Z-stage device name is required.")
    sps = _stage_positions(data)
    if len(z_values) != len(sps):
        raise PosFormatError("Number of Z values does not match number of positions.")
    new = copy.deepcopy(data)
    updated = added = 0
    for i, (sp, z) in enumerate(zip(_stage_positions(new), z_values), 1):
        if not (z == z and abs(z) != float("inf")):
            raise PosFormatError(f"Z for position {i} is not a finite number.")
        if _dev_name(_find_xy(sp, i)) == z_device:
            raise PosFormatError(f'"{z_device}" is the XY stage, not a Z stage.')
        e = _find_dev(sp, z_device)
        if e is not None:
            if len(_dev_pos(e)) != 1:
                raise PosFormatError(f'Device "{z_device}" in position {i} is not a single-axis stage.')
            e["Position_um"]["array"] = [float(z)]
            updated += 1
        else:
            if not allow_add:
                raise PosFormatError(f'Position {i} has no "{z_device}" entry. Confirm the device name '
                                     "to allow adding it.")
            sp["DevicePositions"]["array"].append({
                "Device": {"type": "STRING", "scalar": z_device},
                "Position_um": {"type": "DOUBLE", "array": [float(z)]},
                "Properties": {"type": "PROPERTY_MAP", "array": [{}]}})
            added += 1
            if not _scalar(sp, "DefaultZStage"):
                sp["DefaultZStage"] = {"type": "STRING", "scalar": z_device}
    return new, updated, added


def dumps(data):
    return json.dumps(data, indent=2)


def _without_z(data, z_device):
    d = copy.deepcopy(data)
    for sp in _stage_positions(d):
        sp["DevicePositions"]["array"] = [e for e in _entries(sp) if _dev_name(e) != z_device]
        sp.pop("DefaultZStage", None)
    return d


def validate_roundtrip(original, text, expected_z, z_device, tol=1e-9):
    """Re-parse generated text; check Z values and that everything else is unchanged."""
    checks = {}
    try:
        again = parse_pos(text)
    except PosFormatError as e:
        return {"File re-parses as a Micro-Manager position list": False, "error": str(e)}
    checks["File re-parses as a Micro-Manager position list"] = True
    rows = extract_table(again, z_device)
    checks["Position count unchanged"] = len(rows) == len(_stage_positions(original))
    checks["Z values match calculated values"] = (
        len(rows) == len(expected_z) and
        all(r["Original Z"] is not None and abs(r["Original Z"] - z) <= tol for r, z in zip(rows, expected_z)))
    o = extract_table(original)
    checks["X, Y, labels and order unchanged"] = all(
        (a["X"], a["Y"], a["Label"]) == (b["X"], b["Y"], b["Label"]) for a, b in zip(o, rows))
    checks["All other devices/metadata unchanged"] = _without_z(original, z_device) == _without_z(again, z_device)
    return checks
