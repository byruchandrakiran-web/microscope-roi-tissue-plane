"""Builds a TEST FIXTURE in the Micro-Manager 2.x position-list layout (not a verified microscope config)."""
import json, sys


def build(with_z=True):
    def pos(i, x, y):
        devs = [{"Device": {"type": "STRING", "scalar": "XYStage"},
                 "Position_um": {"type": "DOUBLE", "array": [x, y]},
                 "Properties": {"type": "PROPERTY_MAP", "array": [{}]}},
                {"Device": {"type": "STRING", "scalar": "OtherStage"},
                 "Position_um": {"type": "DOUBLE", "array": [7.0]},
                 "Properties": {"type": "PROPERTY_MAP", "array": [{}]}}]
        if with_z:
            devs.append({"Device": {"type": "STRING", "scalar": "ZStage"},
                         "Position_um": {"type": "DOUBLE", "array": [0.0]},
                         "Properties": {"type": "PROPERTY_MAP", "array": [{}]}})
        sp = {"DefaultXYStage": {"type": "STRING", "scalar": "XYStage"},
              "DevicePositions": {"type": "PROPERTY_MAP", "array": devs},
              "GridCol": {"type": "INTEGER", "scalar": 0}, "GridRow": {"type": "INTEGER", "scalar": 0},
              "Label": {"type": "STRING", "scalar": f"Pos{i}"},
              "Properties": {"type": "PROPERTY_MAP", "array": [{}]}}
        if with_z:
            sp["DefaultZStage"] = {"type": "STRING", "scalar": "ZStage"}
        return sp
    return {"format": "Micro-Manager Property Map", "major_version": 2, "minor_version": 0,
            "map": {"StagePositions": {"type": "PROPERTY_MAP",
                                       "array": [pos(1, 100.0, 200.0), pos(2, 300.0, 400.0), pos(3, 500.0, 600.0)]}}}


if __name__ == "__main__":
    json.dump(build(), open(sys.argv[1], "w"), indent=2)
