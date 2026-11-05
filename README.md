# Microscope ROI & 3D Tissue Plane Generator

A Streamlit application integrating the supplied microscope ROI workflow with the supplied 3D tissue plane calculator.

## Coordinate-system integrity

The microscope coordinate pipeline is deliberately preserved:

1. Three image calibration points establish the original local coordinate system: P1=(0,0), P2=(0,30 mm), P3=(40,0 mm).
2. The existing calibration matrix converts image pixels into local sample coordinates.
3. The existing origin shift is applied exactly once:
   - `X_new = X_old - new_origin_x`
   - `Y_new = Y_old - new_origin_y`
4. ROI centres remain in transformed sample coordinates (`X_new_mm`, `Y_new_mm`).
5. Stage coordinates use the original axis signs and mm-to-µm conversion:
   - `X_stage_um = X_new_mm * 1000 * x_sign`
   - `Y_stage_um = Y_new_mm * 1000 * y_sign`
6. Automatically selected Z1–Z3 reference points use those same final stage X/Y coordinates. The user enters only measured Z-stage values in µm.
7. The original `calculations.compute_plane()` and `Plane.z_at()` calculate and predict Z in the same stage coordinate system.

No second calibration, extra origin shift, or second axis inversion is introduced. The active origin defaults follow the supplied master prompt: X=37.543 mm and Y=-16.511 mm.

## Features

- Overview image upload and four-point perspective correction.
- HSV purple tissue segmentation using the original threshold/contour logic.
- Three-point pixel-to-sample calibration and original origin shift.
- ROI tile generation with FOV, overlap, boundary clearance and serpentine ordering.
- XY stage coordinate generation with configurable axis inversion.
- Deterministic maximum-area triangle selection on the original tissue contour for Z references.
- Manual measured Z input, original cross-product plane calculation and Z prediction.
- Single-point and CSV-based Z calculator.
- XYZ Micro-Manager Property Map generation using the original POS processor and round-trip validation.
- Existing POS-file processing mode.
- Original Plotly 3D plane visualization.
- Dependency-aware invalidation of plane and generated outputs.
- Reset for a new tissue sample.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run locally

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Tests

```bash
pytest -q
```

The original plane-calculation and POS-processor tests are retained in `tests/`.

## Micro-Manager POS compatibility

The POS processor strictly supports Micro-Manager 2.x `Micro-Manager Property Map` JSON files. Unsupported or ambiguous formats are rejected. If a lab template is uploaded for the integrated ROI workflow, its position count must match the generated ROI count and the configured XY device must be found in each position. Existing per-position labels, grid fields, device metadata and unrelated fields are retained while the XY values are replaced by the ROI stage coordinates.

## Important operational warning

The generated `.pos` is a position list, not a command that directly controls the microscope. Before real acquisition, load it in Micro-Manager and independently verify position labels, device names, XY/Z values, ordering and safe stage limits. A plane fitted through three points is a local approximation of the tissue surface; positions outside the reference bounding box are extrapolations.


## Session persistence across workflow pages

The uploaded overview image is copied into durable Streamlit session state as bytes. It is not stored only in the page-scoped file-uploader widget state. Navigating to Tissue Boundary Detection, Coordinate Calibration, ROI Generation and later pages therefore does not discard the uploaded image when the uploader widget is temporarily not rendered.
