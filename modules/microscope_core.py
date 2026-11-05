import io, json, math
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ImportError:
    streamlit_image_coordinates = None
try:
    from shapely.geometry import Polygon, box
except ImportError:
    Polygon = None
    box = None
def read_uploaded_image(uploaded):

    data = np.frombuffer(uploaded.getvalue(), dtype=np.uint8)

    img = cv2.imdecode(data, cv2.IMREAD_COLOR)

    return img

def cv_rgb(img_bgr):

    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def jpg_bytes(img_bgr, quality=95):

    ok, enc = cv2.imencode(

        ".jpg",

        img_bgr,

        [cv2.IMWRITE_JPEG_QUALITY, quality],

    )

    return enc.tobytes() if ok else b""

def csv_bytes(df):

    return df.to_csv(index=False).encode("utf-8")

def overlay_points(img_bgr, points, labels=None):

    out = img_bgr.copy()

    labels = labels or [f"P{i+1}" for i in range(len(points))]

    for i, p in enumerate(points):

        x, y = int(round(p[0])), int(round(p[1]))

        cv2.circle(out, (x, y), 10, (0, 0, 255), 3)

        cv2.putText(

            out,

            labels[i],

            (x + 15, y - 15),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (0, 0, 255),

            2,

        )

    return out

def perspective_correct(img, points):

    # Exact geometric logic from image_processing(1).py:

    # P1 top-left, P2 top-right, P3 bottom-right, P4 bottom-left.

    p1, p2, p3, p4 = points

    src = np.float32([p1, p2, p3, p4])

    width_top = np.linalg.norm(np.array(p2) - np.array(p1))

    width_bottom = np.linalg.norm(np.array(p3) - np.array(p4))

    height_left = np.linalg.norm(np.array(p4) - np.array(p1))

    height_right = np.linalg.norm(np.array(p3) - np.array(p2))

    output_width = int(max(width_top, width_bottom))

    output_height = int(max(height_left, height_right))

    if output_width < 2 or output_height < 2:

        raise ValueError("Selected perspective points produce an invalid output size.")

    dst = np.float32([

        [0, 0],

        [output_width - 1, 0],

        [output_width - 1, output_height - 1],

        [0, output_height - 1],

    ])

    matrix = cv2.getPerspectiveTransform(src, dst)

    corrected = cv2.warpPerspective(

        img,

        matrix,

        (output_width, output_height),

    )

    return corrected, matrix, output_width, output_height

def detect_purple_tissue(img_bgr, lower, upper):

    # Exact threshold/contour logic from boundary(1).py.

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(

        hsv,

        np.array(lower, dtype=np.uint8),

        np.array(upper, dtype=np.uint8),

    )

    contours, _ = cv2.findContours(

        mask,

        cv2.RETR_EXTERNAL,

        cv2.CHAIN_APPROX_NONE,

    )

    if len(contours) == 0:

        return mask, None

    tissue_contour = max(contours, key=cv2.contourArea)

    return mask, tissue_contour

def calibrate_pixel_to_sample(points, new_origin_x, new_origin_y):

    # Exact calibration logic from boundary(1).py.

    origin_pixel = np.array(points[0], dtype=np.float64)

    y30_pixel = np.array(points[1], dtype=np.float64)

    x40_pixel = np.array(points[2], dtype=np.float64)

    vector_y = y30_pixel - origin_pixel

    vector_x = x40_pixel - origin_pixel

    calibration_matrix = np.column_stack(

        (

            vector_x / 40.0,

            vector_y / 30.0,

        )

    )

    determinant = np.linalg.det(calibration_matrix)

    if abs(determinant) < 1e-10:

        raise ValueError(

            "The 3 calibration points are invalid. "

            "P1, P2 and P3 must not lie on the same line."

        )

    inverse_matrix = np.linalg.inv(calibration_matrix)

    return origin_pixel, inverse_matrix, calibration_matrix, determinant

def contour_to_new_coordinates(

    tissue_contour,

    origin_pixel,

    inverse_matrix,

    new_origin_x,

    new_origin_y,

):

    # Exact boundary conversion logic from boundary(1).py.

    coords = []

    for point in tissue_contour:

        pixel_point = np.array(

            [

                float(point[0][0]),

                float(point[0][1]),

            ],

            dtype=np.float64,

        )

        pixel_difference = pixel_point - origin_pixel

        old_coordinates = inverse_matrix @ pixel_difference

        X_old = old_coordinates[0]

        Y_old = old_coordinates[1]

        X_new = X_old - new_origin_x

        Y_new = Y_old - new_origin_y

        coords.append((X_new, Y_new))

    return coords

def make_boundary_image(original, contour, calibration_points):

    out = original.copy()

    cv2.drawContours(

        out,

        [contour],

        -1,

        (0, 0, 255),

        2,

    )

    labels = [

        "OLD ORIGIN (0,0)",

        "(0,30 mm)",

        "(40,0 mm)",

    ]

    for i, p in enumerate(calibration_points):

        x, y = int(p[0]), int(p[1])

        if i == 0:

            color = (255, 0, 0)  # BGR blue

        else:

            color = (0, 255, 0)

        cv2.circle(out, (x, y), 9, color, -1)

        cv2.putText(

            out,

            labels[i],

            (x + 12, y - 12),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            color,

            2,

        )

    return out

def make_tile_preview(

    boundary_xy,

    tile_centers,

    fov_x_mm,

    fov_y_mm,

    clearance_um=0.0,

):

    # Draw original tissue boundary and its outward-expanded scan boundary.

    poly = np.asarray(boundary_xy, dtype=np.float64)

    tissue_poly = Polygon(poly) if Polygon is not None else None

    scan_poly = (
        tissue_poly.buffer(float(clearance_um) / 1000.0)
        if tissue_poly is not None and float(clearance_um) > 0
        else tissue_poly
    )

    min_x, min_y = poly.min(axis=0)

    max_x, max_y = poly.max(axis=0)

    if scan_poly is not None and not scan_poly.is_empty:
        sx0, sy0, sx1, sy1 = scan_poly.bounds
        min_x, min_y = min(min_x, sx0), min(min_y, sy0)
        max_x, max_y = max(max_x, sx1), max(max_y, sy1)

    pad_x = fov_x_mm

    pad_y = fov_y_mm

    width = max_x - min_x + 2 * pad_x

    height = max_y - min_y + 2 * pad_y

    W = 1000

    H = max(700, int(W * height / max(width, 1e-9)))

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    def to_px(x, y):

        px = int((x - (min_x - pad_x)) / width * (W - 1))

        py = int((max_y + pad_y - y) / height * (H - 1))

        return px, py

    # Blue = outward-expanded scan boundary; red = original tissue boundary.
    if scan_poly is not None and not scan_poly.is_empty:
        polygons = list(scan_poly.geoms) if scan_poly.geom_type == "MultiPolygon" else [scan_poly]
        for expanded in polygons:
            expanded_pts = np.array(
                [to_px(x, y) for x, y in list(expanded.exterior.coords)],
                dtype=np.int32,
            )
            if len(expanded_pts) >= 3:
                cv2.polylines(canvas, [expanded_pts], True, (255, 0, 0), 2)

    pts = np.array([to_px(x, y) for x, y in poly], dtype=np.int32)

    if len(pts) >= 3:

        cv2.polylines(canvas, [pts], True, (0, 0, 255), 3)

    # Tile rectangles

    for idx, (cx, cy) in enumerate(tile_centers):

        x1, y1 = to_px(cx - fov_x_mm / 2, cy + fov_y_mm / 2)

        x2, y2 = to_px(cx + fov_x_mm / 2, cy - fov_y_mm / 2)

        cv2.rectangle(

            canvas,

            (x1, y1),

            (x2, y2),

            (0, 150, 0),

            1,

        )

        tx, ty = to_px(cx, cy)

        cv2.circle(canvas, (tx, ty), 3, (255, 0, 0), -1)

        if idx < 200:

            cv2.putText(

                canvas,

                str(idx + 1),

                (tx + 4, ty - 4),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.35,

                (0, 0, 0),

                1,

            )

    return canvas

def generate_roi_tiles(

    boundary_xy,

    fov_x_um,

    fov_y_um,

    overlap_percent,

    min_coverage_percent=0.0,

    clearance_um=0.0,

):

    """

    Generate tile centers in the calibrated NEW SAMPLE coordinate system.

    FOV is converted to mm because boundary coordinates are X_new_mm/Y_new_mm.

    A tile is retained when it intersects the tissue polygon expanded by

    the requested clearance. Optional minimum tissue coverage can be applied.

    Clearance is used only for selecting scan tiles. The original

    X_new_mm/Y_new_mm boundary coordinates are not modified.

    """

    if Polygon is None:

        raise RuntimeError("Shapely is required for ROI tile generation.")

    tissue = Polygon(boundary_xy)

    if not tissue.is_valid:

        tissue = tissue.buffer(0)

    if tissue.is_empty:

        raise ValueError("The tissue polygon is empty after validation.")

    clearance_mm = float(clearance_um) / 1000.0

    scan_region = tissue.buffer(clearance_mm) if clearance_mm > 0 else tissue

    if scan_region.is_empty:

        raise ValueError("The tissue + clearance region is empty.")

    fov_x_mm = fov_x_um / 1000.0

    fov_y_mm = fov_y_um / 1000.0

    step_x_mm = fov_x_mm * (1.0 - overlap_percent / 100.0)

    step_y_mm = fov_y_mm * (1.0 - overlap_percent / 100.0)

    if step_x_mm <= 0 or step_y_mm <= 0:

        raise ValueError("Overlap must be less than 100%.")

    minx, miny, maxx, maxy = scan_region.bounds

    # Centers are placed so the first tile is half an FOV inside

    # the bounding box of the tissue.

    start_x = minx + fov_x_mm / 2.0

    start_y = miny + fov_y_mm / 2.0

    centers = []

    rows = []

    y = start_y

    row_index = 0

    while y <= maxy + fov_y_mm / 2.0 + 1e-12:

        row_candidates = []

        x = start_x

        col_index = 0

        while x <= maxx + fov_x_mm / 2.0 + 1e-12:

            tile = box(

                x - fov_x_mm / 2.0,

                y - fov_y_mm / 2.0,

                x + fov_x_mm / 2.0,

                y + fov_y_mm / 2.0,

            )

            if tile.intersects(scan_region):

                # Coverage is reported against the ORIGINAL tissue, not

                # against the clearance region. This keeps the metric

                # physically meaningful.

                intersection_area = tile.intersection(tissue).area

                coverage = (

                    100.0 * intersection_area / tile.area

                    if tile.area > 0

                    else 0.0

                )

                if coverage >= min_coverage_percent:

                    row_candidates.append(

                        {

                            "row": row_index,

                            "col": col_index,

                            "X_sample_mm": x,

                            "Y_sample_mm": y,

                            "tissue_coverage_percent": coverage,

                        }

                    )

            x += step_x_mm

            col_index += 1

        # Serpentine order helps reduce unnecessary stage travel.

        if row_index % 2 == 1:

            row_candidates.reverse()

        rows.extend(row_candidates)

        y += step_y_mm

        row_index += 1

    return pd.DataFrame(rows), fov_x_mm, fov_y_mm, step_x_mm, step_y_mm

def affine_from_3_points(src_xy, dst_xy):

    """

    3-point affine calibration:

      source = NEW SAMPLE coordinates in mm

      target = Micro-Manager XY stage coordinates in um

    Returns 2x3 affine matrix.

    """

    src = np.asarray(src_xy, dtype=np.float64)

    dst = np.asarray(dst_xy, dtype=np.float64)

    if src.shape != (3, 2) or dst.shape != (3, 2):

        raise ValueError("Exactly 3 source and 3 stage points are required.")

    A = np.column_stack(

        [

            src[:, 0],

            src[:, 1],

            np.ones(3),

        ]

    )

    det = np.linalg.det(A)

    if abs(det) < 1e-12:

        raise ValueError(

            "Stage calibration source points are collinear."

        )

    params_x = np.linalg.solve(A, dst[:, 0])

    params_y = np.linalg.solve(A, dst[:, 1])

    M = np.vstack([params_x, params_y])

    return M

def apply_affine(M, xy):

    xy = np.asarray(xy, dtype=np.float64)

    A = np.column_stack([xy[:, 0], xy[:, 1], np.ones(len(xy))])

    return A @ M.T

def draw_stage_preview(df):

    if len(df) == 0:

        return None

    xs = df["X_stage_um"].to_numpy(float)

    ys = df["Y_stage_um"].to_numpy(float)

    minx, maxx = xs.min(), xs.max()

    miny, maxy = ys.min(), ys.max()

    pad_x = max((maxx - minx) * 0.08, 100.0)

    pad_y = max((maxy - miny) * 0.08, 100.0)

    W = 1000

    H = 700

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)

    def px(x, y):

        xx = int((x - (minx - pad_x)) / (maxx - minx + 2 * pad_x) * (W - 1))

        yy = int((maxy + pad_y - y) / (maxy - miny + 2 * pad_y) * (H - 1))

        return xx, yy

    pts = [px(x, y) for x, y in zip(xs, ys)]

    for i in range(1, len(pts)):

        cv2.line(canvas, pts[i - 1], pts[i], (0, 130, 0), 2)

    for i, p in enumerate(pts):

        cv2.circle(canvas, p, 5, (255, 0, 0), -1)

        if i < 250:

            cv2.putText(

                canvas,

                str(i + 1),

                (p[0] + 5, p[1] - 5),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.4,

                (0, 0, 0),

                1,

            )

    return canvas

def click_selector(image_bgr, key, expected_count, instructions):

    import streamlit as st

    if streamlit_image_coordinates is None:

        st.error(

            "Install streamlit-image-coordinates first. "

            "See requirements.txt."

        )

        return []

    st.info(instructions)

    current = st.session_state.get(key, [])

    if len(current) < expected_count:

        clicked = streamlit_image_coordinates(

            cv_rgb(image_bgr),

            key=f"{key}_click_{len(current)}",

        )

        if clicked is not None:

            p = (int(clicked["x"]), int(clicked["y"]))

            # The component key changes after a successful click, so

            # the same event is not repeatedly appended.

            current = current + [p]

            st.session_state[key] = current

            st.rerun()

    return st.session_state.get(key, [])

def default_pos_structure():

    # Matches the observed Micro-Manager Property Map format in

    # the supplied .pos file.

    return {

        "encoding": "UTF-8",

        "format": "Micro-Manager Property Map",

        "major_version": 2,

        "minor_version": 0,

        "map": {

            "StagePositions": {

                "type": "PROPERTY_MAP",

                "array": [],

            }

        },

    }

def generate_pos(
    stage_xy_um,
    device_name,
    labels_prefix="ROI",
    template=None,
    z_stage_device=None,
    z_values=None,
):
    """Create a Micro-Manager Property Map PositionList (XY-only or XYZ)."""
    data = default_pos_structure() if template is None else deepcopy(template)
    if z_values is not None and len(z_values) != len(stage_xy_um):
        raise ValueError("The number of Z values must match the number of XY positions.")
    z_stage_name = (z_stage_device or "").strip() if z_values is not None else ""
    if z_values is not None and not z_stage_name:
        raise ValueError("Enter the exact Micro-Manager Z stage device name.")

    stage_positions = []
    for i, item in enumerate(stage_xy_um):
        x = float(item["X_stage_um"])
        y = float(item["Y_stage_um"])
        row = int(item.get("row", 0))
        col = int(item.get("col", i))
        device_positions = [{
            "Device": {"type": "STRING", "scalar": device_name},
            "Position_um": {"type": "DOUBLE", "array": [x, y]},
        }]
        if z_values is not None:
            device_positions.append({
                "Device": {"type": "STRING", "scalar": z_stage_name},
                "Position_um": {"type": "DOUBLE", "scalar": float(z_values[i])},
            })
        stage_positions.append({
            "DefaultXYStage": {"type": "STRING", "scalar": ""},
            "DefaultZStage": {"type": "STRING", "scalar": z_stage_name},
            "DevicePositions": {"type": "PROPERTY_MAP", "array": device_positions},
            "GridCol": {"type": "INTEGER", "scalar": col},
            "GridRow": {"type": "INTEGER", "scalar": row},
            "Label": {"type": "STRING", "scalar": f"{labels_prefix}-{i+1:04d}"},
            "Properties": {"type": "PROPERTY_MAP", "scalar": {}},
        })

    data["encoding"] = "UTF-8"
    data["format"] = "Micro-Manager Property Map"
    data["major_version"] = 2
    data["minor_version"] = 0
    data.setdefault("map", {})
    data["map"]["StagePositions"] = {"type": "PROPERTY_MAP", "array": stage_positions}
    return json.dumps(data, indent=2).encode("utf-8")

def choose_three_boundary_points(contour, sample_xy, candidate_count=180):
    """Choose three widely separated points on the ORIGINAL tissue contour."""
    pts = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    sample = np.asarray(sample_xy, dtype=np.float64)
    if len(pts) < 3 or len(sample) != len(pts):
        raise ValueError("The original tissue contour does not contain enough calibrated points.")
    closed = np.vstack([pts, pts[0]])
    seg_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    perimeter = cumulative[-1]
    if perimeter <= 0:
        raise ValueError("The detected tissue boundary has zero perimeter.")
    distances = np.linspace(0.0, perimeter, candidate_count, endpoint=False)
    pixel_candidates = np.column_stack([
        np.interp(distances, cumulative, closed[:, 0]),
        np.interp(distances, cumulative, closed[:, 1]),
    ])
    sample_closed = np.vstack([sample, sample[0]])
    sample_candidates = np.column_stack([
        np.interp(distances, cumulative, sample_closed[:, 0]),
        np.interp(distances, cumulative, sample_closed[:, 1]),
    ])
    # Maximize triangle area: this spreads the three focus points around the boundary.
    best = None
    best_area2 = -1.0
    for i in range(candidate_count - 2):
        a = sample_candidates[i]
        for j in range(i + 1, candidate_count - 1):
            ab = sample_candidates[j] - a
            for k in range(j + 1, candidate_count):
                ac = sample_candidates[k] - a
                area2 = abs(ab[0] * ac[1] - ab[1] * ac[0])
                if area2 > best_area2:
                    best_area2 = area2
                    best = (i, j, k)
    if best is None or best_area2 < 1e-10:
        raise ValueError("Could not find three well-spaced, non-collinear boundary points.")
    return pixel_candidates[list(best)], sample_candidates[list(best)]
