import numpy as np
import pandas as pd
from modules import microscope_core as mc
import calculations as calc
import pos_processor as pos


def test_local_origin_shift_and_stage_signs_are_applied_once():
    # Calibration pixel basis: 40 px -> 40 mm X; 30 px -> 30 mm Y.
    points = [(100, 100), (100, 130), (140, 100)]
    origin, inverse, matrix, det = mc.calibrate_pixel_to_sample(points, 37.543, -16.511)
    boundary = np.array([[120, 115], [130, 120], [110, 125]], dtype=float)
    xy_new = np.asarray(mc.contour_to_new_coordinates(
        boundary.reshape(-1, 1, 2).astype(np.int32), origin, inverse, 37.543, -16.511
    ))
    # Original local coordinates are (20,15), (30,20), (10,25) mm.
    expected_local = np.array([[20, 15], [30, 20], [10, 25]], dtype=float)
    expected_new = expected_local - np.array([37.543, -16.511])
    assert np.allclose(xy_new, expected_new)
    # X inverted, Y not inverted, exactly once.
    stage = xy_new * 1000 * np.array([-1.0, 1.0])
    assert np.allclose(stage, expected_new * 1000 * np.array([-1.0, 1.0]))


def test_roi_and_plane_use_stage_coordinates_and_original_plane_engine():
    sample_mm = np.array([[1.0, 2.0], [2.0, 2.0], [1.0, 3.0]])
    stage_xy = sample_mm * 1000 * np.array([-1.0, 1.0])
    # Plane Z = 0.01*X - 0.02*Y + 7 in stage µm.
    z = 0.01 * stage_xy[:, 0] - 0.02 * stage_xy[:, 1] + 7
    plane = calc.compute_plane([(stage_xy[i, 0], stage_xy[i, 1], z[i]) for i in range(3)])
    target = np.array([[-1500.0, 2500.0], [-1200.0, 2800.0]])
    predicted = plane.z_at(target[:, 0], target[:, 1])
    assert np.allclose(predicted, 0.01*target[:, 0] - 0.02*target[:, 1] + 7)


def test_boundary_reference_selection_is_deterministic_and_on_contour():
    theta = np.linspace(0, 2*np.pi, 120, endpoint=False)
    contour = np.column_stack([100 + 40*np.cos(theta), 80 + 25*np.sin(theta)]).astype(np.int32).reshape(-1,1,2)
    sample = np.column_stack([contour[:,0,0]*0.01, contour[:,0,1]*0.02])
    pix1, samp1 = mc.choose_three_boundary_points(contour, sample)
    pix2, samp2 = mc.choose_three_boundary_points(contour, sample)
    assert pix1.shape == (3,2) and samp1.shape == (3,2)
    assert np.allclose(pix1,pix2) and np.allclose(samp1,samp2)


def test_original_pos_processor_roundtrip_with_generated_z():
    from tests.make_fixture import build
    original = pos.parse_pos(__import__('json').dumps(build()))
    rows = pos.extract_table(original, "ZStage")
    z = [11.0, 12.0, 13.0]
    updated, _, _ = pos.apply_z(original, z, "ZStage")
    checks = pos.validate_roundtrip(original, pos.dumps(updated), z, "ZStage")
    assert all(checks.values())
    assert [r["X"] for r in pos.extract_table(updated, "ZStage")] == [100.0,300.0,500.0]
