import numpy as np, pytest
import calculations as c


def test_known_plane():
    f = lambda x, y: 0.1 * x + 0.2 * y + 5
    pts = [(0, 0, f(0, 0)), (100, 0, f(100, 0)), (0, 50, f(0, 50))]
    p = c.compute_plane(pts)
    assert p.z_usable
    assert p.z_at(30, 40) == pytest.approx(f(30, 40))
    assert np.allclose(p.residuals(), 0, atol=p.tolerance())


def test_cross_product_and_d():
    p = c.compute_plane([(0, 0, 0), (1, 0, 0), (0, 1, 1)])
    assert (p.A, p.B, p.C) == (0.0, -1.0, 1.0)
    assert p.D == 0.0


def test_collinear_rejected():
    with pytest.raises(c.PlaneError):
        c.compute_plane([(0, 0, 0), (1, 1, 1), (2, 2, 2)])


def test_duplicate_and_missing_and_nan():
    with pytest.raises(c.PlaneError):
        c.compute_plane([(0, 0, 0), (0, 0, 0), (1, 2, 3)])
    with pytest.raises(c.PlaneError):
        c.compute_plane([(0, 0, None), (1, 0, 0), (0, 1, 1)])
    with pytest.raises(c.PlaneError):
        c.compute_plane([(0, 0, float("nan")), (1, 0, 0), (0, 1, 1)])


def test_vertical_plane_refuses_z():
    p = c.compute_plane([(0, 0, 0), (1, 0, 0), (0, 0, 1)])
    assert not p.z_usable
    with pytest.raises(c.PlaneError):
        p.z_at(1, 1)


def test_vectorised_and_micron_scale():
    pts = [(1000.5, 2000.25, 50.123), (5000, 1500, 55.5), (2000, 4000, 52.75)]
    p = c.compute_plane(pts)
    z = p.z_at([pt[0] for pt in pts], [pt[1] for pt in pts])
    assert np.allclose(z, [pt[2] for pt in pts])
