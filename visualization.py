"""Plotly 3D visualisation of the three points, the plane and its normal."""
import numpy as np
import plotly.graph_objects as go


def make_figure(plane, unit="units", true_scale=False):
    pts = plane.points
    span = np.maximum(np.ptp(pts, axis=0), 1e-9)
    lo, hi = pts.min(axis=0) - 0.25 * span, pts.max(axis=0) + 0.25 * span
    fig = go.Figure()
    if plane.z_usable:
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 25), np.linspace(lo[1], hi[1], 25))
        fig.add_trace(go.Surface(x=gx, y=gy, z=plane.z_at(gx, gy), opacity=0.55, showscale=False,
                                 colorscale=[[0, "#4C9BE8"], [1, "#4C9BE8"]], name="Plane", showlegend=True))
    else:
        fig.add_trace(go.Mesh3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=[0], j=[1], k=[2],
                                opacity=0.5, color="#4C9BE8", name="Plane (triangle only)", showlegend=True))
    fig.add_trace(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers+text",
                               text=["P1", "P2", "P3"], textposition="top center",
                               marker=dict(size=6, color="#D62728"), name="Reference points"))
    c = pts.mean(axis=0)
    L = 0.3 * float(np.max(hi - lo))
    tip = c + L * plane.n / plane.normal_mag
    fig.add_trace(go.Scatter3d(x=[c[0], tip[0]], y=[c[1], tip[1]], z=[c[2], tip[2]], mode="lines+text",
                               text=["", "n"], line=dict(color="#2CA02C", width=6), name="Normal vector"))
    fig.update_layout(
        height=650, margin=dict(l=0, r=0, t=30, b=0), legend=dict(orientation="h"),
        scene=dict(xaxis_title=f"X ({unit})", yaxis_title=f"Y ({unit})", zaxis_title=f"Z ({unit})",
                   aspectmode="data" if true_scale else "cube"))
    return fig
