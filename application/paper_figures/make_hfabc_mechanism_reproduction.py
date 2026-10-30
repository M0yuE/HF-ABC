from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Polygon,
    Rectangle,
)


OUT_DIR = Path(__file__).resolve().parent / "outputs"

W, H = 1800, 1220
PANEL_W, PANEL_H = 874, 512
LEFT_X, RIGHT_X = 18, 908
BOTTOM_Y, TOP_Y = 160, 690

COL = {
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#cbd5e1",
    "panel": "#ffffff",
    "page": "#f8fafc",
    "teal": "#76c8c2",
    "teal_edge": "#267f79",
    "teal_dark": "#0f766e",
    "gray": "#cfd4dc",
    "gray_dark": "#9aa3ae",
    "blue": "#2765ff",
    "blue_light": "#7bb6ff",
    "violet": "#6d28d9",
    "violet_light": "#9b5de5",
    "purple_deep": "#2e144f",
    "gold": "#f6bd22",
    "gold_edge": "#9a6400",
    "orange": "#d97706",
    "white": "#ffffff",
}


SURF_CMAP = LinearSegmentedColormap.from_list(
    "feasible_surface",
    [
        (0.00, "#291042"),
        (0.20, "#563596"),
        (0.43, "#74b7c6"),
        (0.68, "#9bdad6"),
        (1.00, "#e7fbf8"),
    ],
)

GRAY_CMAP = LinearSegmentedColormap.from_list(
    "infeasible_surface",
    ["#f4f5f7", "#d7dbe1", "#b9c0ca"],
)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def pt(rect: tuple[float, float, float, float], u: float, v: float) -> tuple[float, float]:
    x, y, w, h = rect
    return x + u * w, y + v * h


def smooth_closed_path(points: list[tuple[float, float]]) -> MplPath:
    pts = np.asarray(points, dtype=float)
    verts = [tuple(pts[0])]
    codes = [MplPath.MOVETO]
    n = len(pts)
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        verts.extend([tuple(c1), tuple(c2), tuple(p2)])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    verts.append(tuple(pts[0]))
    codes.append(MplPath.CLOSEPOLY)
    return MplPath(verts, codes)


def smooth_path(points: list[tuple[float, float]]) -> MplPath:
    pts = np.asarray(points, dtype=float)
    verts = [tuple(pts[0])]
    codes = [MplPath.MOVETO]
    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, len(pts) - 1)]
        c1 = p1 + (p2 - p0) / 6.0
        c2 = p2 - (p3 - p1) / 6.0
        verts.extend([tuple(c1), tuple(c2), tuple(p2)])
        codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    return MplPath(verts, codes)


def field_image(n: int, basins: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    Z = 0.72 + 0.08 * np.sin(10.5 * X + 1.4) * np.cos(8.0 * Y - 0.6)
    Z += 0.05 * np.sin(17.0 * (X + 0.25 * Y))
    for cx, cy, sx, sy, depth in basins:
        Z -= depth * np.exp(-(((X - cx) / sx) ** 2 + ((Y - cy) / sy) ** 2))
    Z = (Z - Z.min()) / (Z.max() - Z.min())
    return Z


def draw_panel(ax, rect: tuple[float, float, float, float], label: str, title: str) -> None:
    x, y, w, h = rect
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.0,rounding_size=12",
        facecolor=COL["panel"],
        edgecolor=COL["line"],
        linewidth=1.1,
        zorder=1,
    )
    ax.add_patch(patch)
    ax.text(x + 22, y + h - 39, label, fontsize=18, weight="bold", color=COL["ink"], va="center")
    ax.text(x + 75, y + h - 39, title, fontsize=18, weight="bold", color=COL["ink"], va="center")


def draw_surface(
    ax,
    rect: tuple[float, float, float, float],
    *,
    kind: str = "large",
    final: bool = False,
    zoom: bool = False,
    mini: bool = False,
    stage: int = 0,
) -> PathPatch:
    if zoom:
        shape_u = [
            (0.03, 0.34),
            (0.09, 0.49),
            (0.22, 0.52),
            (0.37, 0.63),
            (0.54, 0.57),
            (0.72, 0.64),
            (0.88, 0.56),
            (0.97, 0.38),
            (0.92, 0.21),
            (0.76, 0.18),
            (0.61, 0.23),
            (0.45, 0.20),
            (0.29, 0.25),
            (0.14, 0.22),
            (0.04, 0.27),
        ]
        basins = [(0.50, 0.38, 0.21, 0.20, 0.83), (0.34, 0.45, 0.14, 0.16, 0.25)]
        mesh_alpha = 0.42
    elif mini:
        shape_u = [
            (0.04, 0.42),
            (0.18, 0.58),
            (0.37, 0.62),
            (0.54, 0.55),
            (0.76, 0.61),
            (0.96, 0.45),
            (0.86, 0.27),
            (0.61, 0.22),
            (0.43, 0.26),
            (0.20, 0.24),
        ]
        if stage == 0:
            basins = [(0.45, 0.43, 0.20, 0.18, 0.36)]
        elif stage == 1:
            basins = [(0.55, 0.38, 0.22, 0.20, 0.72)]
        else:
            basins = [(0.57, 0.38, 0.23, 0.18, 0.86)]
        mesh_alpha = 0.36
    else:
        shape_u = [
            (0.06, 0.34),
            (0.14, 0.48),
            (0.27, 0.50),
            (0.37, 0.57),
            (0.50, 0.70),
            (0.62, 0.63),
            (0.75, 0.68),
            (0.90, 0.55),
            (0.95, 0.39),
            (0.84, 0.29),
            (0.70, 0.30),
            (0.56, 0.25),
            (0.42, 0.28),
            (0.30, 0.22),
            (0.17, 0.26),
            (0.08, 0.28),
        ]
        basins = [(0.41, 0.34, 0.17, 0.14, 0.78), (0.73, 0.44, 0.13, 0.12, 0.58)]
        mesh_alpha = 0.38

    points = [pt(rect, u, v) for u, v in shape_u]
    path = smooth_closed_path(points)

    if not mini:
        gray_points = [pt(rect, min(0.98, u * 1.02 - 0.01), min(0.78, v + 0.07)) for u, v in shape_u]
        gray_path = smooth_closed_path(gray_points)
        gray_patch = PathPatch(
            gray_path,
            facecolor="none",
            edgecolor="none",
            zorder=6,
            path_effects=[pe.withSimplePatchShadow(offset=(7, -8), alpha=0.14, rho=0.98), pe.Normal()],
        )
        ax.add_patch(gray_patch)
        gx0, gy0, gw, gh = rect
        gimg = 0.55 + 0.45 * field_image(360, [(0.52, 0.42, 0.23, 0.20, 0.20)])
        im_g = ax.imshow(
            gimg,
            extent=(gx0 + 15, gx0 + gw - 15, gy0 + 86, gy0 + gh - 118),
            origin="lower",
            cmap=GRAY_CMAP,
            alpha=0.57,
            interpolation="bicubic",
            zorder=6,
        )
        im_g.set_clip_path(gray_patch)
        draw_mesh(ax, rect, gray_patch, color="#94a3b8", alpha=0.18, zorder=7, count=7)

    patch = PathPatch(
        path,
        facecolor="none",
        edgecolor="none",
        zorder=10,
        path_effects=[pe.withSimplePatchShadow(offset=(4, -5), alpha=0.18, rho=0.98), pe.Normal()],
    )
    ax.add_patch(patch)

    x, y, w, h = rect
    img = field_image(540 if not mini else 260, basins)
    im = ax.imshow(
        img,
        extent=(x + 8, x + w - 8, y + 42, y + h - 88),
        origin="lower",
        cmap=SURF_CMAP,
        alpha=0.88 if not mini else 0.86,
        interpolation="bicubic",
        zorder=10,
    )
    im.set_clip_path(patch)
    outline = PathPatch(path, facecolor="none", edgecolor=COL["teal_edge"], linewidth=1.6 if not mini else 0.9, zorder=13)
    ax.add_patch(outline)
    draw_mesh(ax, rect, patch, color=COL["white"], alpha=mesh_alpha, zorder=12, count=9 if not mini else 5)

    if not mini:
        draw_basin_rings(ax, rect, basins, zorder=16)
    return patch


def draw_mesh(
    ax,
    rect: tuple[float, float, float, float],
    clip: PathPatch,
    *,
    color: str,
    alpha: float,
    zorder: int,
    count: int,
) -> None:
    x, y, w, h = rect
    us = np.linspace(0.04, 0.96, 220)
    for vv in np.linspace(0.22, 0.70, count):
        pts = []
        for u in us:
            v = vv + 0.025 * np.sin(8 * u + 5 * vv) + 0.018 * (u - 0.5)
            pts.append(pt(rect, u, v))
        line = PathPatch(smooth_path(pts), facecolor="none", edgecolor=color, linewidth=0.7, alpha=alpha, zorder=zorder)
        line.set_clip_path(clip)
        ax.add_patch(line)
    vs = np.linspace(0.20, 0.70, 180)
    for uu in np.linspace(0.14, 0.86, max(5, count - 1)):
        pts = []
        for v in vs:
            u = uu + 0.018 * np.sin(10 * v + 3 * uu) - 0.028 * (v - 0.45)
            pts.append(pt(rect, u, v))
        line = PathPatch(smooth_path(pts), facecolor="none", edgecolor=color, linewidth=0.65, alpha=alpha, zorder=zorder)
        line.set_clip_path(clip)
        ax.add_patch(line)


def draw_basin_rings(ax, rect: tuple[float, float, float, float], basins, zorder: int) -> None:
    for i, (cx, cy, sx, sy, _depth) in enumerate(basins):
        x, y = pt(rect, cx, cy)
        rw, rh = sx * rect[2] * 1.95, sy * rect[3] * 1.65
        ax.add_patch(
            Ellipse(
                (x, y),
                rw,
                rh,
                angle=-4 if i == 0 else 3,
                fill=False,
                edgecolor=COL["violet"],
                linewidth=1.5,
                linestyle=(0, (5, 4)),
                alpha=0.86,
                zorder=zorder,
            )
        )
        ax.add_patch(
            Ellipse(
                (x, y),
                rw * 0.52,
                rh * 0.48,
                angle=-4 if i == 0 else 3,
                fill=False,
                edgecolor=COL["purple_deep"],
                linewidth=1.0,
                alpha=0.50,
                zorder=zorder,
            )
        )


def blue_dot(ax, x: float, y: float, r: float = 5.4, z: int = 40) -> None:
    ax.add_patch(Ellipse((x, y - r * 0.86), r * 2.6, r * 0.58, facecolor="none", edgecolor="#7aa7ff", alpha=0.42, linewidth=0.9, zorder=z - 1))
    ax.add_patch(Circle((x, y), r, facecolor=COL["blue_light"], edgecolor="#1d4ed8", linewidth=1.2, zorder=z))
    ax.add_patch(Circle((x - 0.35 * r, y + 0.35 * r), r * 0.28, facecolor="white", edgecolor="none", alpha=0.88, zorder=z + 1))


def gold_dot(ax, x: float, y: float, r: float = 5.8, z: int = 45) -> None:
    ax.add_patch(Ellipse((x, y - r * 0.9), r * 2.5, r * 0.55, facecolor="none", edgecolor="#eab308", alpha=0.42, linewidth=0.9, zorder=z - 1))
    ax.add_patch(Circle((x, y), r, facecolor=COL["gold"], edgecolor=COL["gold_edge"], linewidth=1.1, zorder=z))
    ax.add_patch(Circle((x - 0.35 * r, y + 0.34 * r), r * 0.25, facecolor="#fff7cc", edgecolor="none", alpha=0.90, zorder=z + 1))


def rotate(x: float, y: float, angle: float) -> tuple[float, float]:
    a = np.deg2rad(angle)
    return x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a)


def bee(ax, x: float, y: float, s: float = 1.0, angle: float = 0.0, z: int = 55) -> None:
    def p(px: float, py: float) -> tuple[float, float]:
        rx, ry = rotate(px * s, py * s, angle)
        return x + rx, y + ry

    for cx, cy, wing_angle in [(-5.0, 6.0, 28), (5.0, 6.0, -28)]:
        wx, wy = p(cx, cy)
        ax.add_patch(
            Ellipse(
                (wx, wy),
                12.0 * s,
                5.2 * s,
                angle=angle + wing_angle,
                facecolor=(1, 1, 1, 0.78),
                edgecolor="#6b7280",
                linewidth=0.65,
                zorder=z,
            )
        )
    bx, by = p(0, 0)
    ax.add_patch(
        Ellipse(
            (bx, by),
            15.0 * s,
            9.2 * s,
            angle=angle,
            facecolor="#f5c51b",
            edgecolor="#111827",
            linewidth=0.9,
            zorder=z + 1,
        )
    )
    for sx in [-3.2, 1.5]:
        x1, y1 = p(sx, -4.0)
        x2, y2 = p(sx + 1.2, 4.0)
        ax.plot([x1, x2], [y1, y2], color="#111827", linewidth=1.0 * s, zorder=z + 2)
    hx, hy = p(8.2, 0)
    ax.add_patch(Circle((hx, hy), 2.6 * s, facecolor="#111827", edgecolor="none", zorder=z + 2))
    for ay in [-2.0, 2.0]:
        x1, y1 = p(10.0, ay)
        x2, y2 = p(15.0, ay + (2.0 if ay > 0 else -2.0))
        ax.plot([x1, x2], [y1, y2], color="#111827", linewidth=0.8 * s, zorder=z + 2)


def curved_arrow(
    ax,
    p1: tuple[float, float],
    p2: tuple[float, float],
    *,
    color: str,
    lw: float = 1.4,
    rad: float = 0.0,
    dashed: bool = False,
    z: int = 50,
    ms: float = 14,
) -> None:
    arr = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=(0, (4, 4)) if dashed else "solid",
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=3,
        shrinkB=3,
        zorder=z,
    )
    ax.add_patch(arr)


def legend_box(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=8",
            facecolor=(1, 1, 1, 0.94),
            edgecolor=COL["line"],
            linewidth=0.9,
            zorder=80,
        )
    )


def draw_axis_triad(ax, x: float, y: float, s: float = 1.0) -> None:
    props = dict(arrowstyle="-|>", color="#111827", linewidth=1.1, mutation_scale=10 * s, zorder=65)
    ax.add_patch(FancyArrowPatch((x, y), (x + 58 * s, y), **props))
    ax.add_patch(FancyArrowPatch((x, y), (x, y + 48 * s), **props))
    ax.add_patch(FancyArrowPatch((x, y), (x - 38 * s, y - 38 * s), **props))
    ax.text(x + 66 * s, y - 2 * s, r"$x_1$", fontsize=10 * s, va="center")
    ax.text(x - 8 * s, y + 58 * s, r"$x_2$", fontsize=10 * s, ha="center")
    ax.text(x - 48 * s, y - 48 * s, r"$x_3$", fontsize=10 * s, ha="center")


def draw_large_panel_content(ax, rect: tuple[float, float, float, float], final: bool = False) -> None:
    draw_surface(ax, rect, final=final)
    x, y, w, h = rect
    candidates = [
        (0.20, 0.34),
        (0.26, 0.42),
        (0.29, 0.30),
        (0.35, 0.37),
        (0.42, 0.47),
        (0.50, 0.34),
        (0.54, 0.29),
        (0.56, 0.55),
        (0.63, 0.38),
        (0.69, 0.48),
        (0.76, 0.41),
        (0.84, 0.57),
        (0.90, 0.50),
        (0.88, 0.35),
        (0.32, 0.27),
        (0.47, 0.28),
        (0.58, 0.28),
        (0.72, 0.31),
    ]
    for u, v in candidates:
        blue_dot(ax, *pt(rect, u, v), r=5.3)

    if not final:
        bx, by = pt(rect, 0.61, 0.36)
        bw, bh = 106, 76
        ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="none", edgecolor=COL["ink"], linewidth=1.1, zorder=62))
        ax.plot([bx + bw, RIGHT_X + 8], [by + bh, TOP_Y + PANEL_H - 140], color=COL["ink"], linewidth=1.3, zorder=61)
        ax.plot([bx + bw, RIGHT_X + 8], [by, TOP_Y + 116], color=COL["ink"], linewidth=1.3, zorder=61)

        tx, ty = pt(rect, 0.04, 0.78)
        ax.text(tx, ty, r"feasible space $\mathcal{F}$", fontsize=12, weight="bold", color=COL["teal_dark"], zorder=70)
        ax.plot([tx + 170, x + 295], [ty + 3, y + 322], color=COL["muted"], linewidth=0.9, zorder=70)
        ax.add_patch(Circle((x + 295, y + 322), 2.1, facecolor=COL["ink"], zorder=71))
        ax.text(tx, ty - 58, "infeasible space", fontsize=11, color=COL["muted"], zorder=70)
        ax.text(tx, ty - 86, r"$\mathbb{R}^d \setminus \mathcal{F}$", fontsize=12, color=COL["muted"], zorder=70)
        ax.plot([tx + 162, x + 235], [ty - 58, y + 286], color=COL["muted"], linewidth=0.9, zorder=70)
        ax.add_patch(Circle((x + 235, y + 286), 2.1, facecolor=COL["muted"], zorder=71))

    if final:
        path_pts = [pt(rect, u, v) for u, v in [(0.34, 0.27), (0.40, 0.33), (0.48, 0.27), (0.57, 0.34), (0.66, 0.42), (0.70, 0.47), (0.61, 0.36), (0.52, 0.30), (0.43, 0.29), (0.37, 0.36)]]
        ax.add_patch(PathPatch(smooth_path(path_pts), facecolor="none", edgecolor=COL["orange"], linewidth=3.0, zorder=66))
        for i, (px, py) in enumerate(path_pts[::2]):
            gold_dot(ax, px, py, r=5.7, z=68)
            if i < 4:
                ax.text(px + 10, py + 8, f"$t_{i * 2}$", fontsize=9, color=COL["orange"], zorder=69)
        opt_x, opt_y = pt(rect, 0.42, 0.30)
        for u, v in [(0.37, 0.28), (0.39, 0.31), (0.42, 0.34), (0.45, 0.29), (0.48, 0.32)]:
            gold_dot(ax, *pt(rect, u, v), r=5.3, z=69)
        star = star_points(opt_x, opt_y, 22, 9)
        ax.add_patch(Polygon(star, closed=True, facecolor=COL["gold"], edgecolor=COL["gold_edge"], linewidth=1.3, zorder=75))
        ax.text(opt_x + 34, opt_y + 17, r"opt", color=COL["orange"], fontsize=15, weight="bold", zorder=76)
        ax.add_patch(FancyArrowPatch((opt_x + 31, opt_y + 12), (opt_x + 6, opt_y + 5), arrowstyle="-|>", mutation_scale=13, color=COL["orange"], linewidth=1.5, zorder=76))

    draw_axis_triad(ax, x + w - 110, y + 58, 1.0)
    if final:
        draw_final_legend(ax, x + 18, y + 20)
    else:
        draw_initial_legend(ax, x + 18, y + 24)


def star_points(x: float, y: float, r1: float, r2: float, n: int = 5) -> list[tuple[float, float]]:
    pts = []
    for i in range(n * 2):
        a = np.pi / 2 + i * np.pi / n
        r = r1 if i % 2 == 0 else r2
        pts.append((x + r * np.cos(a), y + r * np.sin(a)))
    return pts


def draw_initial_legend(ax, x: float, y: float) -> None:
    legend_box(ax, x, y, 250, 112)
    ax.add_patch(Rectangle((x + 16, y + 76), 32, 20, facecolor=COL["teal"], edgecolor=COL["teal"], linewidth=0, zorder=81))
    ax.text(x + 62, y + 86, r"feasible manifold $\mathcal{F}$", fontsize=10, va="center", zorder=81)
    ax.add_patch(Rectangle((x + 16, y + 48), 32, 20, facecolor=COL["gray"], edgecolor=COL["gray"], linewidth=0, zorder=81))
    ax.text(x + 62, y + 58, "infeasible space", fontsize=10, va="center", zorder=81)
    blue_dot(ax, x + 32, y + 27, r=5, z=82)
    ax.text(x + 62, y + 27, "evaluated candidates", fontsize=10, va="center", zorder=81)


def draw_final_legend(ax, x: float, y: float) -> None:
    legend_box(ax, x, y, 270, 138)
    ax.add_patch(Rectangle((x + 16, y + 102), 32, 20, facecolor=COL["teal"], edgecolor=COL["teal"], linewidth=0, zorder=81))
    ax.text(x + 62, y + 112, r"feasible manifold $\mathcal{F}$", fontsize=10, va="center", zorder=81)
    ax.add_patch(Rectangle((x + 16, y + 74), 32, 20, facecolor=COL["gray"], edgecolor=COL["gray"], linewidth=0, zorder=81))
    ax.text(x + 62, y + 84, "infeasible space", fontsize=10, va="center", zorder=81)
    blue_dot(ax, x + 32, y + 55, r=5, z=82)
    ax.text(x + 62, y + 55, "evaluated candidates", fontsize=10, va="center", zorder=81)
    gold_dot(ax, x + 32, y + 26, r=5.5, z=82)
    ax.text(x + 62, y + 26, "elite / best (opt)", fontsize=10, va="center", zorder=81)


def draw_zoom_panel_content(ax, rect: tuple[float, float, float, float]) -> None:
    draw_surface(ax, rect, zoom=True)
    cand = [
        (0.14, 0.53),
        (0.22, 0.34),
        (0.29, 0.45),
        (0.38, 0.66),
        (0.40, 0.54),
        (0.49, 0.35),
        (0.58, 0.48),
        (0.64, 0.64),
        (0.70, 0.36),
        (0.76, 0.53),
        (0.84, 0.42),
        (0.91, 0.58),
    ]
    for u, v in cand:
        blue_dot(ax, *pt(rect, u, v), r=5.1, z=45)

    bees = [
        (0.18, 0.39, -25),
        (0.29, 0.35, 18),
        (0.33, 0.61, 12),
        (0.43, 0.28, -13),
        (0.52, 0.42, 15),
        (0.60, 0.33, -18),
        (0.66, 0.56, -7),
        (0.80, 0.35, 22),
        (0.86, 0.48, -12),
    ]
    for u, v, a in bees:
        bee(ax, *pt(rect, u, v), s=1.05, angle=a, z=58)

    purple_arrows = [
        ((0.15, 0.43), (0.28, 0.54), 0.24),
        ((0.33, 0.67), (0.44, 0.48), -0.20),
        ((0.60, 0.62), (0.70, 0.47), 0.25),
        ((0.82, 0.34), (0.88, 0.51), -0.28),
        ((0.72, 0.30), (0.55, 0.29), 0.18),
        ((0.31, 0.31), (0.21, 0.39), 0.16),
    ]
    for a, b, rad in purple_arrows:
        curved_arrow(ax, pt(rect, *a), pt(rect, *b), color=COL["violet_light"], dashed=True, rad=rad, lw=1.5, z=54, ms=14)

    white_arrows = [
        ((0.23, 0.47), (0.49, 0.36), -0.10),
        ((0.37, 0.62), (0.48, 0.36), -0.05),
        ((0.64, 0.52), (0.50, 0.37), 0.05),
        ((0.76, 0.48), (0.54, 0.36), 0.13),
    ]
    for a, b, rad in white_arrows:
        curved_arrow(ax, pt(rect, *a), pt(rect, *b), color=COL["white"], dashed=True, rad=rad, lw=1.35, z=59, ms=12)

    x, y, w, h = rect
    legend_box(ax, x + w - 192, y + 28, 156, 76)
    blue_dot(ax, x + w - 166, y + 78, r=5, z=82)
    ax.text(x + w - 138, y + 78, "candidates", fontsize=9, va="center", zorder=82)
    bee(ax, x + w - 166, y + 45, s=0.82, angle=10, z=82)
    ax.text(x + w - 138, y + 45, "bee agents", fontsize=9, va="center", zorder=82)


def draw_stage(ax, rect: tuple[float, float, float, float], idx: int, title: str, subtitle: str, formula_lines: list[str], note: str) -> None:
    x, y, w, h = rect
    ax.add_patch(Circle((x + 18, y + h - 30), 14, facecolor=COL["teal_dark"], edgecolor="none", zorder=40))
    ax.text(x + 18, y + h - 30, str(idx + 1), fontsize=11, color="white", weight="bold", ha="center", va="center", zorder=41)
    ax.text(x + 42, y + h - 18, title, fontsize=11, weight="bold", color=COL["teal_dark"], va="top", zorder=41)
    ax.text(x + 42, y + h - 42, subtitle, fontsize=10, weight="bold", color=COL["teal_dark"], va="top", zorder=41)

    surf_rect = (x + 10, y + 168, w - 20, 135)
    draw_surface(ax, surf_rect, mini=True, stage=idx)
    for u, v in [(0.23, 0.49), (0.50, 0.50), (0.75, 0.45)]:
        blue_dot(ax, *pt(surf_rect, u, v), r=4.2, z=61)
    if idx == 0:
        for bx, by, ba in [(0.20, 0.88, -5), (0.52, 0.93, 7), (0.83, 0.85, -14)]:
            bee(ax, *pt(surf_rect, bx, by), s=0.80, angle=ba, z=63)
            ax.plot([pt(surf_rect, bx, by)[0], pt(surf_rect, bx * 0.9 + 0.05, 0.52)[0]], [pt(surf_rect, bx, by)[1] - 12, pt(surf_rect, bx * 0.9 + 0.05, 0.52)[1] + 6], color="#64748b", linewidth=0.9, linestyle=(0, (3, 3)), zorder=62)
    elif idx == 1:
        for bx, by, ba in [(0.26, 0.84, -9), (0.57, 0.52, 13), (0.82, 0.84, 18)]:
            bee(ax, *pt(surf_rect, bx, by), s=0.82, angle=ba, z=64)
        for a, b, rad in [((0.25, 0.77), (0.55, 0.46), -0.24), ((0.75, 0.78), (0.56, 0.45), 0.22), ((0.43, 0.83), (0.82, 0.48), -0.20)]:
            curved_arrow(ax, pt(surf_rect, *a), pt(surf_rect, *b), color=COL["violet_light"], dashed=True, rad=rad, lw=1.2, z=66, ms=11)
    else:
        for bx, by, ba in [(0.43, 0.48, -12), (0.67, 0.51, 16)]:
            bee(ax, *pt(surf_rect, bx, by), s=0.82, angle=ba, z=64)
        for u, v in [(0.45, 0.40), (0.52, 0.36), (0.58, 0.43), (0.64, 0.36), (0.56, 0.51)]:
            gold_dot(ax, *pt(surf_rect, u, v), r=4.7, z=66)
        for a, b, rad in [((0.18, 0.76), (0.50, 0.40), -0.20), ((0.87, 0.77), (0.62, 0.41), 0.17)]:
            curved_arrow(ax, pt(surf_rect, *a), pt(surf_rect, *b), color=COL["violet_light"], dashed=True, rad=rad, lw=1.2, z=66, ms=11)

    fy = y + 118
    for i, line in enumerate(formula_lines):
        ax.text(x + w / 2, fy - i * 38, line, fontsize=12 if i == 0 else 10.5, ha="center", va="center", color=COL["ink"], zorder=70)
    ax.text(x + 18, y + 24, note, fontsize=9.5, color=COL["ink"], linespacing=1.28, va="bottom", zorder=70)


def draw_algorithm_panel_content(ax, rect: tuple[float, float, float, float]) -> None:
    x, y, w, h = rect
    stage_w = 252
    gap = 48
    stages = [
        (
            0,
            "Swarm initialization",
            "near feasible points",
            [r"$x_i^{(0)} \sim \mathcal{U}(\Omega_b)$"],
            "Initialize bees on/near $\\mathcal{F}$\nwithin bounded region $\\Omega_b$.",
        ),
        (
            1,
            "Constraint-aware perturbation",
            "+ multi-fidelity evaluation",
            [
                r"$x_i'=\Pi_{\mathcal{F}}\,(x_i+\alpha_t\epsilon_i+\beta_t(x_{\rm best}-x_i))$",
                r"$s_i^{(m)}=f_m(x_i'),\quad m=1,\ldots,M$",
            ],
            "Project to $\\mathcal{F}$; evaluate with\nfidelity levels $m=1,\\ldots,M$.",
        ),
        (
            2,
            "Elite-guided local exploitation",
            "near basin floor",
            [r"$x_{\rm best}=\arg\min_{x\in\mathcal{E}} f(x)$"],
            "Update elite set $\\mathcal{E}$ and focus\nsearch around best locations.",
        ),
    ]
    start_x = x + 24
    y0 = y + 72
    for idx, title, subtitle, formula, note in stages:
        sx = start_x + idx * (stage_w + gap)
        draw_stage(ax, (sx, y0, stage_w, h - 118), idx, title, subtitle, formula, note)

    curved_arrow(ax, (start_x + stage_w - 12, y + 200), (start_x + stage_w + gap + 12, y + 200), color="#7c8797", lw=1.4, rad=-0.08, z=85, ms=16)
    curved_arrow(ax, (start_x + 2 * stage_w + gap - 12, y + 200), (start_x + 2 * stage_w + 2 * gap + 12, y + 200), color="#7c8797", lw=1.4, rad=-0.08, z=85, ms=16)

    legend_box(ax, x + 130, y + 20, 610, 42)
    blue_dot(ax, x + 165, y + 42, r=5, z=86)
    ax.text(x + 194, y + 42, "candidates", fontsize=10, va="center", zorder=86)
    gold_dot(ax, x + 340, y + 42, r=5.4, z=86)
    ax.text(x + 370, y + 42, r"elite candidates ($\mathcal{E}$)", fontsize=10, va="center", zorder=86)
    bee(ax, x + 555, y + 42, s=0.78, angle=8, z=86)
    ax.text(x + 585, y + 42, "bee agents", fontsize=10, va="center", zorder=86)


def draw_big_arrow(ax, start: tuple[float, float], end: tuple[float, float], *, z: int = 120) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="Simple,tail_width=18,head_width=64,head_length=46",
            color="#8b95a3",
            linewidth=0,
            alpha=0.95,
            zorder=z,
        )
    )


def draw_footer(ax) -> None:
    x, y, w, h = 112, 20, 1576, 66
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.0,rounding_size=11",
            facecolor="white",
            edgecolor=COL["line"],
            linewidth=1.0,
            zorder=120,
        )
    )
    items = [
        ("a", "Global search space", x + 52),
        ("b", "Local zoom & exploration", x + 460),
        ("c", "Algorithmic mechanism", x + 910),
        ("d", "Optimal solution found", x + 1340),
    ]
    for letter, text, ix in items:
        ax.add_patch(Circle((ix, y + 33), 16, facecolor="white", edgecolor="#94a3b8", linewidth=0.9, zorder=121))
        ax.text(ix, y + 33, letter, fontsize=12, weight="bold", ha="center", va="center", zorder=122)
        ax.text(ix + 36, y + 33, text, fontsize=12, weight="bold", va="center", zorder=122)
    for a, b in [((x + 320, y + 33), (x + 392, y + 33)), ((x + 750, y + 33), (x + 825, y + 33)), ((x + 1195, y + 33), (x + 1270, y + 33))]:
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=28,
                linewidth=2.0,
                color="#7c8797",
                zorder=121,
            )
        )


def make_figure() -> plt.Figure:
    setup_style()
    fig = plt.figure(figsize=(18, 12.2), dpi=200, facecolor=COL["page"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    rect_a = (LEFT_X, TOP_Y, PANEL_W, PANEL_H)
    rect_b = (RIGHT_X, TOP_Y, PANEL_W, PANEL_H)
    rect_d = (LEFT_X, BOTTOM_Y, PANEL_W, PANEL_H)
    rect_c = (RIGHT_X, BOTTOM_Y, PANEL_W, PANEL_H)

    draw_panel(ax, rect_a, "(a)", "Initial hybrid-constrained search space")
    draw_panel(ax, rect_b, "(b)", "Zoomed region and bee-swarm perturbation")
    draw_panel(ax, rect_d, "(d)", "Final result")
    draw_panel(ax, rect_c, "(c)", "Proposed algorithmic mechanism")

    draw_large_panel_content(ax, rect_a, final=False)
    draw_zoom_panel_content(ax, rect_b)
    draw_large_panel_content(ax, rect_d, final=True)
    draw_algorithm_panel_content(ax, rect_c)

    draw_big_arrow(ax, (LEFT_X + PANEL_W - 24, TOP_Y + 260), (RIGHT_X + 36, TOP_Y + 260))
    draw_big_arrow(ax, (RIGHT_X + PANEL_W / 2, TOP_Y + 12), (RIGHT_X + PANEL_W / 2, BOTTOM_Y + PANEL_H - 44))
    draw_big_arrow(ax, (RIGHT_X + 36, BOTTOM_Y + 260), (LEFT_X + PANEL_W - 24, BOTTOM_Y + 260))
    draw_footer(ax)
    return fig


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    stem = OUT_DIR / "hfabc_bee_swarm_mechanism_redrawn"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".pdf"), facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
    fig.savefig(stem.with_suffix(".svg"), facecolor=fig.get_facecolor(), bbox_inches=None, pad_inches=0)
    plt.close(fig)
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".svg"))


if __name__ == "__main__":
    main()
