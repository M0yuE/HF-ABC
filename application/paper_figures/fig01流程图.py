import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def objective_function(x, y):
    base = 2.15 + 0.055 * (x**2 + y**2)

    well_1 = -1.95 * np.exp(-(((x + 1.55) ** 2) / 0.75 + ((y + 1.10) ** 2) / 0.45))
    well_2 = -1.45 * np.exp(-(((x - 1.35) ** 2) / 0.55 + ((y - 1.45) ** 2) / 0.85))
    well_3 = -0.85 * np.exp(-(((x + 0.15) ** 2) / 0.40 + ((y - 2.45) ** 2) / 0.70))

    peak_1 = 1.20 * np.exp(-(((x - 2.45) ** 2) / 0.45 + ((y + 0.60) ** 2) / 0.55))
    peak_2 = 0.95 * np.exp(-(((x + 2.65) ** 2) / 0.80 + ((y - 2.05) ** 2) / 0.65))
    peak_3 = 0.72 * np.exp(-(((x - 0.15) ** 2) / 1.15 + ((y + 2.75) ** 2) / 0.70))

    undulation = (
        0.18 * np.sin(1.85 * x) * np.cos(1.35 * y)
        + 0.10 * np.sin(2.20 * y + 0.55 * x)
        + 0.07 * np.cos(2.70 * x - 1.10 * y)
    )

    return base + well_1 + well_2 + well_3 + peak_1 + peak_2 + peak_3 + undulation


def is_invalid_region(x, y):
    theta = np.deg2rad(28.0)
    xc, yc = 2.15, -1.65
    xr = (x - xc) * np.cos(theta) + (y - yc) * np.sin(theta)
    yr = -(x - xc) * np.sin(theta) + (y - yc) * np.cos(theta)

    ellipse = (xr / 1.00) ** 2 + (yr / 0.55) ** 2 < 1.0
    notch = (
        (x > 1.05)
        & (x < 3.35)
        & (y > -2.65)
        & (y < -0.65)
        & (np.sin(3.3 * x + 1.4 * y) > 0.58)
    )

    return ellipse | notch


def surface_z(x, y, offset=0.055):
    return objective_function(x, y) + offset


def sample_valid_points(rng, n, xlim=(-3.9, 3.9), ylim=(-3.9, 3.9)):
    xs, ys = [], []
    while len(xs) < n:
        x = rng.uniform(xlim[0], xlim[1])
        y = rng.uniform(ylim[0], ylim[1])
        if not is_invalid_region(x, y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def sample_cluster(rng, center, n, scale=(0.30, 0.30)):
    xs, ys = [], []
    while len(xs) < n:
        x = rng.normal(center[0], scale[0])
        y = rng.normal(center[1], scale[1])
        if -4.0 <= x <= 4.0 and -4.0 <= y <= 4.0 and not is_invalid_region(x, y):
            xs.append(x)
            ys.append(y)
    return np.asarray(xs), np.asarray(ys)


def add_3d_arrow(ax, start, end, color="black", lw=1.4):
    direction = np.asarray(end) - np.asarray(start)
    ax.quiver(
        start[0],
        start[1],
        start[2],
        direction[0],
        direction[1],
        direction[2],
        arrow_length_ratio=0.14,
        color=color,
        linewidth=lw,
        normalize=False,
        zorder=80,
    )


def main():
    rng = np.random.default_rng(7)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    x = np.linspace(-4, 4, 240)
    y = np.linspace(-4, 4, 240)
    X, Y = np.meshgrid(x, y)
    Z = objective_function(X, Y)

    invalid = is_invalid_region(X, Y)
    Z_invalid = np.where(invalid, Z + 0.075, np.nan)

    min_idx = np.unravel_index(np.argmin(Z), Z.shape)
    x_best = float(X[min_idx])
    y_best = float(Y[min_idx])
    z_best = float(Z[min_idx])

    cmap = LinearSegmentedColormap.from_list(
        "teal_blue_purple_gold_gray",
        [
            (0.00, "#3b0f70"),
            (0.16, "#2c7fb8"),
            (0.34, "#2dd4bf"),
            (0.58, "#d8f3dc"),
            (0.78, "#e5e7eb"),
            (1.00, "#f4a261"),
        ],
    )

    fig = plt.figure(figsize=(9.4, 7.2), facecolor="white")

    # 关键：关闭 mplot3d 自动深度排序，让 zorder 更可控
    try:
        ax = fig.add_subplot(
            111,
            projection="3d",
            facecolor="white",
            computed_zorder=False,
        )
    except TypeError:
        # 兼容较旧版本 Matplotlib
        ax = fig.add_subplot(111, projection="3d", facecolor="white")

    # 主曲面：降低透明度，减少对点和星号的遮挡
    surface = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        rstride=2,
        cstride=2,
        linewidth=0,
        antialiased=True,
        alpha=0.72,
        shade=True,
        zorder=1,
    )

    # 无效区域：进一步降低透明度
    ax.plot_surface(
        X,
        Y,
        Z_invalid,
        color="#8a8f98",
        linewidth=0,
        antialiased=True,
        alpha=0.25,
        shade=False,
        zorder=2,
    )

    n_eval = 56
    eval_x, eval_y = sample_valid_points(rng, n_eval)
    eval_z = surface_z(eval_x, eval_y, 0.22)

    hf_x1, hf_y1 = sample_cluster(rng, (-1.55, -1.10), 13, scale=(0.50, 0.38))
    hf_x2, hf_y2 = sample_cluster(rng, (1.35, 1.45), 10, scale=(0.38, 0.46))
    hf_x = np.concatenate([hf_x1, hf_x2])
    hf_y = np.concatenate([hf_y1, hf_y2])
    hf_z = surface_z(hf_x, hf_y, 0.28)

    elite_x1, elite_y1 = sample_cluster(rng, (x_best, y_best), 7, scale=(0.18, 0.16))
    elite_x2, elite_y2 = sample_cluster(rng, (1.35, 1.45), 4, scale=(0.22, 0.20))
    elite_x = np.concatenate([elite_x1, elite_x2])
    elite_y = np.concatenate([elite_y1, elite_y2])
    elite_z = surface_z(elite_x, elite_y, 0.34)

    restart_x = np.array([-3.45, -3.15, 3.32, 2.88, -2.62])
    restart_y = np.array([3.35, -3.22, 3.08, -3.20, 2.83])
    valid_restart = ~is_invalid_region(restart_x, restart_y)
    restart_x = restart_x[valid_restart]
    restart_y = restart_y[valid_restart]
    restart_z = surface_z(restart_x, restart_y, 0.30)

    refine_x, refine_y = sample_cluster(rng, (x_best, y_best), 24, scale=(0.13, 0.11))
    refine_z = surface_z(refine_x, refine_y, 0.38)

    ax.scatter(
        eval_x,
        eval_y,
        eval_z,
        s=24,
        c="#8fd3ff",
        edgecolors="white",
        linewidths=0.45,
        depthshade=False,
        alpha=0.90,
        label="Evaluated candidates",
        zorder=30,
    )

    ax.scatter(
        restart_x,
        restart_y,
        restart_z,
        s=44,
        c="#8fd3ff",
        marker="^",
        edgecolors="#1f2937",
        linewidths=0.65,
        depthshade=False,
        alpha=0.98,
        zorder=40,
    )

    ax.scatter(
        hf_x,
        hf_y,
        hf_z,
        s=38,
        c="#0b3c8c",
        edgecolors="white",
        linewidths=0.5,
        depthshade=False,
        alpha=0.98,
        label="Higher-fidelity candidates",
        zorder=50,
    )

    ax.scatter(
        refine_x,
        refine_y,
        refine_z,
        s=32,
        c="#0b3c8c",
        edgecolors="#c7d2fe",
        linewidths=0.4,
        depthshade=False,
        alpha=0.86,
        zorder=55,
    )

    ax.scatter(
        elite_x,
        elite_y,
        elite_z,
        s=68,
        c="#f2c14e",
        edgecolors="#6b4e00",
        linewidths=0.7,
        depthshade=False,
        alpha=1.0,
        label="Elite candidates",
        zorder=70,
    )

    # 最优星号明显上抬，避免被谷底附近的曲面挡住
    ax.scatter(
        [x_best],
        [y_best],
        [z_best + 0.55],
        s=380,
        c="#ffd700",
        marker="*",
        edgecolors="#5c4200",
        linewidths=1.1,
        depthshade=False,
        label="Global optimum",
        zorder=100,
    )

    # 竖向虚线说明星号对应的真实最优点位置
    ax.plot(
        [x_best, x_best],
        [y_best, y_best],
        [z_best + 0.05, z_best + 0.55],
        color="#5c4200",
        linewidth=1.1,
        linestyle="--",
        alpha=0.85,
        zorder=90,
    )

    arrow_start = (x_best + 1.15, y_best - 1.20, z_best + 1.55)
    arrow_end = (x_best + 0.06, y_best - 0.05, z_best + 0.55)
    add_3d_arrow(ax, arrow_start, arrow_end, color="#222222", lw=1.2)

    # # 3D 虚线连接线保留
    # dimension_labels_3d = [
    #     ("seq_len (discrete)", -4.05, -3.85, 3.95),
    #     ("d_model (discrete)", -4.25, -2.20, 4.42),
    #     ("n_heads (discrete)", -4.30, 0.00, 4.55),
    #     ("e_layers (discrete)", -4.18, 2.15, 4.35),
    #     ("d_ff (discrete)", -3.80, 3.90, 3.95),
    #     ("dropout (continuous)", 3.70, 3.90, 4.05),
    #     ("learning_rate (log scale)", 4.18, 2.05, 4.36),
    #     ("batch_size (discrete)", 4.35, -0.15, 4.52),
    #     ("csg_reduction (conditional)", 4.16, -2.25, 4.20),
    # ]

    # for _, tx, ty, tz in dimension_labels_3d:
    #     anchor_x = np.clip(tx * 0.62, -3.7, 3.7)
    #     anchor_y = np.clip(ty * 0.62, -3.7, 3.7)
    #     anchor_z = objective_function(anchor_x, anchor_y) + 0.25
    #     ax.plot(
    #         [tx * 0.92, anchor_x],
    #         [ty * 0.92, anchor_y],
    #         [tz - 0.10, anchor_z],
    #         linestyle="--",
    #         linewidth=0.65,
    #         color="#94a3b8",
    #         alpha=0.72,
    #         zorder=20,
    #     )

    # # 文字说明放到 2D 图层，不参与 3D 深度排序
    # label_positions_2d = [
    #     ("seq_len (discrete)", 0.03, 0.22, "left"),
    #     ("d_model (discrete)", 0.03, 0.36, "left"),
    #     ("n_heads (discrete)", 0.03, 0.50, "left"),
    #     ("e_layers (discrete)", 0.03, 0.64, "left"),
    #     ("d_ff (discrete)", 0.06, 0.78, "left"),
    #     ("dropout (continuous)", 0.97, 0.78, "right"),
    #     ("learning_rate (log scale)", 0.97, 0.64, "right"),
    #     ("batch_size (discrete)", 0.97, 0.50, "right"),
    #     ("csg_reduction (conditional)", 0.97, 0.36, "right"),
    # ]

    for label, x2d, y2d, ha in label_positions_2d:
        ax.text2D(
            x2d,
            y2d,
            label,
            transform=ax.transAxes,
            color="#334155",
            fontsize=7.7,
            ha=ha,
            va="center",
            zorder=100000,
            clip_on=False,
            bbox=dict(
                facecolor="white",
                edgecolor="#cbd5e1",
                linewidth=0.4,
                alpha=0.86,
                pad=2.2,
            ),
        )

    ax.set_xlabel("Latent hyperparameter direction 1", labelpad=10)
    ax.set_ylabel("Latent hyperparameter direction 2", labelpad=10)
    ax.set_zlabel("Objective value", labelpad=8)

    ax.set_title(
        "Searching for the Optimal Configuration in a High-Dimensional Mixed Hyperparameter Space",
        pad=18,
        fontweight="semibold",
    )

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_zlim(np.nanmin(Z) - 0.08, np.nanmax(Z) + 1.35)

    ax.view_init(elev=28, azim=-58)

    ax.xaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.yaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.zaxis.pane.set_facecolor((1, 1, 1, 1))
    ax.xaxis.pane.set_edgecolor("#d1d5db")
    ax.yaxis.pane.set_edgecolor("#d1d5db")
    ax.zaxis.pane.set_edgecolor("#d1d5db")

    ax.grid(True)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["color"] = (0.82, 0.86, 0.90, 0.55)
        axis._axinfo["grid"]["linewidth"] = 0.55
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.2

    cbar = fig.colorbar(surface, ax=ax, shrink=0.62, aspect=18, pad=0.08)
    cbar.set_label("Objective value (Validation MSE / Key-MSE)", rotation=270, labelpad=16)
    cbar.outline.set_linewidth(0.6)

    legend_handles = [
        Patch(
            facecolor=cm.ScalarMappable(cmap=cmap).to_rgba(0.28),
            edgecolor="none",
            label="Objective surface",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#8fd3ff",
            markeredgecolor="white",
            markersize=7,
            label="Evaluated candidates",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#0b3c8c",
            markeredgecolor="white",
            markersize=7,
            label="Higher-fidelity candidates",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#f2c14e",
            markeredgecolor="#6b4e00",
            markersize=8,
            label="Elite candidates",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#ffd700",
            markeredgecolor="#5c4200",
            markersize=13,
            label="Global optimum",
        ),
        Patch(
            facecolor="#8a8f98",
            edgecolor="none",
            alpha=0.42,
            label="Invalid / infeasible region",
        ),
    ]

    # 图例放到 figure 层，避免被 3D axes 遮挡
    legend = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.02),
        ncol=3,
        frameon=True,
        fancybox=False,
        framealpha=0.96,
        borderpad=0.55,
        columnspacing=0.8,
    )
    legend.set_zorder(100000)
    legend.get_frame().set_edgecolor("#d1d5db")
    legend.get_frame().set_linewidth(0.8)

    # 给文字、colorbar、图例留出空间
    fig.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.20)

    fig.savefig(
        "hfabc_multidimensional_optimization_landscape.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    plt.show()


if __name__ == "__main__":
    main()