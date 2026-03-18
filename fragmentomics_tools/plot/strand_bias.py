import colorsys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Custom red–blue colormap, less brown, more red
# ---------------------------------------------------------
def make_red_blue_cmap():
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "redblue_strong",
        [
            (0.0, (0.0, 0.2, 1.0)),   # deep blue
            (0.5, (0.95, 0.95, 0.95)),  # light neutral
            (1.0, (1.0, 0.0, 0.0)),   # pure red
        ]
    )


# ---------------------------------------------------------
# Symmetric luminance inversion so middle bins darkest
# ---------------------------------------------------------
def inverted_symmetric_luminance_colors(cmap, n, min_l=0.85, max_l=0.25):
    idx = np.linspace(-1, 1, n)
    weights = 1 - np.abs(idx)                    # peaks in center
    lightness = min_l - (min_l - max_l) * weights

    out = []
    for i in range(n):
        rgb = cmap(i / max(n - 1, 1))[:3]
        h, l, s = colorsys.rgb_to_hls(*rgb)
        out.append(colorsys.hls_to_rgb(h, lightness[i], s))
    return out


def plot_exp_vs_bias(expression_vs_strand_bias_df, title="Expression vs Strand Bias"):
    tmp = expression_vs_strand_bias_df

    labels    = tmp.columns.get_level_values("label").unique()
    fl_ranges = tmp.columns.get_level_values("fl_bnd").unique()
    n_fl      = len(fl_ranges)

    # Colors + styles
    cmap   = make_red_blue_cmap()
    colors = inverted_symmetric_luminance_colors(cmap, n_fl)
    base_linestyles = [":", "-.", "--", "-", "-", "--", "-.", ":"]
    linestyles = [base_linestyles[i % len(base_linestyles)] for i in range(n_fl)]

    # Typography
    PANEL_TITLE_SIZE = 24
    MAIN_TITLE_SIZE  = 38
    AXIS_LABEL_SIZE  = 24
    TICK_LABEL_SIZE  = 20
    LEGEND_FONT_SIZE = 18

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams.update({
        "axes.labelsize": AXIS_LABEL_SIZE,
        "axes.titlesize": PANEL_TITLE_SIZE,
        "xtick.labelsize": TICK_LABEL_SIZE,
        "ytick.labelsize": TICK_LABEL_SIZE,
        "axes.linewidth": 1.5,
        "lines.linewidth": 2.2,
    })

    # ---------------------------------------------------------
    # 2-column layout, wide panels
    # ---------------------------------------------------------
    n_labels = len(labels)
    ncols = 2
    nrows = int(np.ceil(n_labels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(22, 7 * nrows),
        sharex=True,
        sharey=True,
    )
    axes = np.array(axes).reshape(-1)

    # Hide unused axes in last row
    for ax in axes[n_labels:]:
        ax.set_visible(False)

    # ---------------------------------------------------------
    # Plot each panel
    # ---------------------------------------------------------
    for ax, label in zip(axes, labels):
        sub    = tmp.xs(label, axis=1, level="label")
        x_vals = sub.index.values.astype(float)

        for j, fl in enumerate(sub.columns):
            y_vals = sub[fl].values

            ax.plot(
                x_vals,
                y_vals,
                color=colors[j],
                linestyle=linestyles[j],
                alpha=1.0,
                label=f"{fl[0]}–{fl[1]} bp"
            )

        # ---------------------------------------------------------
        # Add horizontal dotted guide-lines
        # ---------------------------------------------------------
        for y in [0.1, 0.2, 0.3]:
            ax.axhline(y, color="lightgray", linestyle=":", linewidth=1)

        ax.axhline(0, color="black", linewidth=1, alpha=0.4)
        ax.set_title(label, pad=14)

    # ---------------------------------------------------------
    # Tight layout FIRST (so positions are settled)
    # ---------------------------------------------------------
    fig.tight_layout(rect=[0.04, 0.10, 0.96, 0.88], h_pad=2.8)
    fig.canvas.draw_idle()

    # ---------------------------------------------------------
    # Geometry of the panel grid
    # ---------------------------------------------------------
    visible_axes = [ax for ax in axes if ax.get_visible()]
    boxes = [ax.get_position() for ax in visible_axes]

    left_ax  = axes[0]
    right_ax = axes[1]

    left_box  = left_ax.get_position()
    right_box = right_ax.get_position()

    # Horizontal center of panel grid (for legend & x-label)
    mid_x = 0.5 * (left_box.x0 + right_box.x1)

    # Vertical extents of grid (for y-label)
    top    = max(b.y1 for b in boxes)
    bottom = min(b.y0 for b in boxes)
    mid_y  = 0.5 * (top + bottom)

    # x position for y-label (just left of leftmost axes)
    left_edge = min(b.x0 for b in boxes)
    ylab_x = left_edge - 0.06

    # ---------------------------------------------------------
    # Global legend – centered over panels, moved up a bit
    # ---------------------------------------------------------
    handles, labels_legend = axes[0].get_legend_handles_labels()

    for ax in axes:
        leg = ax.get_legend()
        if leg:
            leg.remove()

    fig.legend(
        handles,
        labels_legend,
        title="Fragment length",
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
        ncol=n_fl,
        loc="upper center",
        bbox_to_anchor=(mid_x, 0.955),
        frameon=True,
        borderpad=0.4,
    )

    # ---------------------------------------------------------
    # X-axis label – lowered slightly
    # ---------------------------------------------------------
    xlab_y = bottom - 0.04
    fig.text(
        mid_x,
        xlab_y,
        "log10(Blood Gene Expression)",
        fontsize=AXIS_LABEL_SIZE,
        ha="center",
        va="center"
    )

    # ---------------------------------------------------------
    # Y-axis label (two lines, properly stacked)
    # ---------------------------------------------------------
    line1_size = (AXIS_LABEL_SIZE + 4) - 4
    line2_size = AXIS_LABEL_SIZE - 4

    ylab_x = 0.02
    y_center = 0.5

    fig.text(
        ylab_x,
        y_center,
        "Strand Bias",
        fontsize=line1_size,
        rotation=90,
        ha="center",
        va="center"
    )

    fig.text(
        ylab_x + 0.02,
        y_center,
        "(anti-sense count - sense count)/(anti-sense count + sense count)",
        fontsize=line2_size,
        rotation=90,
        ha="center",
        va="center"
    )

    fig.suptitle(
        title,
        fontsize=MAIN_TITLE_SIZE,
        y=0.995
    )

    plt.show()
