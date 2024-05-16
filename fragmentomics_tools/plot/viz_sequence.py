# MIT License
#
# Copyright (c) 2018 Kundaje Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# flake8: noqa

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_a(ax, base, left_edge, width, height, color):
    a_polygon_coords = [
        np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.7],
                [0.175, 0.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0],
                [0.5, 1.0],
                [0.5, 0.7],
                [0.825, 0.0],
            ]
        ),
        np.array(
            [
                [0.25, 0.425],
                [0.75, 0.425],
                [0.85, 0.275],
                [0.15, 0.275],
            ]
        ),
    ]
    for polygon_coords in a_polygon_coords:
        ax.add_patch(
            matplotlib.patches.Polygon(
                (np.array([width, height])[None, :] * polygon_coords + np.array([left_edge, base])[None, :]),
                facecolor=color,
                edgecolor=color,
            )
        )


def plot_c(ax, base, left_edge, width, height, color):
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65 * width, base + 0.5 * height],
            width=1.3 * width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65 * width, base + 0.5 * height],
            width=0.7 * 1.3 * width,
            height=0.7 * height,
            facecolor="white",
            edgecolor="white",
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + width, base],
            width=width,
            height=height,
            facecolor="white",
            edgecolor="white",
            fill=True,
        )
    )


def plot_g(ax, base, left_edge, width, height, color):
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65 * width, base + 0.5 * height],
            width=1.3 * width,
            height=height,
            facecolor=color,
            edgecolor=color,
        )
    )
    ax.add_patch(
        matplotlib.patches.Ellipse(
            xy=[left_edge + 0.65 * width, base + 0.5 * height],
            width=0.7 * 1.3 * width,
            height=0.7 * height,
            facecolor="white",
            edgecolor="white",
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + width, base],
            width=width,
            height=height,
            facecolor="white",
            edgecolor="white",
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.825 * width, base + 0.085 * height],
            width=0.174 * width,
            height=0.415 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.625 * width, base + 0.35 * height],
            width=0.374 * width,
            height=0.15 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )


def plot_t(ax, base, left_edge, width, height, color):
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge + 0.4 * width, base],
            width=0.2 * width,
            height=height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )
    ax.add_patch(
        matplotlib.patches.Rectangle(
            xy=[left_edge, base + 0.8 * height],
            width=width,
            height=0.2 * height,
            facecolor=color,
            edgecolor=color,
            fill=True,
        )
    )


def plot_n(ax, base, left_edge, width, height, color):
    n_diag_coords = np.array(
        [
            [0.2, 1.0],
            [1.0, 0.0],
            [0.8, 0.0],
            [0.0, 1.0],
        ]
    )

    ax.add_patch(
        matplotlib.patches.Polygon(
            (np.array([width, height])[None, :] * n_diag_coords + np.array([left_edge, base])[None, :]),
            facecolor=color,
            edgecolor=color,
        )
    )
    for left_pos in [0.0, 0.8]:
        ax.add_patch(
            matplotlib.patches.Rectangle(
                xy=[left_edge + left_pos * width, base],
                width=0.2 * width,
                height=height,
                facecolor=color,
                edgecolor=color,
                fill=True,
            )
        )


DEFAULT_COLORS = {0: "xkcd:green", 1: "xkcd:blue", 2: "xkcd:orange", 3: "xkcd:red", 4: "xkcd:grey"}
DEFAULT_PLOT_FUNCS = {0: plot_a, 1: plot_c, 2: plot_g, 3: plot_t, 4: plot_n}


def plot_weights_given_ax(
    ax,
    array,
    height_padding_factor=0.2,
    align_to=None,
    length_padding=1.0,
    subticks_frequency=None,
    highlight=None,  # defaults to an empty dictionary below
    colors=DEFAULT_COLORS,
    plot_funcs=DEFAULT_PLOT_FUNCS,
    x_start=0,
    x_end=None,
    min_height_to_plot=None,
):
    if highlight is None:
        highlight = {}
    if len(array.shape) == 3:
        array = np.squeeze(array)
    assert len(array.shape) == 2, array.shape
    if array.shape[0] == 4 and array.shape[1] != 4:
        array = array.transpose(1, 0)
    assert array.shape[1] == 4
    seq_len = array.shape[0]
    x_end = seq_len if x_end is None else x_end
    letter_width = (x_end - x_start) / seq_len

    max_pos_height = 0.0
    min_neg_height = 0.0
    heights_at_positions = []
    depths_at_positions = []
    # for i, x in enumerate(np.arange(x_start, x_end, letter_width)):
    for i in range(seq_len):
        x = i * letter_width
        # sort from smallest to highest magnitude
        acgt_vals = sorted(enumerate(array[i, :]), key=lambda x: abs(x[1]))
        # total_abs_height = sum(abs(letter[1]) for letter in acgt_vals)
        positive_height_so_far = 0.0
        negative_height_so_far = 0.0
        for letter in acgt_vals:
            plot_func = plot_funcs[letter[0]]
            color = colors[letter[0]]
            if letter[1] > 0:
                height_so_far = positive_height_so_far
                positive_height_so_far += letter[1]
            else:
                height_so_far = negative_height_so_far
                negative_height_so_far += letter[1]
            if min_height_to_plot is None or abs(letter[1]) > min_height_to_plot:
                plot_func(
                    ax=ax,
                    base=height_so_far,
                    left_edge=x + x_start,
                    width=letter_width,
                    height=letter[1],
                    color=color,
                )
        max_pos_height = max(max_pos_height, positive_height_so_far)
        min_neg_height = min(min_neg_height, negative_height_so_far)
        heights_at_positions.append(positive_height_so_far)
        depths_at_positions.append(negative_height_so_far)

    # now highlight any desired positions; the key of
    # the highlight dict should be the color
    for color in highlight:
        for start_pos, end_pos in highlight[color]:
            assert start_pos >= 0.0 and end_pos <= seq_len
            min_depth = np.min(depths_at_positions[start_pos:end_pos])
            max_height = np.max(heights_at_positions[start_pos:end_pos])
            ax.add_patch(
                matplotlib.patches.Rectangle(
                    xy=[start_pos, min_depth],
                    width=end_pos - start_pos,
                    height=max_height - min_depth,
                    edgecolor=color,
                    fill=False,
                )
            )

    ax.set_xlim(-length_padding + x_start, array.shape[0] + length_padding + x_start)
    if subticks_frequency is not None:
        ax.xaxis.set_ticks(np.arange(0.0 + x_start, array.shape[0] + 1 + x_start, subticks_frequency))
    height_padding = max(
        abs(min_neg_height) * (height_padding_factor),
        abs(max_pos_height) * (height_padding_factor),
    )
    if align_to is None:
        ax.set_ylim(min_neg_height - height_padding, max_pos_height + height_padding)
    elif align_to == "top":
        ax.set_ylim(min_neg_height - 2 * height_padding, max_pos_height)
    else:
        assert align_to == "bottom"
        ax.set_ylim(min_neg_height, max_pos_height + 2 * height_padding)


def plot_weights(
    array,
    figsize=(20, 2),
    height_padding_factor=0.2,
    length_padding=1.0,
    subticks_frequency=1.0,
    colors=DEFAULT_COLORS,
    plot_funcs=DEFAULT_PLOT_FUNCS,
    highlight={},
):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    plot_weights_given_ax(
        ax=ax,
        array=array,
        height_padding_factor=height_padding_factor,
        length_padding=length_padding,
        subticks_frequency=subticks_frequency,
        colors=colors,
        plot_funcs=plot_funcs,
        highlight=highlight,
    )
    plt.show()
