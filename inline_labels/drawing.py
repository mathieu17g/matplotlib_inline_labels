from matplotlib import pyplot as plt
from matplotlib.transforms import Transform, Affine2D
from matplotlib.container import ErrorbarContainer
from mpl_toolkits.axes_grid1 import Divider, Size
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from typing import Any
import shapely as shp
from .datatypes import (
    Labelled_Lines_Geometric_Data_Dict,
    Labels_PRcs,
    Label_Inlining_Solutions,
)
from .processing import get_box_rot_and_trans_function
from math import degrees


def get_dbg_axes(ax: Axes) -> tuple[Axes, Axes]:
    """Build a debug figure with to axes :
    - ax_data with lines build on data and inline labels
    - ax_geoms with the geometries used in label placement
    """
    #! It is assumed here that ax.get_figure() has already been drawn in calling function
    plt.close("all")
    # Determine original Axes dimensions,
    # TODO delete transformation to fig coordinates if only ax_bbox ratio is needed (see below) -> after an observation period starting with v0.1.6
    ax_bbox = ax.get_window_extent().transformed(
        ax.figure.dpi_scale_trans.inverted()  # pyright: ignore[reportOptionalMemberAccess]
    )

    fig_dbg = plt.figure(
        dpi=ax.get_figure().get_dpi(),  # pyright: ignore[reportOptionalMemberAccess]
        figsize=(
            ax_bbox.width * 3.2,
            ax_bbox.height * 1.4,
        ),  # Tweak for trying to proprely display Axes in tkAgg backend
    )

    pos = (0, 0, 1, 1)  # Position of the grid in the figure
    horiz = [Size.Fixed(1), Size.Fixed(ax_bbox.width), Size.Fixed(1), Size.Fixed(ax_bbox.width), Size.Fixed(1)]
    vert = [Size.Fixed(1), Size.Fixed(ax_bbox.height), Size.Fixed(1)]
    divider = Divider(fig_dbg, pos, horiz, vert, aspect=False, anchor="W")

    # Add the two Axes for debug drawing
    # TODO maybe have to handle more specific characteristics from the original Axe beyond axis scales and limits
    ax_data = fig_dbg.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=1, ny=1),
        xscale=ax.get_xscale(),
        yscale=ax.get_yscale(),
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
    )
    ax_geoms = fig_dbg.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=3, ny=1),
        xscale=ax.get_xscale(),
        yscale=ax.get_yscale(),
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
    )

    for _, sub_ax in enumerate([ax_data, ax_geoms]):
        for line in retrieve_lines_and_labels(ax)[0]:
            sub_ax.plot(
                *(line.get_data()), linewidth=line.get_linewidth(), label=line.get_label()
            )
        # Rotate x-axis labels manually since autofmt_xdate does not work for multiple axes which are not separated into subplots,
        # Solution found at https://stackoverflow.com/questions/48078540/matplotlib-autofmt-xdate-fails-to-rotate-x-axis-labels-after-axes-cla-is-c
        for label in sub_ax.get_xticklabels():
            label.set_horizontalalignment("right")
            label.set_rotation(30)

    ax_data.axis(ax.axis())
    ax_geoms.sharex(ax_data)

    fig_dbg.canvas.draw()
    return ax_data, ax_geoms


def retrieve_lines_and_labels(ax: Axes) -> tuple[list[Line2D], list[str]]:
    """Retrieves line-like objects from an Axes and the corresponding labels"""
    linelikeHandles, linelikeLabels = [], []
    allHandles, allLabels = ax.get_legend_handles_labels()
    for handle, label in zip(allHandles, allLabels):
        if isinstance(handle, ErrorbarContainer):
            line = handle.lines[0]
        elif isinstance(handle, Line2D):
            line = handle
        else:
            continue
        linelikeHandles.append(line)
        linelikeLabels.append(label)
    return linelikeHandles, linelikeLabels


# //def get_axe_aspect(ax: Axes) -> float:
# //    """Computes the drawn data aspect radio of an Axe to circumvemt 'auto' value
# //    returned by matplotlib.axes.Axes.get_aspect.
# //    Solution found at https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes
# //    """
# //    # Total figure size
# //    (
# //        figW,
# //        figH,
# //    ) = ax.get_figure().get_size_inches()  # pyright: ignore[reportOptionalMemberAccess]
# //    _, _, w, h = ax.get_position().bounds  # Axis size on figure
# //    disp_ratio = (figH * h) / (figW * w)  # Ratio of display units
# //    # Ratio of data units. Negative over negative because of the order of subtraction
# //    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
# //    return disp_ratio / data_ratio


def get_geom2disp_trans(ax: Axes) -> Transform:
    """Returns a transformation from Axe coordinates scaled to be cartesian on display,
    to display coordinates"""
    width, height = ax.get_window_extent().size
    return Affine2D().scale(1, width / height) + ax.transAxes


def plot_geometric_line_chunks(ax_geoms: Axes, ld: Labelled_Lines_Geometric_Data_Dict):
    """Plot curves as geometric objects for DEBUG"""
    # Remove all orginal lines from ax_geoms
    for line in ax_geoms.get_lines():
        line.remove()
    # Plot all the line chunks for all labels
    for label in list(ld):
        for lc_idx, lcg in enumerate(ld[label].lcgl):
            dbg_line_color = ax_geoms.plot(
                *(shp.get_coordinates(lcg.lc).T),
                # label=(label + f" - chunk #{lc_idx}"),
                transform=get_geom2disp_trans(ax_geoms),
                clip_on=False,
                linewidth=0.5,
                linestyle="dashed",
                zorder=5,
            )
            # Add the outline of the original line using the buffered version of the line
            # chunk
            ax_geoms.plot(
                *(shp.get_coordinates(shp.boundary(lcg.lcb)).T),
                # Adding a leading underscore prevent the artist to be included in
                # automatic legend
                label=("_" + label + f" - buffered chunk #{lc_idx}"),
                color=dbg_line_color[0].get_color(),
                transform=get_geom2disp_trans(ax_geoms),
                clip_on=False,
                linewidth=0.5,
                zorder=5,
            )

    if (fig := ax_geoms.get_figure()) is not None:
        fig.canvas.draw()


def plot_labels_PRcs(
    ax_geoms: Axes,
    ld: Labelled_Lines_Geometric_Data_Dict,
    gl_PRcs: Labels_PRcs,
    sep_levels: list[float],
):
    """Plot label Position/Rotation candidates with separation around"""
    #! Case where sep is None to be handled !!!
    bf_colors = {1.0: "tab:green", 0.5: "tab:olive", 0.0: "tab:orange", -1.0: "red"}
    assert set(sep_levels + [-1.0]) == set(list(bf_colors))
    bf_markersize = {1.0: 30, 0.5: 20, 0: 10, -1: 5}
    bf_zorder = {1.0: 0.2, 0.5: 0.3, 0: 0.4, -1: 0.1}
    bf_labels = {
        1.0: "Good separation",
        0.5: "Medium separation",
        0.0: "Tight separation",
        -1.0: "No separation",
    }

    for label in list(ld):
        # Plot label's position candidates with different label colors
        for bf in sep_levels + [-1]:
            if [
                prc
                for prc in gl_PRcs[label]
                if prc.rot is not None and prc.sep is not None and prc.sep == bf
            ]:
                X, Y, R = zip(
                    *(
                        (prc.pos.x, prc.pos.y, prc.rot)
                        for prc in gl_PRcs[label]
                        if prc.rot is not None and prc.sep is not None and prc.sep == bf
                    ),
                    strict=True,
                )
                ax_geoms.plot(
                    X,
                    Y,
                    marker="o",
                    color=bf_colors[bf],
                    markeredgewidth=0.0,
                    alpha=0.6,
                    markersize=bf_markersize[bf],
                    ls="",
                    transform=get_geom2disp_trans(ax_geoms),
                    zorder=bf_zorder[bf],
                    label=bf_labels[bf],
                )
                # Add small ticks on outer marker to hint the exact position of candidates
                # if bf == list(bf_colors)[0]:
                for x, y, r in zip(X, Y, R):
                    ax_geoms.plot(
                        x,
                        y,
                        marker=(2, 0, degrees(r)),
                        color="k",
                        markersize=bf_markersize[0],
                        markeredgewidth=0.5,
                        ls="",
                        transform=get_geom2disp_trans(ax_geoms),
                        zorder=bf_zorder[bf],
                    )

        # Plot all initial candidates as a backdrop
        lpc_list = [
            lpc
            for partial_lpc_list in [lcg.pcl for lcg in ld[label].lcgl]
            for lpc in partial_lpc_list
        ]
        X = [lpc.x for lpc in lpc_list]
        Y = [lpc.y for lpc in lpc_list]
        ax_geoms.plot(
            X,
            Y,
            linewidth=0,
            marker="o",
            markerfacecolor="none",
            markeredgecolor="tab:pink",
            markersize=bf_markersize[0],
            ls="",
            transform=get_geom2disp_trans(ax_geoms),
            zorder=bf_zorder[-1] / 2,
            label="Candidates from preprocessing",
        )

    # Build partial custom legend for ax_geoms
    partial_ax_geoms_legend = []
    for bf in sep_levels + [-1]:
        partial_ax_geoms_legend.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color=bf_colors[bf],
                markeredgewidth=0.0,
                alpha=0.6,
                markersize=bf_markersize[bf],
                ls="",
                transform=get_geom2disp_trans(ax_geoms),
                zorder=bf_zorder[bf],
                label=bf_labels[bf],
            )
        )
    partial_ax_geoms_legend.append(
        Line2D(
            [0],
            [0],
            marker="o",
            markerfacecolor="none",
            markeredgecolor="tab:pink",
            markersize=bf_markersize[0],
            ls="",
            transform=get_geom2disp_trans(ax_geoms),
            zorder=bf_zorder[-1] / 2,
            label="Candidates from preprocessing",
        )
    )

    ax_geoms.legend(  # pyright: ignore[reportPossiblyUnboundVariable]
        handles=partial_ax_geoms_legend,  # pyright: ignore[reportPossiblyUnboundVariable]
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
        handletextpad=1,
        # labelspacing = 0.5,
    )


def draw_inlined_labels(
    ax: Axes,
    l_text_kwarg: dict[str, Any],
    linelikeHandles: list[Line2D],
    linelikeLabels: list[str],
    lis: Label_Inlining_Solutions,
):
    for label in list(lis):
        trans_geom2data = get_geom2disp_trans(ax) + ax.transData.inverted()
        l_x, l_y = trans_geom2data.transform((lis[label].cpt.x, lis[label].cpt.y))
        # Plot labels on ax
        ax.text(
            l_x,
            l_y,
            label,
            color=linelikeHandles[linelikeLabels.index(label)].get_color(),
            backgroundcolor=ax.get_facecolor(),
            horizontalalignment="center",
            verticalalignment="center",
            rotation=degrees(lis[label].rot),
            bbox=dict(
                boxstyle="square, pad=0.3",
                mutation_aspect=1 / 10,
                fc=ax.get_facecolor(),
                lw=0,
            ),
            **l_text_kwarg,
        )


def draw_dbg_inlined_labels(
    ax: Axes,
    l_text_kwarg: dict[str, Any],
    ax_data: Axes,
    ax_geoms: Axes,
    linelikeHandles: list[Line2D],
    linelikeLabels: list[str],
    data_linelikeHandles: list[Line2D],
    data_linelikeLabels: list[str],
    ld: Labelled_Lines_Geometric_Data_Dict,
    lis: Label_Inlining_Solutions,
):
    for label in list(lis):
        trans_geom2data = get_geom2disp_trans(ax) + ax.transData.inverted()
        l_x, l_y = trans_geom2data.transform((lis[label].cpt.x, lis[label].cpt.y))
        # Plot labels on ax
        labelText = ax.text(
            l_x,
            l_y,
            label,
            color=linelikeHandles[linelikeLabels.index(label)].get_color(),
            backgroundcolor=ax.get_facecolor(),
            horizontalalignment="center",
            verticalalignment="center",
            rotation=degrees(lis[label].rot),
            bbox=dict(
                boxstyle="square, pad=0.3",
                mutation_aspect=1 / 10,
                fc=ax.get_facecolor(),
                lw=0,
            ),
            **l_text_kwarg,
        )
        fprop = labelText.get_fontproperties()
        # Plot labels' boxes on ax_data and chosen labels' centers on ax_geoms
        ax_data.text(  # pyright: ignore[reportPossiblyUnboundVariable]
            l_x,
            l_y,
            label,
            fontproperties=fprop,
            color=data_linelikeHandles[  # pyright: ignore[reportPossiblyUnboundVariable]
                data_linelikeLabels.index(  # pyright: ignore[reportPossiblyUnboundVariable]
                    label
                )
            ].get_color(),
            backgroundcolor=ax_data.get_facecolor(),  # pyright: ignore[reportPossiblyUnboundVariable]
            horizontalalignment="center",
            verticalalignment="center",
            rotation=degrees(lis[label].rot),
            bbox=dict(
                boxstyle="square, pad=0.3",
                mutation_aspect=1 / 10,
                fc=ax_data.get_facecolor(),  # pyright: ignore[reportPossiblyUnboundVariable]
                ec=data_linelikeHandles[  # pyright: ignore[reportPossiblyUnboundVariable]
                    data_linelikeLabels.index(  # pyright: ignore[reportPossiblyUnboundVariable]
                        label
                    )
                ].get_color(),
                lw=0.1,
            ),
            **l_text_kwarg,
        )
        ax_geoms.plot(  # pyright: ignore[reportPossiblyUnboundVariable]
            lis[label].cpt.x,
            lis[label].cpt.y,
            marker="o",
            color="k",
            markersize=10,
            ls="",
            transform=get_geom2disp_trans(
                ax_geoms  # pyright: ignore[reportPossiblyUnboundVariable]
            ),
        )
        # Get the label box in geom coordinates
        get_lbox_geom = get_box_rot_and_trans_function(ld[label].boxd)
        rtl_box = shp.boundary(
            get_lbox_geom(  # pyright: ignore[reportPossiblyUnboundVariable]
                "box", lis[label].cpt, lis[label].rot
            )
        )
        # Plot the label box used in algorithm
        ax_geoms.plot(  # pyright: ignore[reportPossiblyUnboundVariable]
            *(shp.get_coordinates(rtl_box).T),
            color="k",
            linewidth=0.5,
            transform=get_geom2disp_trans(
                ax_geoms  # pyright: ignore[reportPossiblyUnboundVariable]
            ),
        )


def add_noninlined_labels_legend(
    ax: Axes,
    l_text_kwarg: dict[str, Any],
    linelikeHandles: list[Line2D],
    linelikeLabels: list[str],
    legend_labels: list[str],
):
    ax.legend(
        handles=[linelikeHandles[linelikeLabels.index(label)] for label in legend_labels],
        labels=legend_labels,
        facecolor=ax.get_facecolor(),
        **l_text_kwarg,
    )


def add_dbg_noninlined_labels_legend(
    ax: Axes,
    debug: bool,
    l_text_kwarg: dict[str, Any],
    ax_data: Axes,
    linelikeHandles: list[Line2D],
    linelikeLabels: list[str],
    data_linelikeHandles: list[Line2D],
    data_linelikeLabels: list[str],
    legend_labels: list[str],
):
    if debug:
        ax_data.legend(  # pyright: ignore[reportPossiblyUnboundVariable]
            handles=[
                data_linelikeHandles[  # pyright: ignore[reportPossiblyUnboundVariable]
                    data_linelikeLabels.index(  # pyright: ignore[reportPossiblyUnboundVariable]
                        label
                    )
                ]
                for label in legend_labels
            ],
            labels=legend_labels,
            **l_text_kwarg,
        )

    ax.legend(
        handles=[linelikeHandles[linelikeLabels.index(label)] for label in legend_labels],
        labels=legend_labels,
        facecolor=ax.get_facecolor(),
        **l_text_kwarg,
    )
