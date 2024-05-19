from inline_labels.utils import (
    # timer,
    esc,
    Timer,
)
from inline_labels.datatypes import (
    Line_Chunk_Geometries,
    Label_Rotation_Estimates_Dict,
    Label_PRcs,
    Labels_lcs_adjPRcs_groups,
    Labels_PRcs,
)
from inline_labels.drawing import (
    get_dbg_axes,
    retrieve_lines_and_labels,
    plot_geometric_line_chunks,
    plot_labels_PRcs,
    draw_inlined_labels,
    draw_dbg_inlined_labels,
    add_noninlined_labels_legend,
    add_dbg_noninlined_labels_legend,
)
from inline_labels.geometries import (
    get_axe_lines_geometries,
    get_axe_lines_widths,
    update_ld_with_label_text_box_dimensions,
)

from inline_labels.preprocessing import (
    update_ld_with_label_position_candidates,
)
from inline_labels.processing import (
    get_box_rot_and_trans_function,
    ROTATION_SAMPLES_NUMBER,
    SEP_LEVELS,
    evaluate_candidates,
)
from inline_labels.postprocessing import solselect_monocrit
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Literal, Optional
import numpy as np
from warnings import warn
from tqdm import tqdm
from contextlib import nullcontext


plt.close("all")


##############################################
### HELPER FUNCTIONS FOR LABEL POSITIONING ###
##############################################


#########################################################
# TODO create a transformation class transGeom linked   #
# TODO to the Axe from which geometries where extracted #
#########################################################


def get_label_rotation_estimates(
    lcg: Line_Chunk_Geometries,
) -> list[int]:
    """Converts labels' rotation estimates from radians to index of rotation samples list"""
    # Get rotation estimations in radians
    lre_rad = np.asarray(lcg.re)
    # Convert rotation estimation in radians to index in rotation samples list
    lre_idx = np.rint(
        (lre_rad + np.pi / 2) / (np.pi / ROTATION_SAMPLES_NUMBER),
        out=np.zeros(lre_rad.shape, np.int64),
        casting="unsafe",
    ).tolist()

    return lre_idx


######################################
#### LABELS POSITIONNING FUNCTION ####
######################################


def add_inline_labels(
    ax: Axes,
    ppf: float = 1,  # Position precision factor
    maxpos: int = 100,
    po: Literal["default"] = "default",  # Placement algorithm option
    with_overall_progress: bool = False,
    overall_progress_desc: str = "Labels placement",
    with_perlabel_progress: bool = False,
    debug: bool = False,
    preprocessing_curv_filter_mode: Literal["fast", "precise"] = "fast",
    **l_text_kwarg,
) -> Optional[Figure]:
    """For an axe made of Line2D plots, draw the legend label for each line on it (centered
    on one of its points) without any intersection with any other line or some of its own
    missing data

    WARNING: depending on the backend some overlapping errors may appear due to vector
    drawing from a discrete resolution. Visual errors disapear when viewed on a real image
    or by increasing the dpi.

    Args:
    - ax: Axe composed of Line2D objects
    - ppf: position precision factor, fraction of the label box height, itself depending on
    the font properties. This fraction of the label's box height is used as a sampling
    distance along the line chunk to define candidates for label's bounding box's center
    positionning
    - maxpos: maximum label's position candidates before preprocessing. Should be stricly
    positive
    - with_overall_progress: progress bar for whole axe's labels positionning
    - with_perlabel_progress: progress bar per label positionning
    - debug: draws a figure showing the algorithm intermediate results
    - **l_text_kwargs: text kwargs used for label drawing

    Returns: figure of the input Axe with labels drawn inline
    """
    # Reset debug timers
    for k in Timer.timers:
        Timer.timers[k] = 0.0

    fig = _add_inline_labels(
        ax,
        ppf,
        maxpos,
        po,
        with_overall_progress,
        overall_progress_desc,
        with_perlabel_progress,
        debug,
        preprocessing_curv_filter_mode,
        **l_text_kwarg,
    )
    if debug:
        print("Elapsed time for:")
        for k, v in ((k, v) for k, v in Timer.timers.items() if k != "add_inline_labels"):
            print("- " + esc(2) + esc("38;5;39") + k + esc(0) + f": {v:0.4f} seconds")
        print(
            "Total elapsed time for "
            + esc(1)
            + esc("38;5;39")
            + "add_inline_labels"
            + esc(0)
            + f": {Timer.timers["add_inline_labels"]:0.4f} seconds"
        )

    return fig


@Timer(name="add_inline_labels")
def _add_inline_labels(
    ax: Axes,
    ppf: float = 1,  # Position precision factor
    maxpos: int = 100,
    po: Literal["default"] = "default",  # Placement algorithm option
    with_overall_progress: bool = False,
    overall_progress_desc: str = "Labels placement",
    with_perlabel_progress: bool = False,
    debug: bool = False,
    preprocessing_curv_filter_mode: Literal["fast", "precise"] = "fast",
    **l_text_kwarg,
) -> Optional[Figure]:
    """For an axe made of Line2D plots, draw the legend label for each line on it (centered
    on one of its points) without any intersection with any other line or some of its own
    missing data

    WARNING: depending on the backend some overlapping errors may appear due to vector
    drawing from a discrete resolution. Visual errors disapear when viewed on a real image
    or by increasing the dpi.

    Args:
    - ax: Axe composed of Line2D objects
    - ppf: position precision factor, fraction of the label box height, itself depending on
    the font properties. This fraction of the label's box height is used as a sampling
    distance along the line chunk to define candidates for label's bounding box's center
    positionning
    - maxpos: maximum label's position candidates before preprocessing. Should be stricly
    positive
    - with_overall_progress: progress bar for whole axe's labels positionning
    - with_perlabel_progress: progress bar per label positionning
    - debug: draws a figure showing the algorithm intermediate results
    - **l_text_kwargs: text kwargs used for label drawing

    Returns: figure of the input Axe with labels drawn inline
    """

    # Draw the figure before anything else, #! Note that draw_idle is not enough
    # TODO: see if it can be detected that the figure has already been drawn in case of
    # TODO: multi axes figure
    if (fig := ax.get_figure()) is not None:
        fig.canvas.draw()

    # Build a debug figure
    if debug:
        ax_data, ax_geoms = get_dbg_axes(ax)

    #############################
    # Retrieve lines and labels #
    #############################

    # Retrieve linelike objects and their labels
    linelikeHandles, linelikeLabels = retrieve_lines_and_labels(ax)
    if debug:
        data_linelikeHandles, data_linelikeLabels = retrieve_lines_and_labels(
            ax_data  # pyright: ignore[reportPossiblyUnboundVariable]
        )

    if not linelikeLabels:
        warn("No line like object with label set")
        return ax.get_figure()

    ########################################################################
    # Transform curves into geometric object for label placement computing #
    ########################################################################

    # Build a dictionary of line's width per label
    ld_lw = get_axe_lines_widths(ax, linelikeHandles, linelikeLabels)
    # Build a dictionary of (line chunks and bufferd line chunk) list (in Axes coordinates)
    # for all labels
    ld = get_axe_lines_geometries(ax, linelikeHandles, linelikeLabels, ld_lw, debug)
    # Plot curves as geometric objects for DEBUG
    if debug:
        plot_geometric_line_chunks(
            ax_geoms, ld  # pyright: ignore[reportPossiblyUnboundVariable]
        )

    ####################
    # Labels placement #
    ####################

    # Clustered and unclustered PR candidates structures initialization for label
    graph_labels_lcs_adjPRcs_groups = Labels_lcs_adjPRcs_groups({})
    graph_labels_PRcs = Labels_PRcs({})

    ########################################################################
    # Identify labels' box geometry and position candidates per line chunk #
    ########################################################################
    assert maxpos > 0, "maxpos should be > 0"

    for label in linelikeLabels:
        # Get the label text bounding box and x sampling points
        update_ld_with_label_text_box_dimensions(
            ax, linelikeHandles, linelikeLabels, ld, label, **l_text_kwarg
        )
        # Find all label position (box center) candidates per label's line chunks
        update_ld_with_label_position_candidates(
            ppf, ld, label, maxpos, curv_filter_mode=preprocessing_curv_filter_mode
        )

    ##############################################################
    # Prepare overall progress context maanager if option chosen #
    ##############################################################
    if with_overall_progress:
        overall_candidates_number = sum(
            [len(lcg.pcl) for label in list(ld) for lcg in ld[label].lcgl]
        )
        overall_progress_cm = tqdm(
            total=overall_candidates_number,
            ascii=True,
            ncols=80,
            desc=overall_progress_desc,
            position=0,
            leave=True,
        )
    else:
        overall_progress_cm = nullcontext()

    #####################################
    # Pre estimation of label rotations #
    #####################################

    lre = Label_Rotation_Estimates_Dict(
        {
            label: [get_label_rotation_estimates(lcg) for lcg in ld[label].lcgl]
            for label in linelikeLabels
        }
    )

    #########################
    # Search for each label #
    #########################
    with overall_progress_cm as overall_pbar:
        for label in linelikeLabels:
            ###############################################################################
            # Create current label's box helper geometries with translation and rotation  #
            # helper functions                                                            #
            ###############################################################################

            get_lbox_geom = get_box_rot_and_trans_function(ld[label].boxd)

            ##################################
            # Iteration over each line chunk #
            ##################################

            # Iterate other each current label line chunks to find:
            # - for each x from the over sampled x data of the current label and within the
            #   current chunk x coordinates boundaries,
            # - for each rotation sample,
            # the distance between the current label box and the other lines + current
            # label line chunks boundaries

            # Clustered and unclustered PR candidates structures initialization for label
            graph_labels_lcs_adjPRcs_groups |= {label: dict[int, list[Label_PRcs]]({})}
            graph_labels_PRcs |= {label: Label_PRcs([])}

            # TODO: add an option to parallelize per line chunk computation for line chunk
            # TODO: with enough candidates
            for lc_idx, lcg in enumerate(ld[label].lcgl):
                if len(lcg.pcl) != 0:
                    evaluate_candidates(
                        ld,
                        label,
                        lc_idx,
                        with_perlabel_progress,
                        with_overall_progress,
                        lcg,
                        lre,
                        get_lbox_geom,
                        debug,
                        graph_labels_lcs_adjPRcs_groups,
                        graph_labels_PRcs,
                        overall_pbar,
                    )

    # Draw visual debug
    if debug:
        plot_labels_PRcs(
            ax_geoms,  # pyright: ignore[reportPossiblyUnboundVariable]
            ld,
            graph_labels_PRcs,
            SEP_LEVELS,
        )

    #############################
    # Find best label positions #
    #############################

    #! Default label best position algorithm: center of longuest contiguous label position
    #! candidates with the highest separation level found for earch line
    if po == "default":
        lis, legend_labels = solselect_monocrit(
            linelikeLabels, graph_labels_lcs_adjPRcs_groups, ld
        )

    # TODO: Add new algorithms

    #############################
    # Draw label for each label #
    #############################

    if not debug:
        draw_inlined_labels(ax, l_text_kwarg, linelikeHandles, linelikeLabels, lis)
    else:
        draw_dbg_inlined_labels(
            ax,
            l_text_kwarg,
            ax_data,  # pyright: ignore[reportPossiblyUnboundVariable]
            ax_geoms,  # pyright: ignore[reportPossiblyUnboundVariable]
            linelikeHandles,
            linelikeLabels,
            data_linelikeHandles,  # pyright: ignore[reportPossiblyUnboundVariable]
            data_linelikeLabels,  # pyright: ignore[reportPossiblyUnboundVariable]
            ld,
            lis,
        )

    # ? Add a legend for all handles which are not line like artists according to function
    # ? retrieve_lines_and_labels

    # Add legend with all labels than could not be positionned properly on their curve
    if legend_labels and debug:
        print(f"Unplaced labels : {legend_labels}")
    if legend_labels and (len(legend_labels) != len(linelikeLabels)):
        if not debug:
            add_noninlined_labels_legend(
                ax, l_text_kwarg, linelikeHandles, linelikeLabels, legend_labels
            )
        else:
            add_dbg_noninlined_labels_legend(
                ax,
                debug,
                l_text_kwarg,
                ax_data,  # pyright: ignore[reportPossiblyUnboundVariable]
                linelikeHandles,
                linelikeLabels,
                data_linelikeHandles,  # pyright: ignore[reportPossiblyUnboundVariable]
                data_linelikeLabels,  # pyright: ignore[reportPossiblyUnboundVariable]
                legend_labels,
            )

    if debug:
        return ax_data.get_figure()  # pyright: ignore[reportPossiblyUnboundVariable]
    else:
        return ax.get_figure()
