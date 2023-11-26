# from IPython import get_ipython
# if (IS := get_ipython()) is not None: IS.run_line_magic("matplotlib", "inline")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.transforms import Transform, Affine2D
from mpl_toolkits.axes_grid1 import Divider, Size
from typing import Any, Tuple, Literal, Union, Callable, Optional, TypedDict
from nptyping import NDArray, Shape, Float, Object

# from datetime import datetime as dt
from matplotlib.dates import date2num  # , DateConverter
import numpy as np
from numpy import ma

# from numpy import linalg as LA
from functools import lru_cache
import warnings
import shapely as shp
from collections import namedtuple
from tqdm.auto import tqdm
import math
from contextlib import nullcontext
from operator import sub

#! DEBUG IMPORTS
import functools
import time

#! END OF DEBUG IMPORTS

plt.close("all")

#####################################################
# Data structures used by label placement algorithm #
#####################################################

Point2D = namedtuple("Point2D", ["x", "y"])  # Named tuple for label center position candidates
Line_Chunk_Geometries = TypedDict(
    "Line_Chunk_Geometries",  # Geometries in cartesian Axe coordinates
    {
        "lc": Union[shp.linestrings, shp.points],  # Line chunk geometry
        "lcb": shp.polygons,  # Line chunk buffered with Line2D width
        "lcp": Union[shp.linestrings, shp.points],  # Line chunk prepared geometry
        "lcbp": shp.polygons,  # Line chunk buffered with Line2D width and prepared
        "pcl": list[Point2D],  # Label box center candidate list
    },
)
Label_Box_Dimensions = TypedDict("Label_Box_Dimensions", {"box_w": float, "box_h": float})
Label_Geom_Data = TypedDict("Label_Data", {"lcd": dict[int, Line_Chunk_Geometries], "boxd": Label_Box_Dimensions})
Plot_Labels_Geom_Data = dict[str, Label_Geom_Data]
Rotation_Search_State = TypedDict(
    "Rotation_Search_State",  # Mono direction label rotation angle search state
    {
        "search": bool,  # Continue search
        "first": float,  # First valid angle found (for valid look at algorithm)
        "last": float,  # Last valid angle found (for valid look at algorithm)
    },
)
Rotation_BiSearch_State = dict[
    Literal[1, -1], Rotation_Search_State
]  # Bi direction (1: ccw, -1:cw) label rotation angle search state
LC_Candidate_With_LR_Candidates = TypedDict(
    "LC_Candidate_With_LR_Candidates",
    {  # Label Center candidate with multiple Label Rotation Candidates
        "c_geom": Point2D,  # Label center position candidate 2D coordinates
        "theta_candidates": dict[
            float,  # Label rotation angle value in radians (theta)
            float,  # Label alignment error with its curve
        ],
    },
)
MLS_LC_Candidates_With_LR_Candidates = dict[  # Multi level seperation label center candidates with multi rotation candidates
    float,  # Separation level as buffer factor to apply on label's box when looking for intersections
    dict[
        int,  # Label center position candidate index within a sampled line chunk (list of points)
        LC_Candidate_With_LR_Candidates,
    ],
]
LCR_Candidate = TypedDict(
    "LCR_Candidate",
    {  # Label Center and Rotation candidate
        "c_geom": Point2D,  # Label center position candidate 2D coordinates
        "theta": float,  # Label rotation candidate (radians)
        "align_err": float,  # Label alignment error with its curve
    },
)
MLS_LCR_Candidates = dict[  # Multi level seperation label center and rotation candidates
    float,  # Separation level as buffer factor applied on label's box when looking for intersections
    dict[int, LCR_Candidate],  # Label center position candidate index within a sampled line chunk (list of points)
]


###############################################
### HELPER FUNCTIONS FOR LABEL POSITIONING ###
###############################################


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


def get_axe_aspect(ax: Axes) -> float:
    """Computes the drawn data aspect radio of an Axe to circumvemt 'auto' value
    returned by matplotlib.axes.Axes.get_aspect.
    Solution found at https://stackoverflow.com/questions/41597177/get-aspect-ratio-of-axes
    """
    figW, figH = ax.get_figure().get_size_inches()  # Total figure size
    _, _, w, h = ax.get_position().bounds  # Axis size on figure
    disp_ratio = (figH * h) / (figW * w)  # Ratio of display units
    data_ratio = sub(*ax.get_ylim()) / sub(
        *ax.get_xlim()
    )  # Ratio of data units. Negative over negative because of the order of subtraction
    return disp_ratio / data_ratio


def get_dbg_axes(ax: Axes, fig_for_debug: Figure) -> Tuple[Axes, Axes]:
    """Build a debug figure with to axes :
    - ax_data with lines build on data and inline labels
    - ax_geoms with the geometries used in label placement
    """
    ax.get_figure().canvas.draw()
    plt.close("all")
    # Determine original Axes dimensions,
    if fig_for_debug is None:
        ax_bbox = ax.get_tightbbox().transformed(ax.figure.dpi_scale_trans.inverted())
        ax_box_dim = max(ax_bbox.width, ax_bbox.height)
    else:
        # Solution found at https://stackoverflow.com/questions/19306510/determine-matplotlib-axis-size-in-pixels
        ax_bbox = ax.get_window_extent().transformed(fig_for_debug.dpi_scale_trans.inverted())
        ax_box_dim = max(ax_bbox.width, ax_bbox.height)
    fig_dbg = plt.figure(
        figsize=((ax_box_dim), (ax_box_dim)), dpi=ax.get_figure().get_dpi()
    )  # , dpi=100, tight_layout=True) => tight_layout and dpi does not work with fixed size axes

    pos = (0, 0, 1, 1)  # Position of the grid in the figure
    horiz = [Size.Fixed(ax_bbox.width), Size.Fixed(1), Size.Fixed(ax_bbox.width)]
    vert = [Size.Fixed(ax_bbox.height)]
    divider = Divider(fig_dbg, pos, horiz, vert, aspect=False)

    # Add the two Axes for debug drawing
    # TODO maybe have to handle more specific characteristics from the original Axe beyond axis scales and limits
    ax_data = fig_dbg.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=0, ny=0),
        xscale=ax.get_xscale(),
        yscale=ax.get_yscale(),
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
    )
    ax_geoms = fig_dbg.add_axes(
        divider.get_position(),
        axes_locator=divider.new_locator(nx=2, ny=0),
        xscale=ax.get_xscale(),
        yscale=ax.get_yscale(),
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
    )

    for _, sub_ax in enumerate([ax_data, ax_geoms]):
        for line in retrieve_lines_and_labels(ax)[0]:
            sub_ax.plot(*(line.get_data()), linewidth=line.get_linewidth(), label=line.get_label())
        # Rotate x-axis labels manually since autofmt_xdate does not work for multiple axes which are not separated into subplots,
        # Solution found at https://stackoverflow.com/questions/48078540/matplotlib-autofmt-xdate-fails-to-rotate-x-axis-labels-after-axes-cla-is-c
        for label in sub_ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation(30)

    ax_data.axis(ax.axis())
    ax_geoms.sharex(ax_data)
    # TODO Commented because leads to wrong ax_geoms aspect -> figure out why I used that initially.
    # ax_data.set_aspect(get_axe_aspect(ax), adjustable="box")
    # ax_geoms.set_aspect(get_axe_aspect(ax), adjustable="datalim")
    fig_dbg.canvas.draw()
    return ax_data, ax_geoms


#########################################################
# TODO create a transformation class transGeom linked   #
# TODO to the Axe from which geometries where extracted #
#########################################################


def get_disp2geom_trans(ax: Axes) -> Transform:
    """Returns a transformation from display coordinates to Axe coordinates scaled
    to be cartesian on display"""
    width, height = ax.get_window_extent().size
    return ax.transAxes.inverted() + Affine2D().scale(1, height / width)


def get_geom2disp_trans(ax: Axes) -> Transform:
    """Returns a transformation from Axe coordinates scaled to be cartesian on display,
    to display coordinates"""
    width, height = ax.get_window_extent().size
    return Affine2D().scale(1, width / height) + ax.transAxes


#######################################################


def get_axe_lines_widths(ax: Axes, linelikeHandles, linelikeLabels) -> dict[str, float]:
    """Build a dictionary of line's width per label
    (note: linewidth for a Line2D object is given in points -> display coordinadates)
    Note : since I do not know how a linewidth is drawn internally by matplotlib
    I take for pixel size in physical coordinates √2
    """
    ld_lw = {}
    # Line width size identified thanks to :
    # - ImportanceOfBeingErnest answer at https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
    # -> showing differences betwen points and pixels
    # - sdau answer at https://stackoverflow.com/questions/16649970/custom-markers-with-screen-coordinate-size-in-older-matplotlib
    # -> showing the function points_to_pixels to convert points into pixels
    for label in linelikeLabels:
        h = linelikeHandles[linelikeLabels.index(label)]
        lw_pts = h.get_linewidth()
        lw_display = ax.get_figure().canvas.get_renderer().points_to_pixels(lw_pts)
        lw_geoms = (1 / ax.get_window_extent().width) * lw_display
        ld_lw.setdefault(label, lw_geoms)
    return ld_lw


def get_axe_lines_geometries(
    ax: Axes, linelikeHandles, linelikeLabels, ld_lw: dict[str, float], nowarn: bool = True
) -> Plot_Labels_Geom_Data:
    """Build a dictionary of (line chunks and bufferd line chunk) list
    (in Axes coordinates) for all labels

    Returns :
    `{<label>: {'lcd': {<line chunk index>: {'lc': <line chunk, 'lcb': <buffered line chunk>}}}}`
    """
    ld = {label: {} for label in linelikeLabels}
    trans_data2geom = ax.transData + get_disp2geom_trans(ax)
    for label in linelikeLabels:
        h = linelikeHandles[linelikeLabels.index(label)]
        # Initialize current label's line chunk dictionary
        ld[label].setdefault("lcd", {})
        # Get the x and y data from Line2D object
        l_xdata_raw, l_ydata_raw = h.get_data(orig=False)
        # Check that x and y data are either of type float or datetime64.
        # And if of type datetime64, converts it to float
        if l_xdata_raw.dtype == np.datetime64:
            # Convert x data from date_time to float
            l_xdata_f = date2num(l_xdata_raw)
        elif l_xdata_raw.dtype == np.dtype("float64"):
            l_xdata_f = l_xdata_raw
        else:
            raise ValueError(f"Line label: {label} has x data neither of type float or date, which is not handle for now")
        if l_ydata_raw.dtype == np.datetime64:
            # Convert y data from date_time to float
            l_ydata_f = date2num(l_ydata_raw)
        elif l_ydata_raw.dtype == np.dtype("float64"):
            l_ydata_f = l_ydata_raw
        else:
            raise ValueError(f"Line label: {label} has y data neither of type float or date, which is not handle for now")
        # Convert from Data coordinates to Axes coordinates
        l_xydata_geom_coords = ma.masked_invalid(trans_data2geom.transform(np.c_[l_xdata_f, l_ydata_f]))
        # Axe ax_geoms box in geometry coordinates, in a shape compatible with shapely.clip_by_rect function
        axe_box = shp.box(*np.concatenate(trans_data2geom.transform(np.c_[ax.get_xlim(), ax.get_ylim()])))
        shp.prepare(axe_box)
        if (seqlen := len(ma.clump_unmasked(l_xydata_geom_coords[:, 1]))) > 1:
            if not nowarn:
                print(f"Line {label} of Axe {ax.get_title()} is splitted in {seqlen} continuous chunks")
        i = 0
        for s in ma.clump_unmasked(l_xydata_geom_coords[:, 1]):
            if s.stop - s.start == 1:
                lc = shp.intersection(shp.points(l_xydata_geom_coords[s.start].data), axe_box)
            else:
                lc = shp.intersection(shp.linestrings(ma.compress_rows(l_xydata_geom_coords[s.start : s.stop])), axe_box)
            if not shp.is_empty(lc):
                if shp.get_num_geometries(lc) > 1:
                    for lc_piece in lc.geoms:
                        shp.prepare(lc_piece)
                        lc_pieceb = shp.buffer(lc_piece, ld_lw[label] / 2)
                        shp.prepare(lc_pieceb)
                        ld[label]["lcd"].setdefault(i, {"lc": lc_piece, "lcb": lc_pieceb})
                        ld[label]["lcd"][i].setdefault("lcp", lc_piece)
                        ld[label]["lcd"][i].setdefault("lcbp", lc_pieceb)
                        i += 1
                else:
                    shp.prepare(lc)
                    lcb = shp.buffer(lc, ld_lw[label] / 2)
                    shp.prepare(lcb)
                    ld[label]["lcd"].setdefault(i, {"lc": lc, "lcb": lcb})
                    ld[label]["lcd"][i].setdefault("lcp", lc)
                    ld[label]["lcd"][i].setdefault("lcbp", lcb)
                    i += 1
    return ld


def plot_geometric_line_chunks(ax_geoms: Axes, ld: Plot_Labels_Geom_Data):
    """Plot curves as geometric objects for DEBUG"""
    # Remove all orginal lines from ax_geoms
    for line in ax_geoms.get_lines():
        line.remove()
    # Plot all the line chunks for all labels
    for label in list(ld):
        for lc_idx in list(ld[label]["lcd"]):
            dbg_line_color = ax_geoms.plot(
                *(shp.get_coordinates(ld[label]["lcd"][lc_idx]["lc"]).T),
                label=(label + f" - chunk #{lc_idx}"),
                transform=get_geom2disp_trans(ax_geoms),
                clip_on=False,
                linewidth=0.5,
                linestyle="dashed",
                zorder=5,
            )
            # Add the outline of the original line using the buffered version of the line chunk
            ax_geoms.plot(
                *(shp.get_coordinates(shp.boundary(ld[label]["lcd"][lc_idx]["lcb"])).T),
                label=(
                    "_" + label + f" - buffered chunk #{lc_idx}"
                ),  # Adding a leading underscore prevent the artist to be included in automatic legend
                color=dbg_line_color[0].get_color(),
                transform=get_geom2disp_trans(ax_geoms),
                clip_on=False,
                linewidth=0.5,
                zorder=5,
            )
    ax_geoms.legend()
    ax_geoms.get_figure().canvas.draw()


def update_ld_with_label_text_box_dimensions(
    ax, linelikeHandles, linelikeLabels, ld, label, **label_text_kwarg
) -> Tuple[float, float]:
    """Updates line data structure with text label box dimensions

    {<label>: {'lcd': {
                    <line chunk index>: {
                        'lc': <line chunk,
                        'lcb': <buffered line chunk>}},
               'boxd': {
                   'box_w': <box width>,
                   'box_h': <box height>}}}

    Returns: <box width> and <box height>
    """
    # Put the label on the axe's center
    l_text = ax.text(
        0.5,
        0.5,
        label,
        transform=ax.transAxes,
        color=linelikeHandles[linelikeLabels.index(label)].get_color(),
        clip_on=False,
        backgroundcolor=ax.get_facecolor(),
        horizontalalignment="center",
        verticalalignment="center",
        bbox=dict(boxstyle="square, pad=0.3", mutation_aspect=1 / 10, fc=ax.get_facecolor(), lw=0),
        **label_text_kwarg,
    )
    ax.draw(ax.get_figure().canvas.get_renderer())
    # Retrieve the label's dimensions in display coordinates
    l_bb = l_text.get_bbox_patch().get_window_extent(ax.get_figure().canvas.get_renderer())
    # Get bbox points in Axes' coordinates
    l_box_points = get_disp2geom_trans(ax).transform(l_bb.get_points())
    # Calculate width and height in Axes' coordinates
    l_box_w, l_box_h = (l_box_points[1][0] - l_box_points[0][0]), (l_box_points[1][1] - l_box_points[0][1])
    # Add dictionary of label box dimensions
    ld[label].setdefault("boxd", {"box_w": l_box_w, "box_h": l_box_h})
    # Delete the label from the plot
    l_text.remove()
    return l_box_w, l_box_h


def pt_fp_buffer(pt: shp.Geometry, precision: int = 6) -> float:
    """Returns a reasonable buffer size around a point for intersects binary predicate"""
    return np.linalg.norm(shp.get_coordinates(pt)) * 0.5 * 10 ** (-precision)


def pt_approx_buffer(pt: shp.Geometry, precision: int = 6) -> shp.Geometry:
    """Returns a point approximated by a Polygom equal to the point buffered with a distance
    equal to the point norm * 0.5 * 10**(-precision)"""
    return shp.buffer(pt, pt_fp_buffer(pt, precision))


def update_ld_with_label_position_candidates(
    ppf: float, ld: Plot_Labels_Geom_Data, label: str, l_box_w: float, l_box_h: float
):
    """Find all label position (box center) candidates per label's line chunks
    and update ld[l] with its candidate list. Label data structure is updated
    with 'pcl' key in 'lcd' key for label l

    Args:
    - ppf: position precision factor, which is used as a fraction of the label box height for line chunks sampling
    - ld: plot's labels' geometric data
    - l: label
    - l_box_w: label box width
    - l_box_h: label box height
    """
    # Define a sampling distance unit for the label's box center position on the ligne chunk, equal
    # to a fraction of of the current label's box height
    l_box_cdu = l_box_h * ppf
    for lc_idx in list(ld[label]["lcd"]):
        l_lc = ld[label]["lcd"][lc_idx]["lc"]
        if shp.get_type_id(l_lc) == shp.GeometryType.POINT:
            ld[label]["lcd"][lc_idx].setdefault("pcl", None)
        # Don't try to put the label on line chunks reduced to a single point
        elif shp.get_type_id(l_lc) != shp.GeometryType.LINESTRING:
            raise ValueError("Geometry collection contains a geometric object which is not either a Point or a LineString")
        elif shp.length(l_lc) <= (l_box_w + l_box_h):
            ld[label]["lcd"][lc_idx].setdefault(
                "pcl", None
            )  # Don't try to put label on line chunks of length inferior to label's box width + height,
        else:
            # Create a set of points along the line chunk separated by the previously defined center distance unit for the current label's box geometry
            # Adjust sampling distance to be integer fraction of the line chunk length
            n = int((shp.length(l_lc) - l_box_h - l_box_w) / l_box_cdu) + 1
            l_lc_adjusted_l_box_cdu = (shp.length(l_lc) - l_box_h - l_box_w) / n
            # Create label's line chunk's box center candidates list
            l_lc_bccl = [
                Point2D(
                    *shp.get_coordinates(
                        shp.line_interpolate_point(l_lc, i * l_lc_adjusted_l_box_cdu + (l_box_h + l_box_w) / 2)
                    ).ravel()
                )
                for i in range(0, n + 1)
            ]
            if shp.length(l_lc) % l_box_cdu == 0.0:
                warnings.warn("Last element of the candidates list deleted")
                del l_lc_bccl[-1]  # Remove the last element if it coincides with the right boundary point,
            # Check that two end points of the list are on the line chunk
            assert shp.intersects(pt_approx_buffer(shp.points(l_lc_bccl[0])), l_lc) and shp.intersects(
                pt_approx_buffer(shp.points(l_lc_bccl[-1])), l_lc
            )
            # Check that two end points of the list are at half of box width + height distance ,
            # from their corresponding current label line chunk extremities
            assert (
                shp.intersects(
                    pt_approx_buffer(shp.points(l_lc_bccl[0])),
                    pt_approx_buffer(shp.line_interpolate_point(l_lc, (l_box_h + l_box_w) / 2)),
                )
            ) and (
                shp.intersects(
                    pt_approx_buffer(shp.points(l_lc_bccl[-1])),
                    pt_approx_buffer(shp.line_interpolate_point(l_lc, shp.length(l_lc) - (l_box_h + l_box_w) / 2)),
                )
            )
            ld[label]["lcd"][lc_idx].setdefault("pcl", l_lc_bccl)


# Define number of samples between -(π/2 - ε) to (π/2 - ε), 0 included,
# to be used in label text box proper rotation fit
rotation_sample_number = 180
assert rotation_sample_number % 2 == 0  # Set an even number


@lru_cache(maxsize=rotation_sample_number)
def get_cos_sin(theta: float) -> Tuple[float, float]:
    ct, st = np.cos(theta), np.sin(theta)
    return ct, st


def get_box_rot_and_trans_function(ev: NDArray[Shape["3,4"], Float]):
    def get_lbox_geom(type: str, c: Point2D, theta: float):
        """Returns label's box pygeos objects with center at point c
        and rotated by angle theta :
        - Box polygon
        - Box short sides
        - Box long sides
        """
        xc, yc = c
        ct, st = get_cos_sin(theta)
        RT = np.array(((ct, -st, xc), (st, ct, yc)))
        rtvt = (RT @ ev).T
        if type == "sides":
            return shp.linestrings([(rtvt[i - 1], rtvt[i]) for i in range(4)])
        elif type == "box":
            return shp.polygons(rtvt)
        else:
            raise ValueError("Side parameter value in cached_rotations should be in ['ss', 'ls', 'box']")

    return get_lbox_geom


def get_other_line_chunks_buffered_geom(ld: Plot_Labels_Geom_Data, label: str, lc_idx: int) -> NDArray[Any, Object]:
    """Create the other line chunks set from other labels line chunks and current label other line chunks if any.
    And and make an union geometry for speedup later queries regarding l_box geometry"""
    olcbl = np.array(
        [
            ld[other_label]["lcd"][idx]["lcb"]
            for other_label in list(ld)
            for idx in list(ld[other_label]["lcd"])
            if (idx != lc_idx) or (other_label != label)
        ]
    )
    olcb_geom = olcbl
    shp.prepare(olcb_geom)
    return olcb_geom


# Label's box rotation sample acceptability check regarding its position to the current label's line chunk and feature calculation
def l_box_rotation_check(
    c: Point2D,
    theta: float,
    ld: Plot_Labels_Geom_Data,
    label: str,
    lc_idx: int,
    get_lbox_geom: Callable,
) -> Tuple[bool, float]:
    """Label's box rotation sample acceptability check and alignment error calculation
    if the line chunk as only two intersections points with the label's box small sides
    Returns :
    - theta_isvalid (bool): boolean indicating if theta is valid
    - align_err (float): measure of the alignment error between label's box and line chunk
    """
    # l_lc = ld[label]["lcd"][lc_idx]["lc"] # TODO: old usage -> find out if can be deleted
    # l_lcb = ld[label]["lcd"][lc_idx]["lcb"] # TODO: old usage -> find out if can be deleted
    l_lcp = ld[label]["lcd"][lc_idx]["lcp"]
    l_lcbp = ld[label]["lcd"][lc_idx]["lcbp"]
    l_box_h = ld[label]["boxd"]["box_h"]

    # Rotate label's box sides and center them on current label position candidate
    rtlbs = get_lbox_geom(type="sides", theta=theta, c=c)
    # shp.prepare(rtlbs) #! preparing the box sides geometry slows down the algorithm
    # 1st check that short sides intersects the line chunk and long sides do not intersect the buffered line chunk
    if (not np.any(shp.intersects(l_lcbp, rtlbs[1::2]))) and np.all(
        shp.intersects(l_lcp, rtlbs[::2])
    ):  # TODO DONE with pygeos 0.9, use prepared geometries for l_lc and l_lcb
        # Compute label's box short sides intersection with current label line chunk
        ssis = shp.intersection(l_lcp, rtlbs[::2])  # TODO DONE with pygeos 0.9, use prepared geometries for l_lc and l_lcb
        # 2nd, check if current label line chunk intersects each current label box's short side in a single point
        if np.all(shp.get_type_id(ssis) == shp.GeometryType.POINT):
            theta_isvalid = True
            align_err = np.sum([shp.distance(shp.centroid(rtlbs[2 * i]), ssis[i]) for i in range(2)]) / l_box_h
        else:  # The current label line chunk does not get in and out neatly from the current label's box both short sides
            theta_isvalid = False
            align_err = np.inf
    else:
        # The current label line chunk does not intersect the current label's box both short sides
        # or the current label line chunk is sticking out of the current label's box from its long sides
        theta_isvalid = False
        align_err = np.inf
    return theta_isvalid, align_err


# Intersection check between label's box
# - centered at point c,
# - rotated about point c with angle theta
# - buffered with buffer size or list of buffer sizes
def rtl_box_intersects_olcl(
    c: Point2D,
    theta: float,
    ld: Plot_Labels_Geom_Data,
    label: str,
    lbbfl: list[float],
    olcb_geom: NDArray[Any, Object],
    get_lbox_geom: Callable,
) -> list[bool]:
    """Returns array of bool if the label's box:
    - translated to current label position candidate and rotated by theta,
    - buffered with a buffer size of buffer_size
        intersects with at least one of all other line chunks geometries
    """
    # Create the current label box for x and theta
    rtl_box = get_lbox_geom(type="box", theta=theta, c=c)
    l_box_h = ld[label]["boxd"]["box_h"]
    rtl_buffered_boxes = np.array([shp.buffer(rtl_box, bf * l_box_h) if bf != 0.0 else rtl_box for bf in lbbfl])
    # shp.prepare(rtl_buffered_boxes) #! Boxes preparation slows down the algorithm
    return np.any(
        shp.intersects(olcb_geom.reshape(1, len(olcb_geom)), rtl_buffered_boxes.reshape(len(rtl_buffered_boxes), 1)),
        axis=1,
    )


#! BEGIN DEBUG


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value

    return wrapper_timer


#! END DEBUG


# Find intersections on line chunk at half label box length distance from label center position candidate
# @timer
def hlbld_intersections_on_lc_from_lcc(cl: list[Point2D], hw: float, lc: shp.Geometry) -> dict[Point2D, list[Point2D]]:
    """Find intersections on line chunk at half label box length distance from label center position candidate

    Args:
    - cl: label center candidates
    - hw: label's half width
    - lc: line chunk

    Raises:
    - ValueError: [description]

    Returns: dictionary of intersections points for each label center candidates
    """
    # TODO find alternate method without pygeos buffer and intersections
    center_pts_candidates = shp.points(cl)
    intersection_points = shp.intersection(
        lc, shp.boundary(shp.buffer(center_pts_candidates, hw))
    )  # print(f'{intersection_points=}')
    # print(f'\nintersection_points[0]\n{intersection_points[0]}')

    # Filter the intersections if different from 2
    intersection_points_number = shp.get_num_geometries(intersection_points)  # print(f'{intersection_points_number=}')
    lcc_indices_with_particular_intersections = np.nonzero(np.not_equal(intersection_points_number, 2))[
        0
    ]  # ; print(f'{lcc_indices_with_particular_intersections=}')
    for idx in lcc_indices_with_particular_intersections:
        # TODO BEGIN DEBUG
        assert intersection_points_number[idx] > 2
        # TODO END DEBUG
        if intersection_points_number[idx] > 2:
            intersections = shp.get_parts(intersection_points[idx])  # ; print(f'{intersections=}')
            c_dist = shp.line_locate_point(lc, center_pts_candidates[idx])  # ; print(f'{c_dist=}')
            intersections_dist = shp.line_locate_point(lc, intersections)  # ; print(f'{intersections_dist=}')
            c2intersections_dist = c_dist - intersections_dist  # ; print(f'{c2intersections_dist=}')
            left_point = intersections[
                ma.argmax(ma.masked_greater_equal(c2intersections_dist, 0))
            ]  # ; print(f'{left_point=}')
            right_point = intersections[
                ma.argmin(ma.masked_less_equal(c2intersections_dist, 0))
            ]  # ; print(f'{right_point=}')
            filtered_intersections = shp.multipoints(
                intersections[
                    [
                        ma.argmax(ma.masked_greater_equal(c2intersections_dist, 0)),
                        ma.argmin(ma.masked_less_equal(c2intersections_dist, 0)),
                    ]
                ]
            )  # ; print(f'{filtered_intersections=}')
        else:
            warnings.warn(
                "Label center candidate with less than 2 points on line chunk at a distance of half the label width"
            )
            filtered_intersections = shp.multipoints(None)
        intersection_points[idx] = filtered_intersections
    # Calculation angles
    intersection_points_coordinates = np.reshape(
        shp.get_coordinates(intersection_points), (-1, 2, 2)
    )  # ; print(f'{len(cl)=}\n{np.shape(intersection_points_coordinates)=}')
    # print(f'intersection_points_coordinates[0:2]: \n{intersection_points_coordinates[0:2]}')
    # print(f'{intersection_points_coordinates[:,0,:][0:2]=}')
    # print(f'{intersection_points_coordinates[:,1,:][0:2]=}')
    intersections_vectors = np.subtract(intersection_points_coordinates[:0:], intersection_points_coordinates[:1:])
    # print(f'intersections_vectors[0:2]: \n{intersections_vectors[0:2]}')
    # inter_intersections_distance = LA.norm(intersection_points_coordinates, axis=0)
    # print(f'{inter_intersections_distance[0:2]=}')


# Theta rotation angle candidate evaluation. Candidate appended to tested theta dict with its align error regardless of the evaluation
# TODO : in order to benefit from vectorizing with pygeos, theta candidates should be evaluated
# TODO : with less calls to geometric intersections. And then intersections of line chunk with
# TODO : pairs of sides and with other line chunks should be later vectorized performed.
# TODO : To build the list of theta candidates, one could, for each center position candidate:
# TODO : - Intersect a circle with the current line chunk -> point I1 to In
# TODO : - Filer cases where n < 2
# TODO : - Select Ia and Ib the two first points found on both sides of the center along the line chunk
# TODO :   In case there are more than 2 intersections, selection can be done by the difference between
# TODO :   the distances along the line chunk (shp.line_locate_point) of all the intersections with the distance of the center
# TODO : - Put limits on the angle between segment CIa and CIb: φ being the angle between
# TODO :   the label box diagonal and its great axe, β being the angle between CIa and CIb,
# TODO :   -2φ < π - β < 2φ
def evaluate_theta(
    theta: float,
    c: Point2D,
    ld: Plot_Labels_Geom_Data,
    label: str,
    lc_idx: int,
    lpc_idx: int,
    search_dir: Literal[-1, 1],
    tss: Rotation_BiSearch_State,
    l_lc_mls_lpc: MLS_LC_Candidates_With_LR_Candidates,
    lbbfl: list[float],
    olcb_geom: NDArray[Any, Object],
    hss_angle: float,
    get_lbox_geom: Callable[[str, Point2D, float], Any],
) -> Rotation_BiSearch_State:
    # Initialize the new theta search state
    new_tss = tss
    # Check if label rotation is acceptable
    theta_isvalid, align_err = l_box_rotation_check(
        c=c, theta=theta, ld=ld, label=label, lc_idx=lc_idx, get_lbox_geom=get_lbox_geom
    )
    # If theta aligns well the the current line chunk,
    if theta_isvalid:
        # Update the theta search state
        new_tss[search_dir]["last"] = theta
        if new_tss[search_dir]["first"] is None:
            new_tss[search_dir]["first"] = theta
        # update the line chunk multi level separation dictionary of label position candidates,
        # checking if buffered label boxes intersect with other line chunks
        l_lc_mls_lpc[-1][lpc_idx]["theta_candidates"].setdefault(theta, align_err)
        # Compute intersects bool array for buffering factors list
        intersects_result = rtl_box_intersects_olcl(
            c=c, theta=theta, ld=ld, label=label, lbbfl=lbbfl, olcb_geom=olcb_geom, get_lbox_geom=get_lbox_geom
        )
        # Add theta candidate for each buffering factor if the corresponding buffered box
        # does not intersect with other labels line width buffered line chunks
        for bf_idx in range(len(lbbfl)):
            if not intersects_result[bf_idx]:
                l_lc_mls_lpc[lbbfl[bf_idx]][lpc_idx]["theta_candidates"].setdefault(theta, align_err)
    # Stop the search in the current direction if current theta difference with the last valid theta found
    # is greater than the angle under which the label's half short side is seen from the label's center
    if (new_tss[search_dir]["last"] is not None) and (abs(theta - new_tss[search_dir]["last"]) > hss_angle):
        new_tss[search_dir]["search"] = False
    return new_tss


# For each label position candidates dictionary separation level, keep only the best theta candidates
def filter_best_rotation_angles_per_buffer_size(
    label: str, l_lc_mls_lpc: MLS_LC_Candidates_With_LR_Candidates, best_align_option: bool = True
) -> MLS_LCR_Candidates:
    """For each label position candidates dictionary separation level keep only the best theta candidates

    Args:
    - label: used for warning in case of two identical best theta candidate are found for a particular center position candidate (this warning could be suppressed in the future)
    - l_lc_mls_lpc: MLS_LC_Candidates_With_LR_Candidates : Multi level seperation label center candidates with multi rotation candidates
    - best_align_option: Choose best theta candidate in 0 buffer theta candidates (defaults to True)

    Returns: a simplified list of label center position candidates and best rotation candidate for each one
    """
    l_lc_mls_lcrc = {}
    if best_align_option:
        largest_bf = max(list(l_lc_mls_lpc))
    for bf in sorted(list(l_lc_mls_lpc)):
        l_lc_mls_lcrc.setdefault(bf, {})
        for c_idx in l_lc_mls_lpc[bf]:
            # if c_idx in l_lc_mls_lpc[bf]: # Check if c_idx key has been added for buffer size bf
            if l_lc_mls_lpc[bf][c_idx][
                "theta_candidates"
            ]:  # Check if any valid theta candidate has been found for c_idx and buffer size bf
                best_theta = min(
                    l_lc_mls_lpc[bf][c_idx]["theta_candidates"], key=l_lc_mls_lpc[bf][c_idx]["theta_candidates"].get
                )
                # Check the number of thera candidates with minimal alignment error and issue a warning if there are several
                if (
                    np.nonzero(
                        np.array(list(l_lc_mls_lpc[bf][c_idx]["theta_candidates"].values()))
                        == l_lc_mls_lpc[bf][c_idx]["theta_candidates"][best_theta]
                    )[0].size
                    > 1
                ):
                    warnings.warn(
                        message=f"More than one theta candidate with minimal error for label {label} at position {l_lc_mls_lpc[bf][c_idx]['c_geom']}"
                    )
                # Replace c_idx key value by the best theta and its alignment error
                lpc = {
                    "c_geom": l_lc_mls_lpc[bf][c_idx]["c_geom"],
                    "theta": best_theta,
                    "align_err": l_lc_mls_lpc[bf][c_idx]["theta_candidates"][best_theta],
                }
                # If best_align_option choose 0 buffer best theta candidate for larger buffer
                if best_align_option and (bf == largest_bf):
                    lpc["theta"] = l_lc_mls_lcrc[0][c_idx]["theta"]
                l_lc_mls_lcrc[bf].setdefault(c_idx, lpc)
    return l_lc_mls_lcrc


def plot_label_centers_position_candidates(
    ax_geoms: Axes, mls_lpc: dict[str, dict[float, list[dict]]], lbbfl: list, label: str
) -> None:
    """Plot label centers position candidates with free space around"""
    bf_colors = {1.0: "tab:green", 0.5: "tab:olive", 0: "tab:orange"}
    for bf in lbbfl:
        for lpc_list in mls_lpc[label][bf]:
            X = [lpc["c_geom"].x for lpc in lpc_list]
            Y = [lpc["c_geom"].y for lpc in lpc_list]
            ax_geoms.plot(
                X,
                Y,
                marker="o",
                color=bf_colors[bf],
                markersize=(10 + 20 * bf),
                ls="",
                transform=get_geom2disp_trans(ax_geoms),
                zorder=0.5 / (1.0 + bf),
            )
            if bf == list(bf_colors)[0]:
                for lpc in lpc_list:
                    mx, my = lpc["c_geom"].x, lpc["c_geom"].y
                    ax_geoms.plot(
                        mx,
                        my,
                        marker=(2, 0, math.degrees(lpc["theta"])),
                        color="k",
                        markersize=(10 + 20 * bf),
                        ls="",
                        transform=get_geom2disp_trans(ax_geoms),
                        zorder=0.5 / (1.0 + bf),
                    )


######################################
#### LABELS POSITIONNING FUNCTION ####
######################################


def add_inline_labels(
    ax: Axes,
    ppf: float = 1,
    with_overall_progress: bool = False,
    overall_progress_desc: str = "Label placement",
    with_perlabel_progress: bool = False,
    nowarn: bool = True,
    debug: bool = False,
    fig_for_debug: Optional[Figure] = None,
    **l_text_kwarg,
) -> Optional[Figure]:
    """For an axe made of Line2D plots, draw the legend label for each line
    on it (centered on one of its points) without any intersection
    with any other line or some of its own missing data

    WARNING: depending on the backend some overlapping errors may appear
    due to vector drawing from a discrete resolution. Visual errors disapear
    when viewed on a real image or by increasing the dpi.

    ### Args:
        - `ax` (`Axes`): Axe composed of Line2D objects
        - `ppf` (`float`): position precision factor, fraction of the label box height, itself depending on the font properties.
          This fraction of the label's box height is used as a sampling distance along the line chunk to define candidates for label's bounding box's center positionning
        - `with_overall_progress` (`bool`): progress bar for whole axe's labels positionning
        - `with_perlabel_progress` (`bool`): progress bar per label positionning
        - `nowarn` (`bool`): no warning for unplaced labels
        - `debug` (`bool`): draws a figure showing the algorithm intermediate results
        - `fig_for_debug` (`Figure` | `None`): matplotlib figure for debugging view
        - `**l_text_kwargs`: text kwargs used for label drawing

    ### Returns:
        The figure of the input Axe with labels drawn inline on the input Axe
    """

    # Draw the figure before anything else
    ax.get_figure().canvas.draw_idle()


    # Nested progress bar does not work in every case => set overall and per label progress option mutualy exclusive
    if with_overall_progress and with_perlabel_progress:
        raise ValueError(
            f"Cannot use {with_overall_progress=} and {with_perlabel_progress=} since tqdm nested progress bars does not always work depending on the environment"
        )

    # Build a debug figure
    if debug:
        ax_data, ax_geoms = get_dbg_axes(ax, fig_for_debug)

    #############################
    # Retrieve lines and labels #
    #############################

    # Verify that x axis is a data axis
    # TODO investigate why the verification below has been added in 2020. Commented meanwhile
    # assert isinstance(ax.xaxis.converter, DateConverter)

    # Retrieve linelike objects and their labels
    linelikeHandles, linelikeLabels = retrieve_lines_and_labels(ax)
    if debug:
        data_linelikeHandles, data_linelikeLabels = retrieve_lines_and_labels(ax_data)

    ########################################################################
    # Transform curves into geometric object for label placement computing #
    ########################################################################

    # Build a dictionary of line's width per label
    ld_lw = get_axe_lines_widths(ax, linelikeHandles, linelikeLabels)
    # Build a dictionary of (line chunks and bufferd line chunk) list (in Axes coordinates) for all labels
    ld = get_axe_lines_geometries(ax, linelikeHandles, linelikeLabels, ld_lw, nowarn=nowarn)
    # Plot curves as geometric objects for DEBUG
    if debug:
        plot_geometric_line_chunks(ax_geoms, ld)

    ####################
    # Labels placement #
    ####################

    # Initialize label's box position candidates (box centers and rotations dictionary) indexed by label
    # with level of separation: 0 = tighly separated, 1 = moderately separated, 2 = largely separated
    # with the structure: {label: {0: [], 1: [], 2: []}}
    # -> multi level separation
    mls_lpc = {}

    # Define the buffering factor for each separation level.
    # The factor is used as a fraction of the label's box height added as a buffer to the label's box
    # for intersections evaluation
    # It is assumed in label position candidate evaluation that
    # - the values are sorted in decreasing order
    # - all factors are >= 0, otherwise the buffer would be erosion instead of dilatation
    lbbfl = sorted([0.0, 0.5, 1], reverse=True)
    assert all(buff_factor >= 0 for buff_factor in lbbfl)

    # Create a rotation values list from -(π/2 - ε) to (π/2 - ε), 0 included
    rotation_samples = np.linspace(-np.pi / 2, np.pi / 2, num=rotation_sample_number + 1)  # odd number to keep 0 in the list
    rotation_samples = rotation_samples[1:-1]
    assert len(rotation_samples) == (rotation_sample_number - 1)  # remove extremities

    # Initialize legend content for labels than cannot be positionned properly on their curve
    legend_labels = []

    ########################################################################
    # Identify labels' box geometry and position candidates per line chunk #
    ########################################################################
    for label in linelikeLabels:
        # Get the label text bounding box and x sampling points
        l_box_w, l_box_h = update_ld_with_label_text_box_dimensions(
            ax, linelikeHandles, linelikeLabels, ld, label, **l_text_kwarg
        )
        # Find all label position (box center) candidates per label's line chunks
        update_ld_with_label_position_candidates(ppf, ld, label, l_box_w, l_box_h)

    ##############################################################
    # Prepare overall progress context maanager if option chosen #
    ##############################################################
    if with_overall_progress:
        overall_candidates_number = sum(
            [
                len(ld[label]["lcd"][lc_idx]["pcl"])
                for label in list(ld)
                for lc_idx in list(ld[label]["lcd"])
                if ld[label]["lcd"][lc_idx]["pcl"] is not None
            ]
        )
        overall_progress_cm = tqdm(
            total=overall_candidates_number, ascii=True, ncols=80, desc=overall_progress_desc, position=0, leave=True
        )
    else:
        overall_progress_cm = nullcontext()

    # TODO####################
    # TODO# Vectorize search #
    # TODO####################
    for label in linelikeLabels:
        for lc_idx in list(ld[label]["lcd"]):
            l_pcl = ld[label]["lcd"][lc_idx]["pcl"]
            l_hw = ld[label]["boxd"]["box_h"]
            l_lc = ld[label]["lcd"][lc_idx]["lc"]
            if l_pcl is not None:
                hlbld_intersections_on_lc_from_lcc(cl=l_pcl, hw=l_hw, lc=l_lc)

    #########################
    # Search for each label #
    #########################

    with overall_progress_cm as overall_pbar:
        for label in linelikeLabels:
            ###############################################################################################
            # Create current label's box helper geometries with translation and rotation helper functions #
            ###############################################################################################

            # Get the label text bounding box and x sampling points
            # Retrieve half width and half height in Axes' coordinates
            hw, hh = ld[label]["boxd"]["box_w"] / 2, ld[label]["boxd"]["box_h"] / 2
            l_box_h = ld[label]["boxd"]["box_h"]

            # Build the label's box vertices array, extended with a one vector for translation computation
            ev = np.array(((-hw, hw, hw, -hw), (hh, hh, -hh, -hh), (1, 1, 1, 1)))
            get_lbox_geom = get_box_rot_and_trans_function(ev)

            # Angle under which half of the current label box short side is seen from the center of it => used as threshold to stop valid thetas search
            hss_angle = np.arctan(hh / hw)

            ##################################
            # Iteration over each line chunk #
            ##################################

            # Iterate other each current label line chunks to find:
            # - for each x from the over sampled x data of the current label and within the current chunk x coordinates boundaries,
            # - for each rotation sample,
            # the distance between the current label box and the other lines + current label line chunks boundaries

            # Multi Level Separation Label Position Candidate -> mls_lpc
            mls_lpc.setdefault(label, {-1: []} | {bf: [] for bf in lbbfl})
            # If python version < 3.9.x
            # mls_lpc.setdefault(l, {-1: []})
            # for buff_factor in lbbfl: mls_lpc[l].setdefault(buff_factor, [])

            # TODO : add an option to parallelize per line chunk computation for line chunck with enough candidates
            for lc_idx in list(ld[label]["lcd"]):
                l_pcl = ld[label]["lcd"][lc_idx]["pcl"]
                if l_pcl is not None:
                    #####################################################################
                    # Graph minus current label's line chunk unionned geometry creation #
                    #####################################################################

                    # Create the other line chunks set from other labels line chunks and current label other line chunks if any.
                    # And and make an union geometry for speedup later queries regarding l_box geometry
                    olcb_geom = get_other_line_chunks_buffered_geom(ld, label, lc_idx)

                    ################################################################################################################
                    # Run through center positions and theta rotations to calculate the three distances from the current label box #
                    ################################################################################################################
                    """
                    Initialize three keyed dictionary of valid label position candidates dictionary indexed by center candidate index with as values:
                    - its center geometry
                    - the theta candidate list
                    - the corresponding alignment error list
                    Using the structure: 
                    {lpc_idx: { # candidate index on line chunk candidates sampling
                                c_geom: Point2D, 
                                theta_candidates: {float: float} # key = theta, value = alignment error}} 
                    1st key -1 with label position candidates list w/o overlapping considered
                    2st key 0 with tightly separated label position candidates list 
                      corresponding to no buffer around the label's box when checking intersection with other line chunks
                    3nd key 0.5 with moderately label position candidates list separated 
                      corresponding to a buffer of 0.5 x the label's box height around the label's box when checking intersection with other line chunks
                    4rd key 1.5 with largely label position candidates list separated 
                      corresponding to a buffer of 1.5 x the label's box height around the label's box when checking intersection with other line chunks
                    """
                    # Label Line Chunk Multi Level Label Position Candidates -> l_lc_mls_lpc
                    l_lc_mls_lpc = {-1: {}} | {bf: {} for bf in lbbfl}
                    # If python version < 3.9.x
                    # l_lc_mls_lpc = {-1: {}}
                    # for buff_factor in lbbfl: l_lc_mls_lpc.setdefault(buff_factor, {})

                    ##################################################
                    # Evaluation of each positions on the line chunk #
                    ##################################################

                    # Initialize the best bet for theta
                    bb_theta_idx = len(rotation_samples) // 2

                    # Context manager depending on with_perlabel_progress parameter
                    if with_perlabel_progress:
                        if with_overall_progress:
                            perlabel_progress_cm = tqdm(
                                total=len(l_pcl),
                                ascii=True,
                                ncols=80,
                                desc=f"{label + ' - lc#' + str(lc_idx): <20}",
                                position=1,
                                leave=False,
                            )
                        else:
                            perlabel_progress_cm = tqdm(
                                total=len(l_pcl),
                                ascii=True,
                                ncols=80,
                                desc=f"{label + ' - lc#' + str(lc_idx): <20}",
                                position=0,
                                leave=True,
                            )
                    else:
                        perlabel_progress_cm = nullcontext()

                    with perlabel_progress_cm as perlabel_pbar:
                        for c_idx, c in enumerate(l_pcl):
                            # Initialize theta search state. first and last refer to first and last valid theta found in radians
                            tss = {
                                1: {"search": True, "first": None, "last": None},
                                -1: {"search": True, "first": None, "last": None},
                            }
                            # Add c_idx to l_lc_mls_lpc for all buffering factors
                            for bf in list(l_lc_mls_lpc):
                                l_lc_mls_lpc[bf].setdefault(c_idx, {"c_geom": c, "theta_candidates": {}})
                            idx_shift = 0
                            # Iterate search on rotation samples bidirectionally as long as the search has not be stopped in both direction
                            while tss[1]["search"] or tss[-1]["search"]:
                                for search_dir in [1, -1]:
                                    if tss[search_dir]["search"]:  # If search has not been stopped in <search_dir> direction
                                        # Evaluate theta index value
                                        theta_idx = bb_theta_idx + search_dir * idx_shift + (search_dir + 1) // 2
                                        if (
                                            0 <= theta_idx < len(rotation_samples)
                                        ):  # If theta index is in rotation_samples list index range
                                            theta = rotation_samples[theta_idx]  # Retrieve theta value from rotation_samples
                                            # Before evaluating current theta, check if search in the current direction might be stopped
                                            # because it has never been successful, whereas the search in the other direction
                                            # has already been successful at least once,
                                            # and difference between the current theta and the first valid theta found in the other direction
                                            # is greater than the angle under which the label's half short side is seen from the label's center
                                            if (
                                                (tss[search_dir]["last"] is None)
                                                and (tss[-search_dir]["first"] is not None)
                                                and (abs(theta - tss[-search_dir]["first"]) > hss_angle)
                                            ):
                                                tss[search_dir]["search"] = False
                                            else:
                                                tss = evaluate_theta(
                                                    theta=theta,
                                                    lpc_idx=c_idx,
                                                    c=c,
                                                    search_dir=search_dir,
                                                    tss=tss,
                                                    ld=ld,
                                                    label=label,
                                                    lc_idx=lc_idx,
                                                    l_lc_mls_lpc=l_lc_mls_lpc,
                                                    lbbfl=lbbfl,
                                                    olcb_geom=olcb_geom,
                                                    hss_angle=hss_angle,
                                                    get_lbox_geom=get_lbox_geom,
                                                )
                                        else:
                                            tss[search_dir]["search"] = False
                                idx_shift += 1

                            # Set the best found theta regardless of the rotated label's box intersections with
                            # all other labels line chunks and current label's other line chunks
                            if c_idx in l_lc_mls_lpc[-1]:
                                if l_lc_mls_lpc[-1][c_idx]["theta_candidates"]:
                                    best_theta = min(
                                        l_lc_mls_lpc[-1][c_idx]["theta_candidates"],
                                        key=l_lc_mls_lpc[-1][c_idx]["theta_candidates"].get,
                                    )
                                    bb_theta_idx = np.nonzero(rotation_samples == best_theta)[0][0]

                            # Update progress for current label line chunk
                            if with_overall_progress:
                                overall_pbar.update()
                            if with_perlabel_progress:
                                perlabel_pbar.update()

                    # For each label position candidates dictionary separation level keep only the best theta candidates
                    l_lc_mls_lcrc = filter_best_rotation_angles_per_buffer_size(label, l_lc_mls_lpc)

                    # Append the current label's separation level dictionary the lists of contiguous label box position candidates
                    # found for the current line chunk
                    for bf in list(l_lc_mls_lcrc):
                        # Split the current line chunk label position candidates dictionary into
                        # lists of candidates with contiguous label position candidate indexes
                        # and add those lists of contiguous candidates the current label position candidates dictionary
                        indexes = np.array(list(l_lc_mls_lcrc[bf]))
                        contiguous_l_lc_mls_lcrc_lists = [
                            [l_lc_mls_lcrc[bf][key] for key in group]
                            for group in np.split(indexes, np.nonzero(np.diff(indexes) > 1)[0] + 1)
                        ]
                        mls_lpc[label][bf].extend(contiguous_l_lc_mls_lcrc_lists)

                    # Plot position candidate with free space around
                    if debug:
                        plot_label_centers_position_candidates(ax_geoms, mls_lpc, lbbfl, label)

            # Pick the best position candidate
            # TODO do better than choose only the center of a continous list of center candidates:
            # take into account neighbours if they have a lower alignment error
            bsl = None
            # Search for the longest continuous label position candidate list starting from the best separation level to the least
            for bf in lbbfl:
                if len(mls_lpc[label][bf]) > 1:
                    bsl = mls_lpc[label][bf][np.argmax([len(continuous_ls_lpc) for continuous_ls_lpc in mls_lpc[label][bf]])]
                    if len(bsl) > 0:
                        break
                elif len(mls_lpc[label][bf]) == 1:
                    bsl = mls_lpc[label][bf][0]
                    if len(bsl) > 0:
                        break
            if bsl is None or len(bsl) == 0:
                legend_labels.append(label)
            else:
                bpc = bsl[len(bsl) // 2]
                trans_geom2data = get_geom2disp_trans(ax) + ax.transData.inverted()
                l_x, l_y = trans_geom2data.transform((bpc["c_geom"].x, bpc["c_geom"].y))
                # Plot labels on ax
                labelText = ax.text(
                    l_x,
                    l_y,
                    label,
                    color=linelikeHandles[linelikeLabels.index(label)].get_color(),
                    backgroundcolor=ax.get_facecolor(),
                    horizontalalignment="center",
                    verticalalignment="center",
                    rotation=math.degrees(bpc["theta"]),
                    bbox=dict(boxstyle="square, pad=0.3", mutation_aspect=1 / 10, fc=ax.get_facecolor(), lw=0),
                    **l_text_kwarg,
                )
                fprop = labelText.get_fontproperties()
                if debug:  # Plot labels' boxes on ax_data and chosen labels' centers on ax_geoms
                    ax_data.text(
                        l_x,
                        l_y,
                        label,
                        fontproperties=fprop,
                        color=data_linelikeHandles[data_linelikeLabels.index(label)].get_color(),
                        backgroundcolor=ax_data.get_facecolor(),
                        horizontalalignment="center",
                        verticalalignment="center",
                        rotation=math.degrees(bpc["theta"]),
                        bbox=dict(
                            boxstyle="square, pad=0.3",
                            mutation_aspect=1 / 10,
                            fc=ax_data.get_facecolor(),
                            ec=data_linelikeHandles[data_linelikeLabels.index(label)].get_color(),
                            lw=0.1,
                        ),
                        **l_text_kwarg,
                    )
                    ax_geoms.plot(
                        bpc["c_geom"].x,
                        bpc["c_geom"].y,
                        marker="o",
                        color="k",
                        markersize=10,
                        ls="",
                        transform=get_geom2disp_trans(ax_geoms),
                    )
                    # Get the label box in geom coordinates
                    rtl_box = shp.boundary(get_lbox_geom(type="box", theta=bpc["theta"], c=bpc["c_geom"]))
                    # Plot the label box used in algorithm
                    ax_geoms.plot(
                        *(shp.get_coordinates(rtl_box).T),
                        color="k",
                        linewidth=0.5,
                        transform=get_geom2disp_trans(ax_geoms),
                    )

    # ? Add a legend for all handles which are not line like artists according to function retrieve_lines_and_labels

    # Add legend with all labels than could not be positionned properly on their curve
    if legend_labels and not nowarn:
        warnings.warn(f"Unplaced labels : {legend_labels}")
    if legend_labels and (len(legend_labels) != len(linelikeLabels)):
        # legend_fs = fs if (not reduced_legend_fs) or ((fs_index := fs_list.index(fs)) == 0) else fs_list[fs_index - 1]
        if debug:
            ax_data.legend(
                handles=[data_linelikeHandles[data_linelikeLabels.index(label)] for label in legend_labels],
                labels=legend_labels,
                **l_text_kwarg,
            )
        ax.legend(
            handles=[linelikeHandles[linelikeLabels.index(label)] for label in legend_labels],
            labels=legend_labels,
            facecolor=ax.get_facecolor(),
            **l_text_kwarg,
        )

    if debug:
        return ax_data.get_figure()
    else:
        return ax.get_figure()
