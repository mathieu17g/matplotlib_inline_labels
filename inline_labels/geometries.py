from .datatypes import (
    Labelled_Line_Geometric_Data,
    Labelled_Lines_Geometric_Data_Dict,
    Line_Chunk_Geometries,
    Label_Box_Dimensions,
)
from .utils import Timer
from matplotlib.transforms import Transform, Affine2D
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
#//from matplotlib.dates import date2num
from typing import cast
from nptyping import NDArray
import shapely as shp
import shapely.ops as shpops
import numpy as np
from numpy import ma
from math import isclose


def get_disp2geom_trans(ax: Axes) -> Transform:
    """Returns a transformation from display coordinates to Axe coordinates scaled
    to be cartesian on display"""
    width, height = ax.get_window_extent().size
    return ax.transAxes.inverted() + Affine2D().scale(1, height / width)


# TODO: split line geometries in line chunks in line chunks by difference with other
# TODO: to avoid intersections from start
# @Timer(name="get_axe_lines_geometries")
def get_axe_lines_geometries(
    ax: Axes,
    linelikeHandles: list[Line2D],
    linelikeLabels: list[str],
    ld_lw: dict[str, float],
    debug: bool = True,
) -> Labelled_Lines_Geometric_Data_Dict:
    """Build a dictionary of (line chunks and bufferd line chunk) list
    (in Axes coordinates) for all labels

    Returns :
    `Plot_Labels_Geom_Data`
    """
    ld = Labelled_Lines_Geometric_Data_Dict({})
    trans_data2geom = ax.transData + get_disp2geom_trans(ax)
    # For each label add the geometries to label's entry of the Plot_Labels_Geom_Data dict
    # structure, except the intersection free
    for label in linelikeLabels:
        # Retrieve Line2D object corresponding to label
        h = linelikeHandles[linelikeLabels.index(label)]

        # Initialize current label's line chunk dictionary
        ld[label] = Labelled_Line_Geometric_Data()

        # Get the x and y data from Line2D object
        X_raw, Y_raw = cast(tuple[NDArray, NDArray], h.get_data(orig=False))

        # TODO: commented in version 2.1-dev -> to delete after some observation time
        #// # Check that x and y data are either of type float or datetime64.
        #// # And if of type datetime64, converts it to float
        #// print(f"{X_raw.dtype=}")
        #// if X_raw.dtype == np.datetime64:
        #//     # Convert x data from date_time to float
        #//     X_raw_float = date2num(X_raw)
        #// elif X_raw.dtype == np.dtype("float64"):
        #//     X_raw_float = X_raw
        #// else:
        #//     raise ValueError(
        #//         f"Line label: {label} has x data neither of type float or date, which is"
        #//         " not handle for now"
        #//     )
        #// if Y_raw.dtype == np.datetime64:
        #//     # Convert y data from date_time to float
        #//     Y_raw_float = date2num(Y_raw)
        #// elif Y_raw.dtype == np.dtype("float64"):
        #//     Y_raw_float = Y_raw
        #// else:
        #//     raise ValueError(
        #//         f"Line label: {label} has y data neither of type float or date, which is"
        #//         " not handle for now"
        #//     )
#//        # Convert from Data coordinates to Axes coordinates
        #//XY_geom = ma.masked_invalid(
        #//    trans_data2geom.transform(np.c_[X_raw_float, Y_raw_float])
        #//)
        XY_geom = ma.masked_invalid(
            trans_data2geom.transform(np.c_[X_raw, Y_raw])
        )

        # Correct floating point imprecision for closed curves
        unmasked_inds = np.nonzero(np.logical_not(XY_geom.mask))[0]
        if np.size(unmasked_inds) >= 2:
            first_um_ind = unmasked_inds[0]
            last_um_ind = unmasked_inds[-1]
            if isclose(XY_geom[first_um_ind][0], XY_geom[last_um_ind][0]) and isclose(
                XY_geom[first_um_ind][1], XY_geom[last_um_ind][1]
            ):
                XY_geom[last_um_ind][0] = XY_geom[first_um_ind][0]
                XY_geom[last_um_ind][1] = XY_geom[first_um_ind][1]

        # Axe ax_geoms box in geometry coordinates, in a shape compatible with
        # shapely.clip_by_rect function
        axe_box = shp.box(
            *np.concatenate(trans_data2geom.transform(np.c_[ax.get_xlim(), ax.get_ylim()]))
        )
        shp.prepare(axe_box)

        if (seqlen := len(ma.clump_unmasked(XY_geom[:, 1]))) > 1:
            if debug:
                print(
                    f"Line {label} of Axe {ax.get_title()} is splitted in"
                    f" {seqlen} continuous chunks"
                )

        # Create point or line chunk geometries for each continuous sequence of data points
        lcl = list[shp.Point | shp.LineString | shp.MultiLineString]([])
        for s in ma.clump_unmasked(XY_geom[:, 1]):
            # Check if sequence of data points is reduced to one point, and create a Point
            # geometry. Point is clipped by Axe area
            if s.stop - s.start == 1:
                geom = shp.intersection(shp.Point(XY_geom[s.start].data), axe_box)
            # Otherwise create a LineString geometry clipped by the Axe area
            else:
                geom = shp.intersection(
                    shp.LineString(ma.compress_rows(XY_geom[s.start : s.stop])),
                    axe_box,
                )
            lcl.append(geom)

        # Split everything around self intersections
        lgeoms = shp.unary_union(lcl)

        if not shp.is_empty(lgeoms):
            # Extract point geometries
            pt_geoms = shp.unary_union(
                [g for g in shp.get_parts(lgeoms) if isinstance(g, shp.Point)]
            )
            ls_geoms = shp.unary_union(
                [g for g in shp.get_parts(lgeoms) if not isinstance(g, shp.Point)]
            )

            if not shp.is_empty(ls_geoms):
                # Close all that can be on the rest
                polygons, cuts, dangles, invalid = shp.polygonize_full(
                    shp.get_parts(ls_geoms)
                )
                # Check that there is no invalid linearring. There should not be any
                # since an unary_union has be done in the first place
                assert shp.is_empty(invalid)

                # Recover exteriors linearings from polygon(s) if any. LinearRings will
                # be converted to LineStrings by the final the final unary_union, but
                # with equal extremeties (used in preprocessing to recreate LinearRings)
                lrl = [p.exterior for p in polygons.geoms]
                # Line merge the rest
                lsl1 = [ls for ls in shp.get_parts(shpops.linemerge(cuts))]
                lsl2 = [ls for ls in shp.get_parts(shpops.linemerge(dangles))]
                ls_geoms = shp.unary_union(lrl + lsl1 + lsl2)

            for geoms in [pt_geoms, ls_geoms]:
                # Add geometries to label's entry of the Plot_Labels_Geom_Data dict structure
                if not shp.is_empty(geoms):
                    if shp.get_num_geometries(geoms) > 1:
                        for g in geoms.geoms:
                            shp.prepare(g)
                            gb = shp.buffer(g, ld_lw[label] / 2)
                            shp.prepare(gb)
                            ld[label].lcgl += [Line_Chunk_Geometries(lc=g, lcb=gb)]
                    else:
                        shp.prepare(geoms)
                        gb = shp.buffer(geoms, ld_lw[label] / 2)
                        shp.prepare(gb)
                        ld[label].lcgl += [Line_Chunk_Geometries(lc=geoms, lcb=gb)]

    return ld


# TODO: see if blitting could help to speed up redrawing needed to get label boxes dimensions
@Timer(name="update_ld_with_label_text_box_dimensions")
def update_ld_with_label_text_box_dimensions(
    ax: Axes,
    linelikeHandles: list[Line2D],
    linelikeLabels: list[str],
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    **label_text_kwarg,
) -> tuple[float, float]:
    """Updates line data structure with text label box dimensions

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
        bbox=dict(
            boxstyle="square, pad=0.3", mutation_aspect=1 / 10, fc=ax.get_facecolor(), lw=0
        ),
        **label_text_kwarg,
    )
    ax.draw(
        ax.get_figure().canvas.get_renderer()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    )
    # Retrieve the label's dimensions in display coordinates
    lbox = l_text.get_bbox_patch().get_window_extent(  # pyright: ignore[reportOptionalMemberAccess]
        ax.get_figure().canvas.get_renderer()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    )
    # Get bbox points in Axes' coordinates
    lbox_pts = get_disp2geom_trans(ax).transform(lbox.get_points())
    # Calculate width and height in Axes' coordinates
    lbox_w, lbox_h = (lbox_pts[1][0] - lbox_pts[0][0]), (lbox_pts[1][1] - lbox_pts[0][1])
    # Add dictionary of label box dimensions
    ld[label].boxd = Label_Box_Dimensions(w=lbox_w, h=lbox_h)
    # Delete the label from the plot
    l_text.remove()
    return lbox_w, lbox_h


def get_axe_lines_widths(
    ax: Axes, linelikeHandles: list[Line2D], linelikeLabels: list[str]
) -> dict[str, float]:
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
        lw_display = (
            ax.get_figure()
            .canvas.get_renderer()  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
            .points_to_pixels(lw_pts)
        )
        lw_geoms = (1 / ax.get_window_extent().width) * lw_display
        ld_lw.setdefault(label, lw_geoms)
    return ld_lw
