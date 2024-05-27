from datatypes import (
    CCTOL,
    LabelledLineGeometricData,
    Labelled_Lines_Geometric_Data_Dict,
    IFLineChunkGeoms,
    LabelBBDims,
)
from utils import Timer
from matplotlib.transforms import Transform, Affine2D
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# //from matplotlib.dates import date2num
from typing import cast
from nptyping import NDArray
import shapely as shp
import shapely.ops as shpops
from shapely import GeometryType as GT
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
    """Build a dictionary of (line chunks and buffered line chunk) list
    (in Axes coordinates) for all labels

    This function takes in the Axes object, a list of Line2D handles, a list of labels,
    a dictionary of line widths per label, and a debug flag. It returns a dictionary
    containing the geometries for each label.

    The function performs the following steps:
    1. Get the transformation from display coordinates to Axes coordinates scaled to be 
    cartesian on display.
    2. Iterate over each label and retrieve the Line2D object corresponding to the label.
    3. Get the x and y data from the Line2D object and transform them to Axes coordinates.
    4. Create Point and LineString geometries for each continuous sequence of data points.
    5. Correct floating point imprecision at end points for closed curves.
    6. Create a box representing the Axes area in geometry coordinates.
    7. Split the line geometries into line chunks and clip them by the Axes area.
    8. Split the line chunks around self-intersections.
    9. Compute the union of all geometries of other labels and apply a buffer to be able
       to remerge line strings without merging at intersection points.
    10. Split the label's geometries around intersections with other labels' geometries.
    11. Add the resulting geometries to the dictionary for each label.

    Parameters:
    - ax (Axes): The Axes object.
    - linelikeHandles (list[Line2D]): A list of Line2D handles.
    - linelikeLabels (list[str]): A list of labels.
    - ld_lw (dict[str, float]): A dictionary of line widths per label.
    - debug (bool, optional): A flag to enable debug output. Defaults to True.

    Returns :
    `Labelled_Lines_Geometric_Data_Dict` containing the geometries for each label
    """
    ld = Labelled_Lines_Geometric_Data_Dict({})
    trans_data2geom = ax.transData + get_disp2geom_trans(ax)
    # For each label add the geometries to label's entry of the Plot_Labels_Geom_Data dict
    # structure
    #! First stage: get geometries from Line2D objects
    for label in linelikeLabels:
        # Retrieve Line2D object corresponding to label
        h = linelikeHandles[linelikeLabels.index(label)]

        # Get the x and y data from Line2D object
        X_raw, Y_raw = cast(tuple[NDArray, NDArray], h.get_data(orig=False))

        XY_geom = ma.masked_invalid(trans_data2geom.transform(np.c_[X_raw, Y_raw]))

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

        centroid = shp.centroid(lgeoms)

        # Initialize current label's geometry data structure with centroid and empty line
        # chunk geometrires lists (siflcgl and iflcgl) and unset bounding box dimensions
        ld[label] = LabelledLineGeometricData(centroid=centroid)

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

            # Save geometries by label in an intermediary data structure
            for geom in [pt_geoms, ls_geoms]:
                if not shp.is_empty(geom):
                    if shp.get_num_geometries(geom) > 1:
                        for g in geom.geoms:
                            ld[label].siflcgl += [g]
                    else:
                        ld[label].siflcgl += [geom]

    #! Second stage: split all line chunks intersections with other labels' geometries
    for label in linelikeLabels:

        # Compute the union of all geometries of all other labels and apply a epsilon buffer
        # to be able to remerge line strings without merging at intersections points between
        # label's geometries and other labels's geometries after difference
        ols = list(ld.keys())
        ols.remove(label)
        if ols:
            aolgb = shp.buffer(
                shp.unary_union([geom for ol in ols for geom in ld[ol].siflcgl]), CCTOL / 10
            )
        else:
            aolgb = shp.Polygon()

        # Split label's geometries around interserctions with all other labels' geometries,
        # add resulting geometries to label's entry of the Plot_Labels_Geom_Data dict
        # structure
        for (i, g) in enumerate(ld[label].siflcgl):
            if shp.get_type_id(g) == GT.POINT or shp.get_type_id(g) == GT.LINESTRING:
                # TODO: filter Point geometries, before linemerge
                geoms = shp.get_parts(g.difference(aolgb))
                pt_geoms = [g for g in geoms if shp.get_type_id(g) == GT.POINT]
                ls_geoms = [g for g in geoms if shp.get_type_id(g) != GT.POINT]
                sgeoms = pt_geoms + [sg for sg in shp.get_parts(shpops.linemerge(ls_geoms))]
                for sg in sgeoms:
                    if not shp.is_empty(sg):
                        shp.prepare(sg)
                        gb = shp.buffer(sg, ld_lw[label] / 2)
                        shp.prepare(gb)
                        ld[label].iflcgl += [IFLineChunkGeoms(lc=sg, lcb=gb)]
                        ld[label].if2siflcgl_inds += [i]
            else:  # pragma: no cover
                raise ValueError(
                    f"Should not have encoutered a geometry of type {shp.get_type_id(g)}."
                    " Please report an issue on the project's github page."
                )

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
    ld[label].boxd = LabelBBDims(w=lbox_w, h=lbox_h)
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
