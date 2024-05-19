from functools import cache
from typing import Any, Callable, get_args, cast, Literal
import numpy as np
import numpy.ma as ma
from nptyping import NDArray, Shape, Float, Object
from .datatypes import (
    Label_Box_Dimensions,
    Labelled_Lines_Geometric_Data_Dict,
    Sign,
    Rotation_BiSearch_State,
    Label_PmR,
    Label_R,
    Label_PRcs,
    Label_PRc,
    Line_Chunk_Geometries,
    Label_Rotation_Estimates_Dict,
    Rotation_Search_State,
    Label_Rs,
    Labels_lcs_adjPRcs_groups,
    Labels_PRcs,
)
import shapely as shp
from shapely import Point
from .utils import Timer
from shapely import GeometryType as GT
from tqdm import tqdm
from contextlib import nullcontext
from math import isclose, inf, atan, degrees, ceil


# Define number of samples between -(π/2 - ε) to (π/2 - ε), 0 included,
# to be used in label text box proper rotation fit
ROTATION_SAMPLES_NUMBER = 180
assert ROTATION_SAMPLES_NUMBER % 2 == 0  # Set an even number

# Create a rotation values list from -(π/2 - ε) to (π/2 - ε), 0 included
# Odd number to keep 0 in the list
RS = np.linspace(-np.pi / 2, np.pi / 2, num=ROTATION_SAMPLES_NUMBER + 1)
IRSrad = (RS[-1] - RS[0]) / ROTATION_SAMPLES_NUMBER
IRSdeg = degrees(IRSrad)

# Define the buffering factor for each separation level.
# The factor is used as a fraction of the label's box height added as a buffer to the
# label's box for intersections evaluation
# It is assumed in label position candidate evaluation that
# - the values are sorted in decreasing order
# - all factors are >= 0, otherwise the buffer would be erosion instead of dilatation
SEP_LEVELS = sorted([0.0, 0.5, 1.0], reverse=True)
assert all(buff_factor >= 0 for buff_factor in SEP_LEVELS)

# Seperation level corresponding to case where the label's bounding box overlaps another
# Geometry
NOSEP_LEVEL = -1.0
assert np.all(NOSEP_LEVEL < np.array(SEP_LEVELS))


@cache
def get_cos_sin(rot: float) -> tuple[float, float]:
    ct, st = np.cos(rot), np.sin(rot)
    return ct, st


@cache
def get_buffered_ev(w: float, h: float, buffer: float) -> NDArray[Shape["3,4"], Float]:
    # Retrieve half width and half height in Axes' coordinates
    hw, hh = w / 2, h / 2

    # Build the label's box vertices, extended with a one vector for translation computation
    ev = np.array(((-hw, hw, hw, -hw), (hh, hh, -hh, -hh), (1, 1, 1, 1)))

    assert buffer >= 0, "`buffer` should be >= 0.0"
    if buffer == 0.0:
        return ev
    else:
        b = 2 * buffer * abs(ev[1, 0])
        return np.array(((np.sign(ev[0]) * b) + ev[0], (np.sign(ev[1]) * b) + ev[1], ev[2]))


def get_box_rot_and_trans_function(boxd: Label_Box_Dimensions):
    def get_lbox_geom(
        type: Literal["sides", "box"], c: Point, rot: float, buffer: float = 0.0
    ):
        """Returns label's box pygeos objects with center at point c
        and rotated by angle rot:
        - Box polygon
        - Box short sides
        - Box long sides
        """
        # Create a Rotation and Translation matrix
        ct, st = get_cos_sin(rot)
        RT = np.array(((ct, -st, c.x), (st, ct, c.y)))

        bev = get_buffered_ev(boxd.w, boxd.h, buffer)

        rtvt = (RT @ bev).T
        if type == "sides":
            return shp.linestrings([(rtvt[i - 1], rtvt[i]) for i in range(4)])
        elif type == "box":
            return shp.polygons(rtvt)

    return get_lbox_geom


def get_other_line_chunks_buffered_geom(
    ld: Labelled_Lines_Geometric_Data_Dict, label: str, lc_idx: int
) -> NDArray[Any, Object]:
    """Create the other line chunks set from other labels line chunks and current label
    other line chunks if any. And make an union geometry to speedup later operations with
    label's bounding box geometry"""
    olcbl = np.array(
        [
            ol_lcg.lcb
            for other_label in list(ld)
            for ol_lc_idx, ol_lcg in enumerate(ld[other_label].lcgl)
            if (ol_lc_idx != lc_idx) or (other_label != label)
        ]
    )
    olcb_geom = olcbl
    shp.prepare(olcb_geom)
    return olcb_geom


@Timer(name="l_box_rot_check")
def l_box_rot_check(
    c: Point,
    rot: float,
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    lc_idx: int,
    get_lbox_geom: Callable[[str, Point, float, float], Any],
    buffer: float = 0.0,
) -> tuple[bool, float]:
    """Label's box rotation sample acceptability check and alignment error calculation
    if the line chunk as only two intersections points with the label's box small sides
    Returns :
    - rot_isvalid (bool): boolean indicating if rot is valid
    - align_err (float): measure of the alignment error between label's box and line chunk
    """
    l_lc = ld[label].lcgl[lc_idx].lc
    l_lcb = ld[label].lcgl[lc_idx].lcb
    lbox_h = ld[label].boxd.h

    # Rotate label's box sides and center them on current label position candidate
    rtlbs = get_lbox_geom("sides", c, rot, buffer)
    # shp.prepare(rtlbs) #! preparing the box sides geometry slows down the algorithm
    # 1st check that short sides intersects the line chunk and long sides do not intersect
    # the buffered line chunk
    if (not np.any(shp.intersects(l_lcb, rtlbs[1::2]))) and np.all(
        shp.intersects(l_lc, rtlbs[::2])
    ):
        # Compute label's box short sides intersection with current label line chunk
        ssis = shp.intersection(l_lc, rtlbs[::2])
        # 2nd, check if current label line chunk intersects each current label box's short
        # side in a single point
        if np.all(shp.get_type_id(ssis) == GT.POINT):
            rot_isvalid = True
            align_err = (
                (
                    np.sum(
                        [
                            shp.distance(shp.centroid(rtlbs[2 * i]), ssis[i])
                            for i in range(2)
                        ]
                    )
                    / lbox_h
                )
                if buffer == 0.0
                else np.nan
            )
        # The current label line chunk does not get in and out neatly from the current
        # label's box both short sides
        else:
            rot_isvalid = False
            align_err = np.inf if buffer == 0.0 else np.nan
    else:
        # The current label line chunk does not intersect the current label's box both short sides
        # or the current label line chunk is sticking out of the current label's box from its long sides
        rot_isvalid = False
        align_err = np.inf if buffer == 0.0 else np.nan
    return rot_isvalid, align_err


# TODO: test if it would be faster to test intersection incrementally starting with the
# TODO: largest buffering factor


@Timer(name="get_sep")
def get_sep(
    c: Point,
    rot: float,
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    lc_idx: int,
    olcb_geom: NDArray[Any, Object],
    get_lbox_geom: Callable,
    min_sep: float = NOSEP_LEVEL,
) -> float:
    """Computes the highest buffering factor in SEP_LEVELS (label's box buffering factor
    list) for which the label's box:
    - translated to current label position candidate and rotated by rot,
    - buffered with the buffering factor
    does not intersect with any of all other line chunks geometries
    Returns <min_sep> if there is above
    """
    # Create the current label box for c and rot
    rtl_box = get_lbox_geom("box", c, rot)
    lbox_h = ld[label].boxd.h
    sep = min_sep
    for bf in SEP_LEVELS:
        if bf > min_sep:
            rtl_buffered_box = shp.buffer(rtl_box, bf * lbox_h)
            if (
                np.any(shp.intersects(olcb_geom, rtl_buffered_box))
                or not l_box_rot_check(c, rot, ld, label, lc_idx, get_lbox_geom, bf)[0]
            ):
                continue
            else:
                sep = bf
                break
    return sep


# Theta rotation angle candidate evaluation.
@Timer(name="eval_rot")
def eval_rot(
    rot: float,
    c: Point,
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    lc_idx: int,
    lpc_idx: int,
    search_dir: Sign,
    rbs: Rotation_BiSearch_State,
    lc_LPmRc: list[Label_PmR],
    min_align_err: float,
    olcb_geom: NDArray[Any, Object],
    get_lbox_geom: Callable,
) -> tuple[Rotation_BiSearch_State, float]:
    # Initialize the new rotation search state
    new_rbs = rbs
    # Initialize the new minimum alignment error
    new_min_align_err = min_align_err
    # Check if label rotation is acceptable (no overlap with other geometries)
    lbox_aligns_well, align_err = l_box_rot_check(c, rot, ld, label, lc_idx, get_lbox_geom)
    # If rotated label aligns "well" with the current line chunk,
    if lbox_aligns_well:
        # Check if the label's bounding box does not does not intersect with other geometries
        rtl_box = get_lbox_geom("box", c, rot)
        assert np.all(shp.is_prepared(olcb_geom))

        if not np.any(shp.intersects(olcb_geom, rtl_box)):
            # Update the rotation search state
            new_rbs[search_dir].last = rot
            if new_rbs[search_dir].first is None:
                new_rbs[search_dir].first = rot

            # If align_err <= min_align_err, set new minimum alignment error and add the
            # rotation to the rotation candidates
            if align_err <= min_align_err:
                new_min_align_err = align_err
                # Update the line chunk's label position rotation candidates with the current
                # rotation candidate
                lc_LPmRc[lpc_idx].rot_cands.append(Label_R(rot, align_err, None))
        else:
            lc_LPmRc[lpc_idx].rot_cands.append(Label_R(rot, align_err, NOSEP_LEVEL))

    else:
        # Stop the search in the current direction as soon as an invalid rotation has been
        # found after having already found a valid one
        # ? Cannot find a case where this assumption would lead to missing a valid rotation
        if rbs[search_dir].first is not None:
            new_rbs[search_dir].search = False
        new_min_align_err = min_align_err

    return new_rbs, new_min_align_err

@Timer(name="filter_best_rotations")
def filter_best_rotations(
    lc_LPmRc: list[Label_PmR],
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    lc_idx: int,
    olcb_geom: NDArray,
    get_lbox_geom: Callable,
) -> Label_PRcs:
    """For each label position candidate of a line chunk, keep only the best rotation
    candidate

    It is assumed that for each rotation candidate of a position candidate, that its sep
    value is either `NOSEP_LEVEL` or `None`:
    - `NOSEP_LEVEL` corresponds to a rotated label which aligns well with the line chunk but overlap
    another line chunk or another line
    - `None` corresponds to a rotated label which aligns well with the line chunk and does not
    overlap any other line chunk or other line

    Args:
    - lc_LPmRc: label center candidates, with multi rotation candidates for each
    - ld: dictionary of labelled lines' geometric data indexed by labels' text
    - label: label's text
    - lc_idx: line chunk index in line chunk geometries list of the current labelled line
    geometric data
    - olcb_geom: other line chunks buffered geometries
    - get_lbox_geom: label's bounding box rotation and transmission function


    Returns:
    - a list of label center position candidates and best rotation candidate for each one
    """
    lc_Label_PRcs = Label_PRcs([])
    for PmR in lc_LPmRc:
        # Check if any valid rotation candidate has been found
        if PmR.rot_cands:
            seps = [rc.sep for rc in PmR.rot_cands]
            mask = [sep is None for sep in seps]
            if any(mask):
                errs = ma.array([rc.err for rc in PmR.rot_cands], mask=np.logical_not(mask))
                best_Rc = PmR.rot_cands[ma.argmin(errs)]
                # Get separation for best candidate
                best_Rc.sep = get_sep(
                    c=PmR.pos,
                    rot=best_Rc.rot,
                    ld=ld,
                    label=label,
                    lc_idx=lc_idx,
                    olcb_geom=olcb_geom,
                    get_lbox_geom=get_lbox_geom,
                    min_sep=0.0,
                )
                assert best_Rc.sep >= 0
                # Create the Label center position and single rotation candidate
                lc_Label_PRcs += [Label_PRc(PmR.pos, best_Rc.rot, best_Rc.err, best_Rc.sep)]
            else:
                assert np.all(
                    np.equal(  # pyright: ignore[reportCallIssue]
                        seps, NOSEP_LEVEL  # pyright: ignore[reportArgumentType]
                    )
                )
                errs = [rc.err for rc in PmR.rot_cands]
                best_Rc = PmR.rot_cands[np.argmin(errs)]
                lc_Label_PRcs += [Label_PRc(PmR.pos, best_Rc.rot, best_Rc.err, NOSEP_LEVEL)]
        else:
            lc_Label_PRcs += [Label_PRc(PmR.pos, None, None)]

    return lc_Label_PRcs

@Timer(name="evaluate_candidates")
def evaluate_candidates(
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    lc_idx: int,
    with_perlabel_progress: bool,
    with_overall_progress: bool,
    lcg: Line_Chunk_Geometries,
    lre: Label_Rotation_Estimates_Dict,
    get_lbox_geom: Callable,
    debug: bool,
    graph_labels_lcs_adjPRcs_groups: Labels_lcs_adjPRcs_groups,
    graph_labels_PRcs: Labels_PRcs,
    overall_pbar: tqdm | None,
):
    #####################################################################
    # Graph minus current label's line chunk unionned geometry creation #
    #####################################################################

    # Create the other line chunks set from other labels line chunks and current label other
    # line chunks if any. And make an union geometry to speedup later operations with
    # label's bounding box geometry
    olcb_geom = get_other_line_chunks_buffered_geom(ld, label, lc_idx)

    #####################################################################
    # Run through center positions and  rotations to calculate the      #
    # three distances from the current label box                        #
    #####################################################################

    # Context manager depending on with_perlabel_progress parameter
    if with_perlabel_progress:
        if with_overall_progress:
            perlabel_progress_cm = tqdm(
                total=len(lcg.pcl),
                ascii=True,
                ncols=80,
                desc=f"{label + ' - lc#' + str(lc_idx): <20}",
                position=1,
                leave=False,
            )
        else:
            perlabel_progress_cm = tqdm(
                total=len(lcg.pcl),
                ascii=True,
                ncols=80,
                desc=f"{label + ' - lc#' + str(lc_idx): <20}",
                position=0,
                leave=True,
            )
    else:
        perlabel_progress_cm = nullcontext()

    # Initialize for the line chunk a list containing for each Label's box position
    # candidates its position and its rotations candidates
    lc_LmPmR = list[Label_PmR]()

    # Set the maximum rotation wander from pre estimated label rotation in the search for
    # the best rotation
    idx_shift_max = ceil(2 * atan(ld[label].boxd.h / ld[label].boxd.w) / IRSrad)

    graph_labels_lcs_adjPRcs_groups[label] |= {lc_idx: list[Label_PRcs]([])}

    ##################################################
    # Evaluation of each positions on the line chunk #
    ##################################################

    with perlabel_progress_cm as perlabel_pbar:
        for c_idx, c in enumerate(lcg.pcl):
            # Set initial rotation idx to pre estimated rotation
            init_rot_idx = lre[label][lc_idx][c_idx]

            # Initialize rotation search state. first and last refer to first and last valid
            # rotation found in radians
            rbs = Rotation_BiSearch_State(
                {
                    1: Rotation_Search_State(search=True, first=None, last=None),
                    -1: Rotation_Search_State(search=True, first=None, last=None),
                }
            )
            lc_LmPmR.append(Label_PmR(c, Label_Rs([])))

            # Reset the rotation index shift from init_rot_idx
            idx_shift = 0

            # Reset the minimum alignment error found for rotation candidates evaludation
            min_align_err = inf

            # Iterate search on rotation samples bidirectionally as long as the search has
            # not be stopped in both direction
            sd: Sign
            while (
                (idx_shift <= ((len(RS) - 1) / 2))
                and (rbs[1].search or rbs[-1].search)
                and (idx_shift <= idx_shift_max)
            ):
                for sd in (sd for sd in get_args(Sign) if rbs[sd].search):
                    # If search has not been stopped in <search_dir> direction, evaluate
                    # rotation index value and rotation
                    rot_idx = (init_rot_idx + sd * idx_shift + (sd + 1) // 2) % len(RS)
                    rot = RS[rot_idx]

                    rbs, min_align_err = eval_rot(
                        rot=rot,
                        lpc_idx=c_idx,
                        c=c,
                        search_dir=sd,
                        rbs=rbs,
                        ld=ld,
                        label=label,
                        lc_idx=lc_idx,
                        lc_LPmRc=lc_LmPmR,
                        min_align_err=min_align_err,
                        olcb_geom=olcb_geom,
                        get_lbox_geom=get_lbox_geom,
                    )

                # Check is worth to be continued in both directions
                for sd in (sd for sd in get_args(Sign) if rbs[sd].search):
                    if rbs[sd].first is None and rbs[-sd].first is not None:
                        rbs[sd].search = False

                idx_shift += 1

            # Update progress for current label line chunk
            if overall_pbar is not None:
                overall_pbar.update()
            if perlabel_pbar is not None:
                perlabel_pbar.update()

    # For each label position candidates dictionary separation level keep only the best
    # rotation candidates
    lc_Label_PRcs = filter_best_rotations(
        lc_LmPmR, ld, label, lc_idx, olcb_geom, get_lbox_geom
    )

    # Append label's position/rotation candidates found on this line chunk to the line other
    # candidates
    # Clustered by line chunk and adjacency
    graph_labels_lcs_adjPRcs_groups[label][lc_idx] = cluster_adj_Pcs(
        lc_Label_PRcs, lcg.lc, lcg.slc_sds
    )

    # Unclustered
    graph_labels_PRcs[label] += lc_Label_PRcs


# @Timer(name="cluster_adj_Pcs")
def cluster_adj_Pcs(
    lc_Label_PRcs: Label_PRcs,
    lc: shp.LineString | Point | None,
    slc_sds: dict[slice, float],
) -> list[Label_PRcs]:
    """Knowing that the label's box's center candidates are chosen with a unique sampling
    distance per line chunk, cluster the candidates by adjacency"""
    # Check that the line chunk is not None or reduced to a point
    assert isinstance(lc, shp.LineString)

    # Project the candidates points on the line chunk to make the clusterization based on
    # the projected distance to origin
    pts = [PRc.pos for PRc in lc_Label_PRcs if PRc is not None]

    PRc_clusters = list[Label_PRcs]([])
    for sl, sd in slc_sds.items():
        sl_pts = pts[sl]
        if len(sl_pts) <= 1:
            PRc_clusters += [lc_Label_PRcs[sl]]
        else:
            sl_dists = shp.line_locate_point(lc, pts[sl])
            sl_diffs = np.ediff1d(sl_dists)
            split_inds = list(np.nonzero(np.greater(sl_diffs, 1.1 * sd))[0] + 1)
            PRc_clusters += [
                lc_Label_PRcs[sl][i:j]
                for i, j in zip([0] + split_inds, split_inds + [None])
            ]
    # Handle the closed line chunk case
    if (
        len(PRc_clusters) > 1
        and isclose(lc.coords[0][0], lc.coords[-1][0])
        and isclose(lc.coords[0][1], lc.coords[-1][1])
    ):
        if (
            shp.length(lc)
            - lc.project(PRc_clusters[-1][-1].pos)
            + lc.project(PRc_clusters[0][0].pos)
        ) <= 1.1 * max(list(slc_sds.values())[0], list(slc_sds.values())[-1]):
            PRc_clusters[0] = Label_PRcs(PRc_clusters.pop() + PRc_clusters[0])

    return cast(list[Label_PRcs], PRc_clusters)
