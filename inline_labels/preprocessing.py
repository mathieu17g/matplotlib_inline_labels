import numpy as np
import shapely as shp
from .utils import Timer
from .datatypes import (
    Labelled_Lines_Geometric_Data_Dict,
)
from typing import Literal
import math
from shapely import GeometryType as GT
import numpy.typing as npt
from nptyping import NDArray, Shape, Object
from math import isclose


def pt_fp_buffer(pt: shp.Geometry, precision: int = 6) -> float:
    """Returns a reasonable buffer size around a point for intersects binary predicate"""
    return np.linalg.norm(shp.get_coordinates(pt)) * 0.5 * 10 ** (-precision)


def pt_approx_buffer(pt: shp.Geometry, precision: int = 4) -> shp.Geometry:
    """Returns a point approximated by a Polygon equal to the point buffered with a distance
    equal to the point norm * 0.5 * 10**(-precision)"""
    return shp.buffer(pt, pt_fp_buffer(pt, precision))


# TODO2: if reused elsewhere, add in line chunk geometries, the (unary) union of all other
# TODO2: line chunks
@Timer(name="update_ld_with_label_position_candidates")
def update_ld_with_label_position_candidates(
    ppf: float,
    ld: Labelled_Lines_Geometric_Data_Dict,
    label: str,
    maxpos: int,
    curv_filter_mode: Literal["fast", "precise"] = "fast",
):
    """Find all label position (box center) candidates per label's line chunks
    and update `ld[label]` with its candidates list. Label data structure is updated
    with 'pcl' member for each line chunk in 'lcd' member for label `label`

    Steps are:
    1. Adjust the position precision factor (ppf) to the minimum and maximum candidates
    2. Find the line chunks without any intersections and long enough
       1. skip line chunks reduced to a point
       2. skip line chunks with a length less than the label's bounding box half perimeter
       3. split line chunks around their self intersections
       4. split the resulting sub line chunks around their intersections with all other
       geometries
       5. filter the resulting sub line chunks with a length less than the label's bounding
       box half perimeter
    3. Sample the line chunks with a distance equal to the adjusted `ppf` x label's bounding
    box height
    4. Filter points for which the circle, of radius half box height, intersects the line
    chunk in more than two points.
    5. Filter points which are in a too high curvatures areas

    Args:
    - ppf: position precision factor, which is used as a fraction of the label box height
    for line chunks sampling
    - ld: labels' geometric data
    - label: current label
    - maxpos: maximum label's position candidates before preprocessing
    - curv_filter_mode: the curvature estimation mode for filtering label's position
    candidates which lies on a too high curvature part
    """

    #! Local variables for readability
    # Define the box semi perimter
    lbox_hp = ld[label].boxd.w + ld[label].boxd.h

    # Define the label's box half diagonal
    lbox_hd = math.sqrt(ld[label].boxd.w**2 + ld[label].boxd.h**2) / 2

    # Define the label's box half width
    lbox_hw = ld[label].boxd.w / 2

    # Define the label's box width and height
    lbox_w = ld[label].boxd.w

    #! Compute the union of all geometries of all other labels
    ols = list(ld.keys())
    ols.remove(label)
    if ols:
        aolg = shp.unary_union([lcg.lc for ol in ols for lcg in ld[ol].lcgl])
    else:
        aolg = shp.LineString()

    for lc_idx, lcg in enumerate(ld[label].lcgl):
        lc = lcg.lc
        #! Filter line chunks too short to accomodate a proper label placement
        if shp.get_type_id(lc) == GT.POINT:
            # Don't try to put the label on line chunks reduced to a single point and leave
            # line chunk center candidates' list empty (default value)
            continue
        assert shp.get_type_id(lc) == GT.LINESTRING
        if shp.length(lc) <= lbox_hp:
            # Don't try to put label on line chunks of size inferior to label's box width
            # + height and leave line chunk's center candidates' list empty (default value)
            continue
        #! For long enough line chunks LineString
        # Compute the union of all geometries of all other labels + all other line
        # chunks of the label associated with the line chunk (lc_idx)
        olc_geoms = [
            lcg.lcb for olc_idx, lcg in enumerate(ld[label].lcgl) if olc_idx != lc_idx
        ]
        if olc_geoms:
            aog = shp.unary_union([aolg, *olc_geoms])
        else:
            aog = aolg  # ? Should we make a copy here?
        shp.prepare(aog)

        #! Split the line chunk around its intersections with all other line chunks
        #! of the current labels and of other labels, by substracting the line chunk
        #! with the other geometries
        slcl = list(shp.get_parts(lc.difference(aog)))
        # Check that all resulting line chunk geometries are unique within the list
        assert len(slcl) == len(set(slcl))
        # Remove Point geometries and assert that all remaining geometries are
        # LineString
        slcl = [geom for geom in slcl if shp.get_type_id(geom) != GT.POINT]
        assert all([shp.get_type_id(geom) == GT.LINESTRING for geom in slcl])
        # Remove sub line chunks that are to short for the label to fit on it, i.e of
        # length less than label's box width + height
        slcl = [geom for geom in slcl if shp.length(geom) >= lbox_hp]

        #! Define a sampling distance for the label's box center position candidates on the
        #! ligne chunk, equal to a fraction of of the current label's box height. The
        #! fraction being the position precision factor (ppf) adjusted by the maximum number
        #! of candidates
        total_useful_length = sum(
            [
                (
                    shp.length(slc)
                    if isclose(slc.coords[0][0], slc.coords[-1][0])
                    and isclose(slc.coords[0][1], slc.coords[-1][1])
                    else shp.length(slc) - lbox_hp
                )
                for slc in slcl
            ]
        )
        minppf = total_useful_length / maxpos / ld[label].boxd.h
        appf = max(ppf, minppf)
        lbox_sd = ld[label].boxd.h * appf

        for slc in slcl:
            #! Sample each sub line chunk with a distance of the label's bounding box's
            #! heigh multiplied by the position precision factor: ppf
            assert shp.get_type_id(slc) == GT.LINESTRING

            # Create a set of points along the sub line chunk separated by the previously
            # defined center distance unit for the current label's box geometry
            if isclose(slc.coords[0][0], slc.coords[-1][0]) and isclose(
                slc.coords[0][1], slc.coords[-1][1]
            ):
                # Closed line chunk case : adjust sampling distance to be an integer
                # fraction of the sub line chunk's length
                n = int(shp.length(slc) / lbox_sd)
                fit_sd = shp.length(slc) / n
                # Distance samples along sub line chunk list
                d_smpls = [i * fit_sd for i in range(n)]
            else:
                # ? ensure that the number of distance samples is odd
                # Adjust sampling distance to be an integer fraction of the sub line chunk's
                # length - minus label's box's half perimeter
                n = max(int((shp.length(slc) - lbox_hp) / lbox_sd), 2)
                fit_sd = (shp.length(slc) - lbox_hp) / n
                # Distance samples along sub line chunk list
                d_smpls = [i * fit_sd + lbox_hp / 2 for i in range(n + 1)]
            slc_Pcl = shp.line_interpolate_point(slc, d_smpls)

            # Remove the last element if it coincides with the right boundary point,
            if slc_Pcl.size:
                assert shp.length(slc) % lbox_sd != 0.0
                # //warn("Last element of the candidates list deleted")
                # //del slc_Pcl[-1]

            # Make some verifications on the two end candidates
            if slc_Pcl.size:
                # Check that two end points of the list are on the line chunk
                assert shp.intersects(pt_approx_buffer(slc_Pcl[0]), slc)
                assert shp.intersects(pt_approx_buffer(slc_Pcl[-1]), slc)

                # Check that two end points of the list are at half of box width + height
                # distance, from their corresponding current label line chunk extremities
                if not (
                    isclose(slc.coords[0][0], slc.coords[-1][0])
                    and isclose(slc.coords[0][1], slc.coords[-1][1])
                ):
                    assert shp.intersects(
                        pt_approx_buffer(slc_Pcl[0]),
                        pt_approx_buffer(shp.line_interpolate_point(slc, lbox_hp / 2)),
                    )
                    assert shp.intersects(
                        pt_approx_buffer(slc_Pcl[-1]),
                        pt_approx_buffer(
                            shp.line_interpolate_point(slc, shp.length(slc) - lbox_hp / 2)
                        ),
                    )
                else:
                    # Check that two end points are separated by a linear distance equal to
                    # the sampling distance
                    assert isclose(
                        (
                            shp.length(slc)
                            - shp.line_locate_point(slc, slc_Pcl[-1])
                            + shp.line_locate_point(slc, slc_Pcl[0])
                        ),
                        fit_sd,
                        rel_tol=1e-2,
                    )

            if slc_Pcl.size:
                #! Filter points for which the circle, of radius half box height,
                #! intersects the line chunk in more than two points
                # TODO: use dilatation and curve topology to further enhance this filter
                bl_slc_Pcl = shp.boundary(shp.buffer(slc_Pcl, ld[label].boxd.h / 2))
                mask = np.equal(
                    np.vectorize(shp.get_num_geometries)(shp.intersection(bl_slc_Pcl, lc)),
                    2,
                )
                slc_Pcl = slc_Pcl[mask]

            if slc_Pcl.size:
                #! Filter points which are closer to aog than label's box half height
                bl_slc_Pcl = shp.buffer(slc_Pcl, ld[label].boxd.h / 2)
                mask = np.logical_not(shp.intersects(bl_slc_Pcl, aog))
                slc_Pcl = slc_Pcl[mask]

            if slc_Pcl.size:
                #! Filter points for which the circle of radius equal to the label's box
                #! half diagonal, intersects the current line chunk in two points which are
                #! closer that the label's box width.
                if curv_filter_mode == "fast":
                    ipts_diag = circles_on_line_closest_biintersections(
                        lbox_hd,
                        slc,
                        slc_Pcl,
                        lbox_hw,
                        mode=curv_filter_mode,
                    )
                    assert np.size(ipts_diag) != 0
                    ipts_width = circles_on_line_closest_biintersections(
                        lbox_hw, slc, slc_Pcl, mode=curv_filter_mode
                    )
                    assert np.size(ipts_width) != 0 and (
                        np.size(ipts_width) == np.size(ipts_diag)
                    )
                    # Drop points with less than 2 "half diag circle" non empty
                    # intersections unless they have at least 2 "half width circle" non
                    # empty intersections
                    mask = np.logical_or(
                        np.vectorize(lambda ipt: not np.any(shp.is_empty(ipt)))(ipts_diag),
                        np.vectorize(lambda ipt: not np.any(shp.is_empty(ipt)))(ipts_width),
                    )
                    slc_Pcl = slc_Pcl[mask]
                    ipts_diag = ipts_diag[mask]
                    ipts_width = ipts_width[mask]

                    # For each couple of intersection points check that the distance between
                    # the two points is greater than the label's box width, replacing the
                    # possible "half diag circle" empty side intersection with the
                    # corresponding non empty one from the "half width circle" intersections
                    # * Uses the fact that with shapely a distance between a Point and an
                    # * empty Point is np.nan and that `np.nan >= var:float` is always false
                    assert not (shp.distance(shp.Point(0, 0), shp.Point()) > 1)
                    mask = [
                        (
                            np.all(np.logical_not(shp.is_empty(ipt_d)))
                            and shp.distance(*ipt_d) >= lbox_w
                        )
                        or (
                            np.any(shp.is_empty(ipt_d))
                            and shp.distance(
                                nonempty_ipt(ipt_d),
                                ipt_w[np.argmax(shp.distance(nonempty_ipt(ipt_d), ipt_w))],
                            )
                            >= lbox_w
                        )
                        for ipt_d, ipt_w in zip(ipts_diag, ipts_width)
                    ]
                    slc_Pcl = slc_Pcl[mask]

                    # Compute an estimation of the label's box rotation based on the angle
                    # of the segment joining the two "half width circle" intersections used
                    # to filter on an estimated curvature
                    if slc_Pcl.size:
                        rot_estimates = segments_angles(list(ipts_width[mask]))

                elif curv_filter_mode == "precise":
                    ipts_diag = circles_on_line_closest_biintersections(
                        lbox_hd,
                        slc,
                        slc_Pcl,
                        lbox_hw,
                        mode=curv_filter_mode,
                    )
                    assert np.size(ipts_diag) != 0
                    mask = [shp.distance(*ipt) >= lbox_w for ipt in ipts_diag]
                    slc_Pcl = slc_Pcl[mask]
                    # Compute an estimation of the label's box rotation based on the angle
                    # of the segment joining the two "half width circle" intersections used
                    # to filter on an estimated curvature
                    if slc_Pcl.size:
                        rot_estimates = segments_angles(list(ipts_diag[mask]))

            else:
                continue

            if slc_Pcl.size:
                prev_pcl_len = len(ld[label].lcgl[lc_idx].pcl)
                ld[label].lcgl[lc_idx].slc_sds |= {
                    slice(prev_pcl_len, prev_pcl_len + len(slc_Pcl)): fit_sd  # type: ignore
                }
                ld[label].lcgl[lc_idx].re += rot_estimates  # pyright: ignore
                ld[label].lcgl[lc_idx].pcl.extend(slc_Pcl)

            else:
                continue


def segments_angles(pts: list[npt.NDArray[shp.Point]]) -> list[float]:
    """From an array of couples of points, returns the slopes of segments"""
    pts_coords = np.reshape(shp.get_coordinates(np.vstack(pts)), (-1, 2, 2))
    # Vectors between the two intersections points
    pts_vecs = np.subtract(pts_coords[:, 1, :], pts_coords[:, 0, :])
    # Angles between x-axis and vectors between pairs of intersections points, in [0, pi]
    pts_angs_modpi = np.mod(np.arctan2(pts_vecs[:, 1], pts_vecs[:, 0]), np.pi)
    # Convert angles in [0, pi] to [-pi/2, pi/2]
    pts_angs = pts_angs_modpi - np.pi * (pts_angs_modpi > (np.pi / 2))

    assert np.all(pts_angs >= (-np.pi / 2)) and np.all(
        pts_angs <= (np.pi / 2)
    ), "Angles beyond right half plan"

    return list(pts_angs)


def nonempty_ipt(pts: npt.NDArray[shp.Point]) -> shp.Point:
    """From an array of points, returns the first non empty one"""
    idx: int = np.nonzero(np.logical_not(shp.is_empty(pts)))[0][0]
    return pts[idx]


# @Timer(name="circles_on_line_closest_biintersections")
def circles_on_line_closest_biintersections(
    R: float,
    LS: shp.LineString,
    Ps: npt.NDArray[shp.Point],  # noqa: F722
    min_R: float | None = None,
    mode: Literal["fast", "precise"] = "fast",
) -> NDArray[Shape["*"], Object]:  # noqa: F722
    """Given a linestring LS and an array of points P on this line, returns an array of, for
    each point P, an array of the closest intersection points on either sides of point P, 
    between the linestring and the circle of radius R centered on point P.

    If for a point P, the circle of radius R does not intersect the linestring on one side
    of P, returns the closest point of the intersection between the correponding
    sublinestring and the circle centered on P with the greatest radius R' intersecting the
    sublinestring. If R' < min_R returns no intersection point for this side.

    Note: It is assumed that points Ps are close to the linestring LS, typically constructed
    by an interpolation
    """
    assert R > 0
    assert min_R is None or min_R > 0
    assert isinstance(LS, shp.LineString)
    assert np.size(Ps) > 0
    # Check that Ps contains only points
    assert np.all(np.vectorize(lambda P: isinstance(P, shp.Point))(Ps))

    if mode == "precise":
        # If min_R in not spectified, set it equal to R
        min_R = R if min_R is None else min_R
        # Check that R >= min_R
        assert R >= min_R

        # Split the linestring with the points
        splittedLSs = np.vstack(cut(LS, Ps))
        assert all([len(splittedLS) == 2 for splittedLS in splittedLSs])
        leftLSs = splittedLSs[:, 0]
        assert np.all(np.vectorize(lambda ls: isinstance(ls, shp.LineString))(leftLSs))
        rightLSs = splittedLSs[:, 1]
        assert np.all(np.vectorize(lambda ls: isinstance(ls, shp.LineString))(rightLSs))

        leftIs = np.array(
            [
                shp.intersection(leftLSs[i], shp.boundary(shp.buffer(Ps[i], R)))
                for i in range(np.size(Ps))
            ]
        )
        rightIs = np.array(
            [
                shp.intersection(rightLSs[i], shp.boundary(shp.buffer(Ps[i], R)))
                for i in range(np.size(Ps))
            ]
        )

        # Replace empty left or right intersections with the intersection with the largest circle
        # centered on P and of radius >= min_R intersecting the left or right half linestring
        leftIs = try_a_lower_radius(R, Ps, min_R, leftLSs, leftIs)
        rightIs = try_a_lower_radius(R, Ps, min_R, rightLSs, rightIs)

        # For each intersection reduce it to the closest point along line LS to point P
        leftIs = choose_closest_intersection(leftIs, LS, Ps)
        rightIs = choose_closest_intersection(rightIs, LS, Ps)
        iPs = np.array(
            [np.array([lI, rI]) for lI, rI in zip(leftIs, rightIs)], dtype=object
        )

    elif mode == "fast":
        iPs = shp.intersection(shp.boundary(shp.buffer(Ps, R)), LS)
        # Drop emtpy geometries
        if np.size(iPs) != 0:
            iPs = iPs[np.logical_not(shp.is_empty(iPs))]
        if np.size(iPs) != 0:
            # Check that intersections are MultiPoints since other improbable cases are not
            # handled for now
            assert all(
                [
                    isinstance(ipt, shp.MultiPoint) or isinstance(ipt, shp.Point)
                    for ipt in iPs
                ]
            )
            iPs = np.vectorize(shp.get_parts)(iPs)
            iPs_nb = np.vectorize(np.size)(iPs)

            # For circles with 1 intersection, add an empty Point as second intersection
            circle_inds_with_1_intersection = np.nonzero(np.less(iPs_nb, 2))[0]
            for idx in circle_inds_with_1_intersection:
                iPs[idx] = np.array(iPs[idx].tolist() + [shp.Point()])

            # For circles with more than 2 intersections, identify the points which are closest
            # to the circle center along the line
            circle_inds_with_more_than_2_intersections = np.nonzero(np.greater(iPs_nb, 2))[
                0
            ]

            for idx in circle_inds_with_more_than_2_intersections:
                P_fromLSorig_linecoord = shp.line_locate_point(LS, Ps[idx])
                ptipts_fromlineorig_linecoord = shp.line_locate_point(LS, iPs[idx])
                ptipts_frompt_linecoord = (
                    ptipts_fromlineorig_linecoord - P_fromLSorig_linecoord
                )

                ptipts_before_pt_mask = np.less(ptipts_frompt_linecoord, 0)
                ptipts_after_pt_mask = np.greater(ptipts_frompt_linecoord, 0)

                closest_iP_before_P_singleton_list = (
                    [
                        iPs[idx][ptipts_before_pt_mask][
                            np.argmax(ptipts_frompt_linecoord[ptipts_before_pt_mask])
                        ]
                    ]
                    if np.any(ptipts_before_pt_mask)
                    else []
                )
                closest_iP_after_P_singleton_list = (
                    [
                        iPs[idx][ptipts_after_pt_mask][
                            np.argmin(ptipts_frompt_linecoord[ptipts_after_pt_mask])
                        ]
                    ]
                    if np.any(ptipts_after_pt_mask)
                    else []
                )
                iPs[idx] = np.array(
                    closest_iP_before_P_singleton_list + closest_iP_after_P_singleton_list
                )

            assert all([np.size(ipt) <= 2 for ipt in iPs])

    return iPs


def choose_closest_intersection(
    sideIs: shp.Geometry,
    LS: shp.LineString,
    Ps: npt.NDArray[shp.Point],
) -> npt.NDArray[shp.Point]:
    for i, sideI in enumerate(sideIs):
        if shp.is_empty(sideI) or isinstance(sideI, shp.Point):
            pass
        else:
            sideIparts = shp.get_parts(sideI)
            assert np.all(np.vectorize(lambda g: isinstance(g, shp.Point))(sideIparts))
            subIlinedists = np.vectorize(lambda subI: LS.project(subI))(sideIparts)
            sideIs[i] = sideIparts[np.argmin(np.abs(subIlinedists - LS.project(Ps[i])))]

    return sideIs


# @Timer(name="cut")
def cut(LS, Ps):
    assert isinstance(LS, shp.LineString), (
        "Trying to cut a geometry which is not a LineString. Please report the case through"
        " an issue"
    )
    buffer_size = max(
        1.1 * max(np.vectorize(lambda p: shp.distance(LS, p))(Ps)),
        shp.length(LS) * 1e-4,
    )
    shp.prepare(LS)
    assert np.all(shp.intersects(LS, shp.buffer(Ps, buffer_size)))
    return np.vectorize(shp.get_parts)(
        np.vectorize(lambda b: shp.difference(LS, b))(shp.buffer(Ps, buffer_size))
    )


# @Timer(name="try_a_lower_radius")
def try_a_lower_radius(
    R: float,
    Ps: npt.NDArray[shp.Point],
    min_R: float,
    sideLSs: npt.NDArray[shp.LineString],
    sideIs: npt.NDArray[shp.Geometry],
) -> npt.NDArray[shp.Geometry]:
    sideIs_emptyinds = np.nonzero(shp.is_empty(sideIs))[0]
    for i in sideIs_emptyinds:
        vertices = shp.points(
            sideLSs[i].coords  # pyright:ignore[reportAttributeAccessIssue]
        )
        segments = (
            np.array([shp.LineString(vertices)])
            if np.size(vertices) < 2
            else np.array(
                [
                    shp.LineString([vertices[i], vertices[i + 1]])
                    for i in range(np.size(vertices) - 1)
                ]
            )
        )
        # Ring of inner radio min_R and outer radius R
        min_R_to_R_ring = shp.difference(shp.buffer(Ps[i], R), shp.buffer(Ps[i], min_R))

        # Segments' intersections whith the ring of inner radio min_R and outer radius R
        intersections = [
            np.array([length for length in np.vectorize(shp.length)(geoms) if length > 0])
            for geoms in np.vectorize(shp.get_parts)(
                shp.intersection(segments, min_R_to_R_ring)
            )
        ]
        if not any(list(map(len, intersections))):
            sideIs[i] = shp.Point()
        else:
            # Segments' intersections normalized lengths
            normalized_lengths = np.asarray(
                intersections,
                dtype=object,
            ) / shp.length(segments)
            # Filter null lengths, correponding to intersections reduced to a point
            nonnull_normalized_lengths = filter(len, normalized_lengths)
            # Densifying ratio for haussdorf distance chosen as the smallest
            densifying_ratio = min(
                map(min, nonnull_normalized_lengths),  # pyright: ignore[reportArgumentType]
            )
            assert 0 < densifying_ratio <= 1
            sideIs[i] = shp.intersection(
                sideLSs[i],
                shp.boundary(
                    shp.buffer(
                        Ps[i],
                        shp.hausdorff_distance(Ps[i], sideLSs[i], densify=densifying_ratio),
                    )
                ),
            )
    return sideIs
