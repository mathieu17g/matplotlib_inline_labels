# from IPython import embed

# embed()

from nptyping import NDArray, Shape, Float, Object
import numpy as np
from math import sqrt, isclose, hypot
from dataclasses import dataclass
import shapely as shp
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient
from utils import Timer
from joblib import Parallel, delayed


def lc_furthest_point_split(
    linestring: LineString,
    center: NDArray[Shape["2"], Float],
) -> NDArray[Shape["2"], Object]:
    """
    Splits a linestring into two parts based on the furthest point from a given center.

    Args:
        linestring: A Shapely LineString representing the linestring.
        center: A numpy array representing the center point with shape (2,).

    Returns:
        A numpy array containing two Shapely LineStrings representing the split parts of the linestring.
    """
    coords = shp.get_coordinates(linestring)
    dists = np.sum((coords - center) ** 2, axis=1)
    furthest_point_idx = np.argmax(dists)
    split1 = LineString(coords[:furthest_point_idx])
    split2 = LineString(coords[furthest_point_idx:])
    return np.array([split1, split2])


def segment_intersection(P1, P2, P3, P4):
    if np.allclose(P1, P2):  # segment1 is degenerate
        if np.all(np.minimum(P3, P4) <= P1) and np.all(P1 <= np.maximum(P3, P4)):
            return P1
        else:
            return None
    if np.allclose(P3, P4):  # segment2 is degenerate
        if np.all(np.minimum(P1, P2) <= P3) and np.all(P3 <= np.maximum(P1, P2)):
            return P3
        else:
            return None

    vecP1P2 = P2 - P1
    vecP3P4 = P4 - P3
    denom = np.cross(vecP1P2, vecP3P4)
    if isclose(denom, 0, abs_tol=1e-9):
        return None  # Lines are parallel.

    vecP3P1 = P1 - P3
    ua = np.cross(vecP3P4, vecP3P1) / denom
    ub = -np.cross(vecP1P2, vecP3P1) / denom

    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # The intersection point is within both segments.
        intersection = P1 + ua * vecP1P2
        return intersection

    return None


# @Timer(name="x2ymod8diff")
def x2ymod8diff(x, y):
    d = (y - x) % 8
    return d - 8 if d > 4 else d


def x2ymod8diffvectorized(x, y):
    d = np.remainder(y - x, 8)
    return np.where(d > 4, d - 8, d)


ROT90CW = np.array([[0, -1], [1, 0]])


# @Timer(name="dist_corner_seg")
def dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx):
    if np.dot(seg_vecs[i], Vn2ePts[Vidx][i]) <= 0:
        return np.sqrt(np.dot(Vn2ePts[Vidx][i], Vn2ePts[Vidx][i]))
    elif np.dot(seg_vecs[i], Vn2sPts[Vidx][i]) >= 0:
        return np.sqrt(np.dot(Vn2sPts[Vidx][i], Vn2sPts[Vidx][i]))
    else:
        return np.abs(np.cross(Vn2sPts[Vidx][i], Vn2ePts[Vidx][i])) / np.sqrt(
            np.dot(seg_vecs[i], seg_vecs[i])
        )


# @Timer(name="dist_corner_line")
def dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx):
    return np.abs(np.cross(Vn2sPts[Vidx][i], Vn2ePts[Vidx][i])) / np.sqrt(
        np.dot(seg_vecs[i], seg_vecs[i])
    )


# @Timer(name="dist_pt_side")
def dist_pt_side(side_vecs_norm, seg_ePts_coords, i, Sidx, s2eregdiffsign):
    if s2eregdiffsign == -1:
        return np.abs(seg_ePts_coords[i, Sidx]) / side_vecs_norm[Sidx]
    else:
        return (
            np.abs(seg_ePts_coords[i, Sidx]) / side_vecs_norm[Sidx] - side_vecs_norm[Sidx]
        )


# @Timer(name="halflinechunk2boxsep_numpy")
def halflinechunk2boxsep_numpy(
    lc: shp.LineString,
    box: shp.Polygon,
):
    box_vertices = shp.get_coordinates(orient(box).exterior)  # Box polygon oriented ccw
    sides = np.stack((box_vertices[:-1], box_vertices[1:]), axis=1)
    side_vecs = sides[:, 1] - sides[:, 0]
    side_vecs_norm = np.sqrt(np.einsum("ij,ij->i", side_vecs, side_vecs))
    lc_vertices = shp.get_coordinates(lc)
    segs = np.stack((lc_vertices[:-1], lc_vertices[1:]), axis=1)
    seg_sPts = segs[:, 0, :]
    seg_ePts = segs[:, 1, :]
    seg_vecs = seg_ePts - seg_sPts
    # Regions are identified by the part of the box (side or vertice) which is the closest
    # to all its points
    Vn2sPts = seg_sPts[np.newaxis, :, :] - box_vertices[:-1, np.newaxis, :]
    Vn2ePts = seg_ePts[np.newaxis, :, :] - box_vertices[:-1, np.newaxis, :]
    seg_sPts_coords = np.einsum("ijk, ik -> ji", Vn2sPts, side_vecs)
    seg_ePts_coords = np.einsum("ijk, ik -> ji", Vn2ePts, side_vecs)
    seg_sPts_reg = np.sign(seg_sPts_coords)
    seg_ePts_reg = np.sign(seg_ePts_coords)

    dists = np.zeros(len(seg_sPts_coords))

    # For the first segment, assume that segment is moving away. It avoids handling the case
    # where the first segment starts from inside the box due to floating point errors.
    if len(dists) == 1:
        return np.array([np.inf])
    else:
        dists[0] = np.inf

    for i in range(1, len(seg_sPts_coords)):
        sreg_idx = REGIONS[tuple(seg_sPts_reg[i])]
        # Raise an error when the 1st point of any segment is inside the box, except for the
        # 1st segment
        if sreg_idx == -1:
            raise ValueError("Unexpected case, please report this issue.")
        sregtype = sreg_idx % 2
        sreg_gidx = sreg_idx // 2
        ereg_idx = REGIONS[tuple(seg_ePts_reg[i])]
        # Decompose into cases depending on start and end points regions
        if not sregtype:  # Start point is in a corner region
            # First check if the end point is beyond the tangent to the start point
            # relatively to the corner's vertice. In that case the segment is moving away
            # Rotate the vector between the corner's vertice and the start point 90 degrees
            # clocwise, and the test if the cross product of the rotated vector and the
            # segment is positive. If it is, the segment is moving away from the box.
            if np.cross(Vn2sPts[sreg_gidx][i] @ ROT90CW, seg_ePts[i] - seg_sPts[i]) >= 0:
                dists[i] = np.inf
            else:
                s2eregdiff = x2ymod8diff(sreg_idx, ereg_idx)
                s2eregdelta = np.abs(s2eregdiff)
                if s2eregdelta == 0:  # End point is in the same corner
                    Vidx = sreg_gidx
                    dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                elif s2eregdelta == 1:  # End point is in an adjacent side region
                    Sidx = (sreg_gidx - 1 - (np.sign(s2eregdiff) - 1) // 2) % 4
                    if np.abs(seg_ePts_coords[i, Sidx]) <= np.abs(seg_sPts_coords[i, Sidx]):
                        dists[i] = dist_pt_side(
                            side_vecs_norm, seg_ePts_coords, i, Sidx, np.sign(s2eregdiff)
                        )
                    else:
                        Vidx = sreg_gidx
                        dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                elif s2eregdelta == 2:  # End point in an adjacent corner
                    Sidx = (sreg_gidx - 1 - (np.sign(s2eregdiff) - 1) // 2) % 4
                    if np.abs(seg_ePts_coords[i, Sidx]) > np.abs(seg_sPts_coords[i, Sidx]):
                        Vidx = sreg_gidx
                        dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                    elif np.abs(seg_ePts_coords[i, Sidx]) == np.abs(
                        seg_sPts_coords[i, Sidx]
                    ):
                        dists[i] = dist_pt_side(
                            side_vecs_norm, seg_ePts_coords, i, Sidx, np.sign(s2eregdiff)
                        )
                    else:  # Compute the distance between the end point corner and the segment
                        Vidx = (sreg_gidx + np.sign(s2eregdiff)) % 4
                        dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                elif s2eregdelta == 3:  # End point in an opposite side region
                    Vidx = (sreg_gidx + np.sign(s2eregdiff)) % 4
                    dists[i] = dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                else:  # End point in the opposite corner
                    # TODO: optimize this part
                    Vidx1 = (sreg_gidx - 1) % 4
                    Vidx2 = (sreg_gidx + 1) % 4
                    dists[i] = min(
                        dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx1),
                        dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx2),
                    )
        else:  # Start point is in a side region
            # First check if the end point is beyond the line parallel to the side passing by
            # the start point. In that case the segment is moving away from the box.
            Sidx = (sreg_gidx - 1) % 4
            if np.abs(seg_ePts_coords[i, Sidx]) > np.abs(seg_sPts_coords[i, Sidx]):
                dists[i] = np.inf
            else:
                s2eregdiff = x2ymod8diff(sreg_idx, ereg_idx)
                s2eregdelta = np.abs(s2eregdiff)
                if s2eregdelta == 0:  # End point is in the same side region
                    Sidx = (sreg_gidx - 1) % 4
                    dists[i] = dist_pt_side(
                        side_vecs_norm, seg_ePts_coords, i, Sidx, np.sign(s2eregdiff)
                    )
                elif s2eregdelta == 1:  # End point is in an adjacent corner
                    Vidx = (sreg_gidx + (1 + np.sign(s2eregdiff)) // 2) % 4
                    dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)

                elif (
                    s2eregdelta == 2 or s2eregdelta == 3
                ):  # End point in an adjacent side region or an opposite corner
                    Vidx = (sreg_gidx + (1 + np.sign(s2eregdiff)) // 2) % 4
                    dists[i] = dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                else:  # pragma: no cover
                    # End point in the opposite side region
                    raise ValueError("Unexpected case, please report this issue.")

    return dists.min()


# Division of the plane around a box into 8 regions plus the box inside region
# The regions are numbered as follows:
#
#          |             |
#     6    |      5      |    4
#        V3|             |V2
# ---------¤<------------¤---------
#          |             ↑
#     7    |     -1      |    3
#          ↓             |
# ---------¤------------>¤---------
#        V0|             |V1
#     0    |      1      |    2
#          |             |
#
# The corner regions include their corresponding half side prolongations and vertice
# The side regions include their corresponding side
# The key tuple corresponds to the sign of the coordinates of the points expressed with an
# origin at its region corresponding vertice (1st vertice ccw for a side region) and the
# fourside vectors taken in ccw order starting at vertice 0 in the lower left.
REGIONS = {
    (-1, -1, 1, 1): 0,
    (-1, 0, 1, 1): 0,
    (-1, 1, 1, -1): 6,
    (-1, 1, 1, 0): 6,
    (-1, 1, 1, 1): 7,
    (0, -1, 1, 1): 0,
    (0, 0, 1, 1): 0,
    (0, 1, 1, -1): 6,
    (0, 1, 1, 0): 6,
    (0, 1, 1, 1): 7,
    (1, -1, -1, 1): 2,
    (1, -1, 0, 1): 2,
    (1, -1, 1, 1): 1,
    (1, 0, -1, 1): 2,
    (1, 0, 0, 1): 2,
    (1, 0, 1, 1): 1,
    (1, 1, -1, -1): 4,
    (1, 1, -1, 0): 4,
    (1, 1, -1, 1): 3,
    (1, 1, 0, -1): 4,
    (1, 1, 0, 0): 4,
    (1, 1, 0, 1): 3,
    (1, 1, 1, -1): 5,
    (1, 1, 1, 0): 5,
    (1, 1, 1, 1): -1,
}


# The half vectorized implementation is 10 times faster than the shapely implementation. The
# shapely and numpy non vectorized implementations are kept for reference and testing purposes.
# ? Could be further vectorized for cases where distances are calculated more often (few
# ? segments moving away from the box).
# @Timer(name="halflinechunk2boxsep_numpy_vectorized")
def halflinechunk2boxsep_numpy_vectorized(
    lc: shp.LineString,
    box: shp.Polygon,
    INFINITE: float = 1.0,  # TODO: verify that 1.0 is a good value for INFINITE
) -> float:
    """
    Compute the separation distance between a line chunk starting on the box side and the
    box using a vectorized implementation (10x faster than the brute shapely implementation).

    Parameters:
    - lc: The line segment represented as a LineString object.
    - box: The box represented as a Polygon object.

    Returns:
    - The separation distance between the line chunk and the box.

    Note:
    - The separation distances for each line segment are calculated by finding the distance
      between the segment and the box. For each distance if it is less than the distance
      between the segment's start point and the box, the distance is set INFINITE.
    - The separation distance of the first segment is set to INFINITE.
    """
    box_vertices = shp.get_coordinates(orient(box).exterior)  # Box polygon oriented ccw
    sides = np.stack((box_vertices[:-1], box_vertices[1:]), axis=1)
    side_vecs = sides[:, 1] - sides[:, 0]
    side_vecs_norm = np.sqrt(np.einsum("ij,ij->i", side_vecs, side_vecs))
    lc_vertices = shp.get_coordinates(lc)
    segs = np.stack((lc_vertices[:-1], lc_vertices[1:]), axis=1)
    seg_sPts = segs[:, 0, :]
    seg_ePts = segs[:, 1, :]
    seg_vecs = seg_ePts - seg_sPts
    # Regions are identified by the part of the box (side or vertice) which is the closest
    # to all its points
    Vn2sPts = seg_sPts[np.newaxis, :, :] - box_vertices[:-1, np.newaxis, :]
    Vn2ePts = seg_ePts[np.newaxis, :, :] - box_vertices[:-1, np.newaxis, :]
    seg_sPts_coords = np.einsum("ijk, ik -> ji", Vn2sPts, side_vecs)
    seg_ePts_coords = np.einsum("ijk, ik -> ji", Vn2ePts, side_vecs)
    # TODO: compute a vector of s2eregdiff with x2ymod8diffvectorized

    # Compute the region indices for the start points
    sreg_inds = np.array([REGIONS[tuple(row)] for row in np.sign(seg_sPts_coords)])
    # Raise an error when the 1st point of any segment is inside the box, except for the 1st
    # segment. For which the 1st point may be inside the box due to floating point errors.
    if np.any(sreg_inds[1:] == -1):
        raise ValueError("Unexpected case, please report this issue.")

    # Compute the region indices for the end points
    ereg_inds = np.array([REGIONS[tuple(row)] for row in np.sign(seg_ePts_coords)])
    # Raise an error when the end point of any segment is inside the box
    if np.any(ereg_inds == -1):
        raise ValueError("Unexpected case, please report this issue.")

    # Determine which start points are in side regions
    sPts_in_sides = np.remainder(sreg_inds, 2).astype(bool)

    # Determine which start points are in corner regions
    sPts_in_corners = np.logical_not(sPts_in_sides)

    # Determine the corner or side region index of the start points
    sreg_ginds = sreg_inds // 2

    # Check if the end point is beyond the tangent to the start point relatively to the
    # corner's vertice. In that case the segment is moving away
    # Rotate the vector between the corner's vertice and the start point 90 degrees
    # clocwise, and the test if the cross product of the rotated vector and the segment is
    # positive. If it is, the segment is moving away from the box.

    # Compute the beyond_tangent values for segments with start points in corner regions
    beyond_tangent = np.full(np.shape(segs)[0], False)
    if np.any(sPts_in_corners):
        beyond_tangent[sPts_in_corners] = (
            np.cross(
                Vn2sPts[[sreg_ginds], [np.arange(Vn2sPts.shape[1])]][0][sPts_in_corners]
                @ ROT90CW,
                seg_vecs[sPts_in_corners],
            )
            >= 0
        )

    # Compute the beyond_tangent values for segments with start points in side regions
    if np.any(sPts_in_sides):
        # TODO: Optimize by limiting Sind computation to the sPts_in_sides mask
        Sinds = np.remainder(sreg_ginds - 1, 4)
        beyond_tangent[sPts_in_sides] = np.abs(
            seg_ePts_coords[np.arange(seg_ePts_coords.shape[0]), Sinds][sPts_in_sides]
        ) > np.abs(
            seg_sPts_coords[np.arange(seg_sPts_coords.shape[0]), Sinds][sPts_in_sides]
        )

    # For the first segment, assume that segment is moving away. It avoids handling the case
    # where the first segment starts from inside the box due to floating point errors.
    beyond_tangent[0] = True

    # Initialize the distances with zeros and set the distances of segments moving away to
    # INFINITE
    dists = np.zeros(len(seg_sPts_coords))
    dists[beyond_tangent] = INFINITE

    # Shortcut for the case of a single segment
    if len(dists) == 1:
        return INFINITE

    for i in np.nonzero(np.logical_not(beyond_tangent))[0]:
        sreg_idx = sreg_inds[i]
        sreg_gidx = sreg_ginds[i]
        ereg_idx = ereg_inds[i]
        # Decompose into cases depending on start and end points regions
        if sPts_in_corners[i]:  # Start point is in a corner region
            s2eregdiff = x2ymod8diff(sreg_idx, ereg_idx)
            s2eregdelta = np.abs(s2eregdiff)
            if s2eregdelta == 0:  # End point is in the same corner
                Vidx = sreg_gidx
                dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
            elif s2eregdelta == 1:  # End point is in an adjacent side region
                Sidx = (sreg_gidx - 1 - (np.sign(s2eregdiff) - 1) // 2) % 4
                if np.abs(seg_ePts_coords[i, Sidx]) <= np.abs(seg_sPts_coords[i, Sidx]):
                    dists[i] = dist_pt_side(
                        side_vecs_norm, seg_ePts_coords, i, Sidx, np.sign(s2eregdiff)
                    )
                else:
                    Vidx = sreg_gidx
                    dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
            elif s2eregdelta == 2:  # End point in an adjacent corner
                Sidx = (sreg_gidx - 1 - (np.sign(s2eregdiff) - 1) // 2) % 4
                if np.abs(seg_ePts_coords[i, Sidx]) > np.abs(seg_sPts_coords[i, Sidx]):
                    Vidx = sreg_gidx
                    dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
                elif np.abs(seg_ePts_coords[i, Sidx]) == np.abs(seg_sPts_coords[i, Sidx]):
                    dists[i] = dist_pt_side(
                        side_vecs_norm, seg_ePts_coords, i, Sidx, np.sign(s2eregdiff)
                    )
                else:  # Compute the distance between the end point corner and the segment
                    Vidx = (sreg_gidx + np.sign(s2eregdiff)) % 4
                    dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
            elif s2eregdelta == 3:  # End point in an opposite side region
                Vidx = (sreg_gidx + np.sign(s2eregdiff)) % 4
                dists[i] = dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
            else:  # End point in the opposite corner
                # TODO: optimize this part
                Vidx1 = (sreg_gidx - 1) % 4
                Vidx2 = (sreg_gidx + 1) % 4
                dists[i] = min(
                    dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx1),
                    dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx2),
                )
        else:  # Start point is in a side region
            s2eregdiff = x2ymod8diff(sreg_idx, ereg_idx)
            s2eregdelta = np.abs(s2eregdiff)
            if s2eregdelta == 0:  # End point is in the same side region
                Sidx = (sreg_gidx - 1) % 4
                dists[i] = dist_pt_side(
                    side_vecs_norm, seg_ePts_coords, i, Sidx, np.sign(s2eregdiff)
                )
            elif s2eregdelta == 1:  # End point is in an adjacent corner
                Vidx = (sreg_gidx + (1 + np.sign(s2eregdiff)) // 2) % 4
                dists[i] = dist_corner_seg(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)

            elif s2eregdelta == 2 or s2eregdelta == 3:
                # End point in an adjacent side region or an opposite corner
                Vidx = (sreg_gidx + (1 + np.sign(s2eregdiff)) // 2) % 4
                dists[i] = dist_corner_line(seg_vecs, Vn2sPts, Vn2ePts, i, Vidx)
            else:  # pragma: no cover
                # End point in the opposite side region
                raise ValueError("Unexpected case, please report this issue.")

    if np.all(beyond_tangent):
        return INFINITE
    else:
        return dists[np.logical_not(beyond_tangent)].min()


# @Timer(name="halflinechunk2boxsep_shapely")
def halflinechunk2boxsep_shapely(
    lc: shp.LineString,
    box: shp.Polygon,
):
    """
    Calculate the separation distances between a line chunk and a box using the shapely
    library.

    Parameters:
    - lc: The line chunk represented as a LineString.
    - box: The box represented as a Polygon.

    Returns:
    - The separation distance between the line chunk and the box.

    Note:
    - The separation distances for each line segment are calculated by finding the distance
      between the segment and the box. For each distance if it is less than the distance
      between the segment's start point and the box, the distance is set infinite.
    - The separation distance of the first segment is set to infinite.
    """
    distsegs2box = box.distance(
        np.array(
            [
                LineString((P, Q))
                for P, Q in zip(shp.get_coordinates(lc)[:-1], shp.get_coordinates(lc)[1:])
            ]
        )
    )
    distsegstartpts2box = box.distance(
        np.array([Point(pt) for pt in shp.get_coordinates(lc)[:-1]])
    )
    # Fix floating point errors for the start point of the first segment which is on the box
    # contour
    distsegstartpts2box[0] = -1.0
    adjusted_dists = np.where(distsegstartpts2box <= distsegs2box, np.inf, distsegs2box)
    return adjusted_dists


# TODO: Try to estimate local kolmogorov complexity. Probably with an "approximate entropy"
# TODO: measure (see wikipedia article). For the sampling use the minimum of the minimum segment
# TODO: length and the a 25th of the box hald width.
# TODO: Then measure the integral the resulting measure weigthed with a decreasing exponential
@Timer(name="halflinechunk_pareto_weighted_total_curvature")
def halflinechunk_pareto_weighted_total_curvature(hlc: LineString, box_h: float) -> float:
    """
    Calculate the pareto-weighted total curvature of a half line chunk.

    Parameters:
    - hlc: The half line chunk represented as a LineString.

    Returns:
    - The pareto-weighted total curvature of the half line chunk.

    Note:
    - The pareto-weighted total curvature is calculated by summing the angles between consecutive
      segments of the half line chunk, weighted by the accumulated segment lengths.
    """
    hlc_vertices = shp.get_coordinates(hlc)
    seg_vecs = np.diff(hlc_vertices, axis=0)
    seg_norms = np.sqrt(np.einsum("ij, ij -> i", seg_vecs, seg_vecs))
    seg_accum_norms = np.add.accumulate(seg_norms)
    total_length = seg_accum_norms[-1]
    if np.shape(hlc_vertices)[0] < 3:
        iseg_angles = np.array([0])
        inds = np.array([0])
        # print(f"Angles: {np.round(np.degrees(iseg_angles))}")
    else:
        seg_uvecs = seg_vecs / seg_norms[:, None]
        iseg_dots = np.einsum("ij, ij -> i", seg_uvecs[:-1], seg_uvecs[1:])
        iseg_angles = np.append(np.arccos(np.clip(iseg_dots, -1.0, 1.0)), 0)
        inds = np.nonzero(iseg_angles)[0]
    #     print(f"Angles: {np.round(np.degrees(iseg_angles))}")
    # print(f"{iseg_angles[inds]=}")
    # print(f"{(seg_accum_norms)[inds]=}")
    # print(f"{np.exp(-seg_accum_norms[inds] / total_length)=}")
    # print(
    #     "\nWeighted angles:"
    #     f" {iseg_angles[inds] * np.exp(-seg_accum_norms[inds] / total_length)}\n"
    # )
    return (iseg_angles[inds] * np.exp(-seg_accum_norms[inds] / total_length)).sum()


@Timer(name="halflinechunk_pareto_weighted_total_curvature2")
def halflinechunk_pareto_weighted_total_curvature2(hlc: LineString, box_h: float) -> float:
    pts = shp.get_coordinates(hlc)
    seg_lengths = np.sqrt(np.einsum("ij, ij -> i", np.diff(pts, axis=0), np.diff(pts, axis=0)))
    length = shp.length(hlc)
    N = 1000
    sample_length = length / N
    j = 0
    startpt = np.empty_like(pts, shape=(N, 2))
    endpt = np.empty_like(pts, shape=(N, 2))
    endpt[0] = pts[j]
    j += 1
    for i in range(1, N):
        startpt[i] = endpt[i - 1]
        endpt[i] = shp.get_coordinates(hlc.interpolate(i * sample_length))[0]


def approximate_entropy(series, m, r):
    """
    Compute the approximate entropy for a given r.
    """
    N = len(series)
    dist_mat = np.abs(
        np.array(
            [
                series[i : i + m + 1] - series[j : j + m + 1]
                for i in range(N - m)
                for j in range(N - m)
            ]
        )
    ).max(1)
    C_r = np.mean(dist_mat <= r)
    return -np.log(C_r) if C_r > 0 else float("inf")


def optimal_r(series, m):
    """
    Finds an optimal r value that maximizes Approximate Entropy.
    """
    std_dev = np.std(series)
    r_values = np.linspace(0.1 * std_dev, 0.5 * std_dev, 20)
    # Using parallel processing to find the optimal r that maximizes approximate entropy
    ap_entropies = Parallel(n_jobs=-1)(
        delayed(approximate_entropy)(series, m, r) for r in r_values
    )
    max_index = np.argmax(ap_entropies)
    return r_values[max_index]


def sample_entropy(series, m, r=None):
    """
    Compute the Sample Entropy of a given time series data.
    If r is not provided, r is optimized based on maximizing approximate entropy.

    Parameters:
        series (np.array): Input time series data.
        m (int): Length of sequences to compare (embedding dimension).
        r (float, optional): Tolerance for accepting matches. If None, finds optimal r.

    Returns:
        float: The sample entropy of the series.

    References:
    - Delgado-Bonal, A.; Marshak, A. Approximate Entropy and Sample Entropy: A Comprehensive
      Tutorial. Entropy 2019, 21, 541. [link](https://doi.org/10.3390/e21060541)

    Note:
    - SampEn is a measure of the complexity or regularity of a time series.
    - It quantifies the likelihood that similar patterns of length m remain similar
      when the length of the patterns is increased to m+1.
    - A lower SampEn value indicates more regularity or less complexity in the time series.
    - A higher SampEn value indicates more irregularity or more complexity in the time series.
    """
    if r is None:
        r = optimal_r(series, m)

    def _count_matches(dist_mat, r):
        """Count the number of matches within the tolerance r for each template vector."""
        return np.sum(dist_mat <= r) - len(dist_mat)  # Exclude self-match

    N = len(series)
    dist_mat_m = np.abs(
        np.array(
            [
                series[i : i + m] - series[j : j + m]
                for i in range(N - m + 1)
                for j in range(N - m + 1)
            ]
        ).max(1)
    )
    dist_mat_m1 = np.abs(
        np.array(
            [
                series[i : i + m + 1] - series[j : j + m + 1]
                for i in range(N - m)
                for j in range(N - m)
            ]
        ).max(1)
    )

    count_m = _count_matches(dist_mat_m, r) / (N - m) ** 2
    count_m1 = _count_matches(dist_mat_m1, r) / (N - m - 1) ** 2

    return -np.log(count_m1 / count_m) if count_m > 0 else float("inf")


def sampen(L, m, r):
    """
    Calculate the sample entropy (SampEn) of a time series.

    Parameters:
    - L: The time series as a list or array.
    - m: The length of the template.
    - r: The tolerance or similarity criterion.

    Returns:
    - The sample entropy of the time series.

    References:
    - Delgado-Bonal, A.; Marshak, A. Approximate Entropy and Sample Entropy: A Comprehensive
      Tutorial. Entropy 2019, 21, 541. [link](https://doi.org/10.3390/e21060541)

    Note:
    - SampEn is a measure of the complexity or regularity of a time series.
    - It quantifies the likelihood that similar patterns of length m remain similar
      when the length of the patterns is increased to m+1.
    - A lower SampEn value indicates more regularity or less complexity in the time series.
    - A higher SampEn value indicates more irregularity or more complexity in the time series.
    """

    N = len(L)
    B = 0.0
    A = 0.0

    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn
    return -np.log(A / B)


############################
# OLD IMPLEMENTATION BELOW #
############################


def distpt2box(
    CV0: NDArray[Shape["2"], Float],
    CV1: NDArray[Shape["2"], Float],
    CP: NDArray[Shape["2"], Float],
    Pr: NDArray[Shape["2"], Float],
) -> float:
    """
    Calculate the distance between a point and a box defined by two consecutive vertices.

    Parameters:
    - CV0: The vector from the center of the box to V0.
    - CV1: The vector from the center of the box to V1.
    - CP: The vector from the center of the box to the point.
    - Pr: The region of the point P. Region [1, 1] is the region next to V0V1 side.

    Returns:
    - The distance between the point and the box.

    Note:
    - The Pr values represent the regions of the point P relative to the vectors CV0 and CV1.
    - Pr = [1, 1] indicates that the point P is in the sector formed by vectors CV0 and CV1
    - Pr = [1, -1] indicates that the point P is in the sector formed by vectors CV1 and -CV0.
    - Pr = [-1, -1] indicates that the point P is in the sector formed by vectors -CV0 and -CV1.
    - Pr = [-1, 1] indicates that the point P is in the sector formed by vectors -CV1 and CV0.
    - Pr[0] = 0 indicates that the point P is on the line defined by CV0.
    - Pr[1] = 0 indicates that the point P is on the line defined by CV1.
    - The distance is calculated based on the corresponding region of the point P.
    """
    if Pr[0] == 0:
        dP = norm(CP - Pr[1] * CV0)
    elif Pr[1] == 0:
        dP = norm(CP - Pr[0] * CV1)
    else:
        dP = distpt2seg(CP, [Pr[0] * CV0, Pr[1] * CV1])
    return dP  # type: ignore


def norm(V: NDArray[Shape["*"], Float]) -> float:  # noqa: F722
    """
    Calculate the Euclidean norm of a vector.

    Parameters:
    - V: The input vector.

    Returns:
    - The Euclidean norm of the vector.

    Note:
    - This implementation is twice faster than numpy.linalg.norm.
    """
    return hypot(*V)


# Two times faster than the shapely implementation
def distpt2seg(point, segment) -> float:
    """
    Calculate the distance between a point and a line segment.

    Parameters:
    - point: A tuple or list representing the coordinates of the point.
    - segment: A tuple or list representing the coordinates of the line segment endpoints.

    Returns:
    - The distance between the point and the line segment.

    Reference:
    - https://www.geometrictools.com/Documentation/DistancePointLine.pdf

    Note:
    - The distance is calculated using the Euclidean distance formula.
    - If the line segment is degenerate (i.e., the endpoints are the same), the distance between the point and the endpoint is returned.
    - If the projection of the point onto the line segment is outside the segment, the distance to the corresponding endpoint is returned.
    - Otherwise, the closest point on the line segment is calculated, and the distance between the point and the closest point is returned.
    """
    # Convert to numpy arrays for easier manipulation
    point = np.asarray(point)
    segment = np.asarray(segment)

    # Vector from segment start to point
    vec_point = point - segment[0]

    # In case of degenerated segment return distance between two points
    if np.array_equal(segment[0], segment[1]):
        return norm(vec_point)

    # Vector of the segment
    vec_segment = segment[1] - segment[0]

    # Projection of vec_point onto the segment
    projection = np.dot(vec_point, vec_segment)

    # If the projection is outside the segment, return the distance to the corresponding endpoint directly
    if projection <= 0:
        return norm(point - segment[0])
    elif projection >= (len_segment_sq := np.dot(vec_segment, vec_segment)):
        return norm(point - segment[1])

    # Otherwise, the closest point is on the segment
    closest_point = segment[0] + (projection / len_segment_sq) * vec_segment

    # Return the distance between the point and the closest point
    return norm(point - closest_point)


def distseg2seg(
    seg1: NDArray[Shape["2, 2"], Float],
    seg2: NDArray[Shape["2, 2"], Float],
) -> float:
    return ComputeRobust(seg1[0], seg1[1], seg2[0], seg2[1]).distance


def distseg2seg_shapely(
    seg1: NDArray[Shape["2, 2"], Float],
    seg2: NDArray[Shape["2, 2"], Float],
) -> float:
    return ComputeShapely(seg1[0], seg1[1], seg2[0], seg2[1])


def distpt2seg_shapely(
    pt: NDArray[Shape["2"], Float],
    seg: NDArray[Shape["2, 2"], Float],
) -> float:
    return ComputeShapely(seg[0], seg[1], pt, pt)


@dataclass
class Result:
    distance: float = 0.0
    sqrDistance: float = 0.0
    parameter = np.zeros(2)
    closest = np.zeros((2, 2))


def ComputeRobust_matrix(
    segs1: NDArray[Shape["*, 2, 2"], Float],  # noqa: F722
    segs2: NDArray[Shape["*, 2, 2"], Float],  # noqa: F722
):
    result_matrix = np.full((len(segs1), len(segs2)), Result())
    return result_matrix


# Compute the closest points for two segments in nD.
#
# The segments are P[0] + s[0] * (P[1] - P[0]) for 0 <= s[0] <= 1 and
# Q[0] + s[1] * (Q[1] - Q[0]) for 0 <= s[1] <= 1. The D[i] are not required
# to be unit length.
#
# The closest point on segment[i] is stored in closest[i] with parameter[i]
# storing s[i]. When there are infinitely many choices for the pair of
# closest points, only one of them is returned.
#
# The algorithm is robust even for nearly parallel segments. Effectively, it
# uses a conjugate gradient search for the minimum of the squared distance
# function, which avoids the numerical problems introduced by divisions in
# the case the minimum is located at an interior point of the domain. See the
# document
#   https://www.geometrictools.com/Documentation/DistanceLine3Line3.pdf
# for details.
def ComputeRobust(
    P0: NDArray[Shape["2"], Float],
    P1: NDArray[Shape["2"], Float],
    Q0: NDArray[Shape["2"], Float],
    Q1: NDArray[Shape["2"], Float],
):
    result = Result()

    # The code allows degenerate line segments; that is, P0 and P1 can be the same point or
    # Q0 and Q1 can be the same point.  The quadratic function for squared distance between
    # the segment is
    #   R(s,t) = a*s^2 - 2*b*s*t + c*t^2 + 2*d*s - 2*e*t + f
    # for (s,t) in [0,1]^2 where
    #   a = Dot(P1-P0,P1-P0), b = Dot(P1-P0,Q1-Q0), c = Dot(Q1-Q0,Q1-Q0),
    #   d = Dot(P1-P0,P0-Q0), e = Dot(Q1-Q0,P0-Q0), f = Dot(P0-Q0,P0-Q0)
    P1mP0 = P1 - P0
    Q1mQ0 = Q1 - Q0
    P0mQ0 = P0 - Q0
    a = np.dot(P1mP0, P1mP0)
    b = np.dot(P1mP0, Q1mQ0)
    c = np.dot(Q1mQ0, Q1mQ0)
    d = np.dot(P1mP0, P0mQ0)
    e = np.dot(Q1mQ0, P0mQ0)

    # The derivatives dR/ds(i,j) at the four corners of the domain.
    f00 = d
    f10 = f00 + a
    f01 = f00 - b
    f11 = f10 - b

    # The derivatives dR/dt(i,j) at the four corners of the domain.
    g00 = -e
    g10 = g00 - b
    g01 = g00 + c
    g11 = g10 + c

    zero = 0.0
    one = 1.0

    if a > zero and c > zero:
        # Compute the solutions to dR/ds(s0,0) = 0 and dR/ds(s1,1) = 0. The location of sI
        # on the s-axis is stored in classifyI (I = 0 or 1).  If sI <= 0, classifyI is -1.
        # If sI >= 1, classifyI is 1.  If 0 < sI < 1, classifyI is 0. This information
        # helps determine where to search for the minimum point (s,t). The fij values are
        # dR/ds(i,j) for i and j in {0,1}.
        sValue = [GetClampedRoot(a, f00, f10), GetClampedRoot(a, f01, f11)]

        classify = [0, 0]
        for i in range(2):
            if sValue[i] <= zero:
                classify[i] = -1
            elif sValue[i] >= one:
                classify[i] = 1
            else:
                classify[i] = 0

        if classify[0] == -1 and classify[1] == -1:
            # The minimum must occur on s = 0 for 0 <= t <= 1.
            result.parameter[0] = zero
            result.parameter[1] = GetClampedRoot(c, g00, g01)
        elif classify[0] == 1 and classify[1] == 1:
            # The minimum must occur on s = 1 for 0 <= t <= 1.
            result.parameter[0] = one
            result.parameter[1] = GetClampedRoot(c, g10, g11)
        else:
            # The line dR/ds = 0 intersects the domain [0,1]^2 in a
            # nondegenerate segment. Compute the endpoints of that
            # segment, end[0] and end[1]. The edge[i] flag tells you
            # on which domain edge end[i] lives: 0 (s=0), 1 (s=1),
            # 2 (t=0), 3 (t=1).
            edge = [0, 0]
            end = [[0, 0], [0, 0]]
            ComputeIntersection(sValue, classify, b, f00, f10, edge, end)

            # The directional derivative of R along the segment of
            # intersection is
            #   H(z) = (end[1][1]-end[1][0]) *
            #          dR/dt((1-z)*end[0] + z*end[1])
            # for z in [0,1]. The formula uses the fact that
            # dR/ds = 0 on the segment. Compute the minimum of
            # H on [0,1].
            ComputeMinimumParameters(
                edge, end, b, c, e, g00, g10, g01, g11, result.parameter
            )
    else:
        if a > zero:
            # The Q-segment is degenerate (Q0 and Q1 are the same
            # point) and the quadratic is R(s,0) = a*s^2 + 2*d*s + f
            # and has (half) first derivative F(t) = a*s + d.  The
            # closest P-point is interior to the P-segment when
            # F(0) < 0 and F(1) > 0.
            result.parameter[0] = GetClampedRoot(a, f00, f10)
            result.parameter[1] = zero
        elif c > zero:
            # The P-segment is degenerate (P0 and P1 are the same
            # point) and the quadratic is R(0,t) = c*t^2 - 2*e*t + f
            # and has (half) first derivative G(t) = c*t - e.  The
            # closest Q-point is interior to the Q-segment when
            # G(0) < 0 and G(1) > 0.
            result.parameter[0] = zero
            result.parameter[1] = GetClampedRoot(c, g00, g01)
        else:
            # P-segment and Q-segment are degenerate.
            result.parameter[0] = zero
            result.parameter[1] = zero

    result.closest[0] = [
        (one - result.parameter[0]) * P0[0] + result.parameter[0] * P1[0],
        (one - result.parameter[0]) * P0[1] + result.parameter[0] * P1[1],
    ]
    result.closest[1] = [
        (one - result.parameter[1]) * Q0[0] + result.parameter[1] * Q1[0],
        (one - result.parameter[1]) * Q0[1] + result.parameter[1] * Q1[1],
    ]
    diff = [
        result.closest[0][0] - result.closest[1][0],
        result.closest[0][1] - result.closest[1][1],
    ]
    result.sqrDistance = np.dot(diff, diff)
    result.distance = sqrt(result.sqrDistance)
    return result


def GetClampedRoot(slope, h0, h1):
    """Return the root of the line segment that is clamped to [0,1].

    Compute the root of h(z) = h0 + slope*z and clamp it to the interval [0,1]. It is
    required that for h1 = h(1), either (h0 < 0 and h1 > 0) or (h0 > 0 and h1 < 0).

    Theoretically, r is in (0,1). However, when the slope is
    nearly zero, then so are h0 and h1. Significant numerical
    rounding problems can occur when using floating-point
    arithmetic. If the rounding causes r to be outside the
    interval, clamp it. It is possible that r is in (0,1) and has
    rounding errors, but because h0 and h1 are both nearly zero,
    the quadratic is nearly constant on (0,1). Any choice of p
    should not cause undesirable accuracy problems for the final
    distance computation.

    NOTE: You can use bisection to recompute the root or even use
    bisection to compute the root and skip the division. This is
    generally slower, which might be a problem for high-performance
    applications.
    """
    zero = 0.0
    one = 1.0

    if h0 < zero:
        if h1 > zero:
            r = -h0 / slope
            if r > one:
                r = 0.5
            # The slope is positive and -h0 is positive, so there is
            # no need to test for a negative value and clamp it.
        else:
            r = one
    else:
        r = zero

    return r


def ComputeIntersection(s_value, classify, b, f00, f10, edge, end):
    """Compute the intersection of the line dR/ds = 0 with the domain [0,1]^2. The direction
    of the line dR/ds is conjugate to (1,0), so the algorithm for minimization is
    effectively the conjugate gradient algorithm for a quadratic function.

    The divisions are theoretically numbers in [0,1]. Numerical
    rounding errors might cause the result to be outside the
    interval. When this happens, it must be that both numerator
    and denominator are nearly zero. The denominator is nearly
    zero when the segments are nearly perpendicular. The
    numerator is nearly zero when the P-segment is nearly
    degenerate (f00 = a is small). The choice of 0.5 should not
    cause significant accuracy problems.

    NOTE: You can use bisection to recompute the root or even use
    bisection to compute the root and skip the division. This is
    generally slower, which might be a problem for high-performance
    applications.
    """
    zero = 0.0
    half = 0.5
    one = 1.0

    if classify[0] < 0:
        edge[0] = 0
        end[0][0] = zero
        end[0][1] = f00 / b
        if end[0][1] < zero or end[0][1] > one:
            end[0][1] = half

        if classify[1] == 0:
            edge[1] = 3
            end[1][0] = s_value[1]
            end[1][1] = one
        else:  # classify[1] > 0
            edge[1] = 1
            end[1][0] = one
            end[1][1] = f10 / b
            if end[1][1] < zero or end[1][1] > one:
                end[1][1] = half
    elif classify[0] == 0:
        edge[0] = 2
        end[0][0] = s_value[0]
        end[0][1] = zero

        if classify[1] < 0:
            edge[1] = 0
            end[1][0] = zero
            end[1][1] = f00 / b
            if end[1][1] < zero or end[1][1] > one:
                end[1][1] = half
        elif classify[1] == 0:
            edge[1] = 3
            end[1][0] = s_value[1]
            end[1][1] = one
        else:
            edge[1] = 1
            end[1][0] = one
            end[1][1] = f10 / b
            if end[1][1] < zero or end[1][1] > one:
                end[1][1] = half
    else:  # classify[0] > 0
        edge[0] = 1
        end[0][0] = one
        end[0][1] = f10 / b
        if end[0][1] < zero or end[0][1] > one:
            end[0][1] = half

        if classify[1] == 0:
            edge[1] = 3
            end[1][0] = s_value[1]
            end[1][1] = one
        else:
            edge[1] = 0
            end[1][0] = zero
            end[1][1] = f00 / b
            if end[1][1] < zero or end[1][1] > one:
                end[1][1] = half


def ComputeMinimumParameters(edge, end, b, c, e, g00, g10, g01, g11, parameter):
    """Compute the location of the minimum of R on the segment of intersection for the line
    dR/ds = 0 and the domain [0,1]^2.
    """
    zero = 0.0
    one = 1.0
    delta = end[1][1] - end[0][1]
    h0 = delta * (-b * end[0][0] + c * end[0][1] - e)
    if h0 >= zero:
        if edge[0] == 0:
            parameter[0] = zero
            parameter[1] = GetClampedRoot(c, g00, g01)
        elif edge[0] == 1:
            parameter[0] = one
            parameter[1] = GetClampedRoot(c, g10, g11)
        else:
            parameter[0] = end[0][0]
            parameter[1] = end[0][1]
    else:
        h1 = delta * (-b * end[1][0] + c * end[1][1] - e)
        if h1 <= zero:
            if edge[1] == 0:
                parameter[0] = zero
                parameter[1] = GetClampedRoot(c, g00, g01)
            elif edge[1] == 1:
                parameter[0] = one
                parameter[1] = GetClampedRoot(c, g10, g11)
            else:
                parameter[0] = end[1][0]
                parameter[1] = end[1][1]
        else:  # h0 < 0 and h1 > 0
            z = min(max(h0 / (h0 - h1), zero), one)
            omz = one - z
            parameter[0] = omz * end[0][0] + z * end[1][0]
            parameter[1] = omz * end[0][1] + z * end[1][1]


def ComputeShapely(
    P0: NDArray[Shape["2"], Float],
    P1: NDArray[Shape["2"], Float],
    Q0: NDArray[Shape["2"], Float],
    Q1: NDArray[Shape["2"], Float],
):
    LSP = shp.LineString([P0, P1])
    LSQ = shp.LineString([Q0, Q1])

    return LSP.distance(LSQ)
