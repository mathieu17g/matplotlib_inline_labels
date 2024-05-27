from dataclasses import dataclass, field
from shapely import LineString, Point, Polygon, get_coordinates
from math import isclose
from typing import Literal, NewType

#####################################################
# Data structures used by label placement algorithm #
#####################################################

############################################################
# Data structures used for the geometric representation of #
# - graph lines,                                           #
# - labels' (centers) position candidates and              #
# - labels' bounding boxes                                 #
############################################################

"""Tolerance used to identify curve closure"""
CCTOL = 1e-9


# TODO: create and use a lcisclosed function and/or attribute using CCTOL
def lc_isclosed(lc: LineString) -> bool:
    """Check if a line chunk is closed

    Args:
    - lc: LineString to be checked

    Returns:
    - True if the line chunk is closed
    - False otherwise
    """
    if lc.is_closed:
        return True
    else:
        coords = get_coordinates(lc)
        return isclose(
            coords[0][0], coords[-1][0], rel_tol=CCTOL, abs_tol=CCTOL
        ) and isclose(coords[0][1], coords[-1][1], rel_tol=CCTOL, abs_tol=CCTOL)


@dataclass
class SIFLineChunkGeom:
    """Self-intersection free line chunk defined by
    - line continuous data subsets (NaN values create gaps in the line chunk)
    - clipping by plot area
    - splitting around self-intersections
    - splitting into individual points and linestrings geometries

    Attributes:
    - `lc` (LineString | Point): Line chunk geometry"""

    lc: LineString | Point
    """Line chunk geometry"""


SIFLineChunkGeomsList = NewType("SIFLineChunkGeomsList", list[LineString | Point])
"""List of geometries associated with one labelled line
- continuous data line chunks (NaN values create gaps in the line chunk)
- clipping by plot area
- splitting around self-intersections
- splitting into individual points and linestrings geometries

Type: `list[LineString | Point]`"""


# TODO2 Once a densier separation value will be calculated for each PRc, enabling to get rid
# TODO2 of the PRc adjacency considerations when choosing the final PRc, DELETE the sd:
# TODO2 sampling distance parameter
@dataclass
class IFLineChunkGeoms:
    """Line chunk associated geometries in cartesian Axe coordinates

    Attributes:
    - `lc`: Line chunk geometry
    - `lcb`: Line chunk buffered with Line2D width
    - `pcl`: Label box center candidate list
    - `lc_sds`: Line chunk adjusted sampling distance per corresponding candidates indices
    range in the label's box's center candidates list
    - `re`: Rotation estimates for each candidates in radians
    """

    lc: LineString | Point
    """Line chunk geometry"""
    lcb: Polygon
    """Line chunk buffered with Line2D width"""
    pcl: list[Point] = field(default_factory=list)
    """Label box center candidate list"""
    lc_sds: dict[slice, float] = field(default_factory=dict)
    """Line chunk adjusted sampling distance per corresponding candidates indices range in 
    the label's box's center candidates list"""
    re: list[float] = field(default_factory=list)
    """Rotation estimate for each candidates in radians"""


IFLineChunkGeomsList = NewType("IFLineChunkGeomsList", list[IFLineChunkGeoms])
"""List of geometries associated with one labelled line intersection free line chunks

Type: `list[IFLineChunkGeoms]`"""


@dataclass
class LabelBBDims:
    """Label's bounding box dimensions

    Attributes:
        `box_w`: Label's bounding box width
        `box_h`: Label's bounding box height
    """

    w: float = 0.0
    """Label's bounding box width"""
    h: float = 0.0
    """Label's bounding box height"""


@dataclass
class LabelledLineGeometricData:
    """Labelled line geometric data composed of line chunks' geometries and label bounding box dimensions

    Attributes:
    - `centroid`: Label's line chunks' centroid
    - `siflcgl`: Self-Intersection Free Line Chunks' associated Geometries List
    - `iflcgl`: Intersection Free Line Chunks' associated Geometries List
    - `if2siflcgl_inds`: Indices of the intersection free line chunks in the self-intersection free line chunks list
    - `boxd`: Label's Bounding Box dimensions

    Args:
    - `centroid`: Label's line chunks' centroid
    - `siflcgl`: Self-Intersection Free Line Chunks' associated Geometries List
    - `iflcgl`: Intersection Free Line Chunks' associated Geometries List
    - `if2siflcgl_inds`: Indices of the intersection free line chunks in the self-intersection free line chunks list
    - `boxd`: Label's Bounding Box dimensions
    """

    centroid: Point
    """Label's line chunks' centroid"""
    siflcgl: SIFLineChunkGeomsList = field(
        default_factory=lambda: SIFLineChunkGeomsList([])
    )
    """Self-Intersection Free Line Chunks' associated Geometries List"""
    iflcgl: IFLineChunkGeomsList = field(default_factory=lambda: IFLineChunkGeomsList([]))
    """Intersection Free Line Chunks' associated Geometries List"""
    if2siflcgl_inds: list[int] = field(default_factory=list)
    """Indices of the intersection free line chunks in the self-intersection free line chunks list"""
    boxd: LabelBBDims = field(default_factory=LabelBBDims)
    """Label's Bounding Box dimensions"""


Labelled_Lines_Geometric_Data_Dict = NewType(
    "Labelled_Lines_Geometric_Data_Dict", dict[str, LabelledLineGeometricData]
)
"""Dictionary of labelled lines' geometric data indexed by labels' text

Type: `dict[str, Labelled_Line_Geometric_Data]`"""

Label_Rotation_Estimates_Dict = NewType(
    "Label_Rotation_Estimates_Dict", dict[str, list[list[int]]]
)
""" Labels' rotation estimates as indices of `rotation_samples`"""

###########################################################################
# Data structures used to track the search for labels' suitable rotations #
###########################################################################


@dataclass
class Rotation_Search_State:
    """Mono direction label rotation angle search state. A valid rotation is when the
    label's bounding box has a good alignment with the current line chunk and does not
    overlap with geometries other than the the current line chunk.

    Attributes:
        `search`: Search to be continued status
        `first`: First valid rotation found
        `last`: Last valid rotaiton found
    """

    search: bool | None = None
    """Search to be continued status"""
    first: float | None = None
    """First valid rotation found"""
    last: float | None = None
    """Last valid rotation found"""


Sign = Literal[1, -1]


# Bi direction (1: ccw, -1:cw) label rotation angle search state
Rotation_BiSearch_State = NewType(
    "Rotation_BiSearch_State", dict[Sign, Rotation_Search_State]
)
"""Bi direction label best rotation angle search state (ccw (+1) and cw (-1))

Dictionary with:
- `key: Sign` search direction (ccw (+1) and cw (-1)
- `value: Rotation_Search_State` search active? Plus, first and last valid rotation found
"""


#############################################################
# Data structures used collecting label rotation candidates #
#############################################################


@dataclass
class Label_R:
    """Label rotation angle candidate with alignment error

    Attributes:
        `rot`: label rotation angle in radians
        `err`: label alignment error on its curve"""

    rot: float
    """Label rotation angle in radians"""
    aerr: float
    """Label alignment error on its curve"""
    osep: float | None
    """Label separation distance from other graphic geometries"""


Label_Rs = NewType("Label_Rs", list[Label_R])
"""List of label rotation candidates with their corresponding label alignment errors"""


@dataclass
class Label_PmR:
    """Label center position with multiple rotations candidates, with their corresponding label alignment errors

    Args:
    - pos: label center position
    - theta_candidates: list of label rotation candidates with their corresponding label alignment errors
    """

    pos: Point
    """Label center position"""
    rot_cands: Label_Rs
    """List of label rotation candidates with their corresponding label alignment errors"""


@dataclass
class Label_PRc:
    """Label center position and rotation candidate

    Attributes:
    - `pos`: label center position in 2D coordinates
    - `rot`: label rotation for the center position with minimal alignment error
    - `aerr`: label alignment error
    - `osep`: label's separation from graphical objects other than its line chunk
    - `fsep`: label's separation from its line chunk folds
    - `isep`: label's separation from its line chunk curvature variations (information)
    - `csep`: label's separation from the line chunks' centroid
    """

    pos: Point
    """Label center position in 2D coordinates"""
    rot: float | None
    """Label rotation angle value in radians"""
    aerr: float | None
    """Label alignment error with its curve"""
    osep: float | None = None
    """Label's separation from graphical objects other than its line chunk"""
    fsep: float | None = None
    """Label's separation from its line chunk folds"""
    isep: float | None = None
    """Label's separation from its line chunk curvature variations (information)"""
    csep: float | None = None
    """Label's separation from the line chunks' centroid"""


Label_PRcs = NewType("Label_PRcs", list[Label_PRc])


@dataclass
class Label_PR:
    """Position and Rotation selection for a label after selection process among candidates

    Attributes:
    - `cpt`: Label's center position
    - `rot`: Label's rotation in radian
    """

    cpt: Point
    """Label's center position"""
    rot: float
    """Label's rotation"""


Labels_lcs_adjPRcs_groups = NewType(
    "Labels_lcs_adjPRcs_groups", dict[str, dict[int, list[Label_PRcs]]]
)
"""Set of adjacent Postition and Rotation candidates clustered by:
- label: refering to a `Labelled_Line_Geometric_Data` in the corresponding `Labelled_Lines_Geometric_Data_Dict`
- line chunk index: refering to a  `Line_Chunk_Geometries` with the corresponding `Labelled_Line_Geometric_Data.lcgl` list
- `Label_PRc` adjacencies relatively to each sub line chunk (defined in preprocessing) sampling distance
"""

Labels_PRcs = NewType("Labels_PRcs", dict[str, Label_PRcs])
"""Flat list of labels' Postition and Rotation candidates"""

########################################################################
# Datastructures for labels' position and rotation candidate selection #
########################################################################

PLACEMENT_ALGORITHM_OPTIONS = Literal["basic", "advanced"]

Label_Inlining_Solutions = NewType("Label_Inlining_Solutions", dict[str, Label_PR])
"""Solutions to the label inlining problem. 

Built on `dict[str, Label_PR]` with:
- `keys: str`: label text
- `values: Label_PR`: label position and rotation

`Label_PR` being a dataclass with attributes:
- `cpt`: Label's center position 
- `rot`: Label's rotation in radian
"""


@dataclass
class SepLongestSubgroups:
    """Structure used to find the longest adjacent Postion and Rotation candidates sub group
    on all a line chunk of a line for a particular separation level"""

    card: int = -1
    """Cardinality of the longest adjacent Postion and Rotation 
    candidates for a particular separation level"""
    length: float = -1
    """Linear length along line chunk for the longest adjacent Postion and Rotation 
    candidates for a particular separation level"""
    sgd: dict[tuple[int, int], list[list[int]]] = field(
        default_factory=lambda: dict[tuple[int, int], list[list[int]]]()
    )
    """Longest adjacent PRc subgroups indices (in line chunks' adjacent PRc groups indices), 
    indexed by:
    - line chunk index within `Line_Chunk_Geometries_List` of a `Labelled_Line_Geometric_Data`
    - adjacent PRc subgroup (`Label_PRcs`) index within the line chunk's adjacent PRc subgroups list"""
