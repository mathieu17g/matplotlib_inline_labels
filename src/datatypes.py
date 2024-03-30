from dataclasses import dataclass, field
from shapely import LineString, Point, Polygon
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


# TODO2 Once a densier separation value will be calculated for each PRc, enabling to get rid
# TODO2 of the PRc adjacency considerations when choosing the final PRc, DELETE the sd:
# TODO2 sampling distance parameter
@dataclass
class Line_Chunk_Geometries:
    """Line chunk associated geometries in cartesian Axe coordinates

    Attributes:
    - `lc`: Line chunk geometry
    - `lcb`: Line chunk buffered with Line2D width
    - `pcl`: Label box center candidate list
    - `slc_sds`: Line chunk adjusted sampling distance per corresponding candidates indices
    range in the label's box's center candidates list
    - `re`: Rotation estimates for each candidates in radians
    """

    lc: LineString | Point
    """Line chunk geometry"""
    lcb: Polygon
    """Line chunk buffered with Line2D width"""
    pcl: list[Point] = field(default_factory=list)
    """Label box center candidate list"""
    slc_sds: dict[slice, float] = field(default_factory=dict)
    """Line chunk adjusted sampling distance per corresponding candidates indices range in 
    the label's box's center candidates list"""
    re: list[float] = field(default_factory=list)
    """Rotation estimate for each candidates in radians"""


@dataclass
class Label_Box_Dimensions:
    """Label's bounding box dimensions

    Attributes:
        `box_w`: Label's bounding box width
        `box_h`: Label's bounding box height
    """

    w: float = 0.0
    """Label's bounding box width"""
    h: float = 0.0
    """Label's bounding box height"""


Line_Chunk_Geometries_List = NewType(
    "Line_Chunk_Geometries_List", list[Line_Chunk_Geometries]
)
"""List of geometries associated with one labelled line intersection free line chunks

Type: `list[Line_Chunk_Geometries]`"""


@dataclass
class Labelled_Line_Geometric_Data:
    """Labelled line geometric data composed of line chunks' geometries label bounding box dimensions

    Attributes:
    - `lcgl`: Label's line chunks' associated geometries list
    - `boxd`: Label's bounding box dimensions
    """

    lcgl: Line_Chunk_Geometries_List = field(
        default_factory=lambda: Line_Chunk_Geometries_List([])
    )
    """Intersection free line chunks' associated geometries list"""
    boxd: Label_Box_Dimensions = field(default_factory=Label_Box_Dimensions)
    """Label's bounding box dimensions"""


Labelled_Lines_Geometric_Data_Dict = NewType(
    "Labelled_Lines_Geometric_Data_Dict", dict[str, Labelled_Line_Geometric_Data]
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
    err: float
    """Label alignment error on its curve"""
    sep: float | None
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
    - `align_err`: label alignment error
    - `sep`: label's separation from other graphical objects
    - `c_dte`: linear distance from center to the edges of its adjacency subset for the highest
      separation buffer with candidates
    """

    pos: Point
    """Label center position in 2D coordinates"""
    rot: float | None
    """Label rotation angle value in radians"""
    align_err: float | None
    """Label alignment error with its curve"""
    sep: float | None = None
    """Label's separation from other graphical objects"""
    # TODO: reconsider the utility of distance below
    c_dte: float | None = None
    """Linear distance of center to the edges of its adjacency subset for the highest 
    separation buffer with candidates"""


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
