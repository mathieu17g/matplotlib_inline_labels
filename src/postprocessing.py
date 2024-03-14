from datatypes import (
    Label_Inlining_Solutions,
    Labels_lcPRcs,
    Labelled_Lines_Geometric_Data_Dict,
    Label_PR,
    Labels_PRcs,
)
from typing import Tuple, NamedTuple
from processing import SEP_LEVELS
from math import isclose
import shapely as shp
import numpy as np

#! BEGIN algo1 imports
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from scipy.spatial import ConvexHull
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

#! END algo1 imports


def argmax(values, keyfunc=None):
    """Equivalent of numpy argmax on a python iterable"""
    if keyfunc is None:
        return max((x for x in enumerate(values)), key=lambda x: x[1])[0]
    else:
        return max((x for x in enumerate(values)), key=lambda x: keyfunc(x[1]))[0]


# TODO: refine docstring
def solselect_monocrit(
    linelikeLabels: list[str],
    graph_labels_lcPRcs: Labels_lcPRcs,
    ld: Labelled_Lines_Geometric_Data_Dict,
) -> Tuple[Label_Inlining_Solutions, list[str]]:
    """For each label, takes the middle of the longuest continuous label position candidate
    list with the highest separation label found"""
    # Initialize best (label center) position candidate, per label dictionary
    lis = Label_Inlining_Solutions({})

    for label in (label for label in linelikeLabels if graph_labels_lcPRcs[label]):
        # Search for the longest continuous label position candidate list starting from the
        # best separation level to the least
        for bf in SEP_LEVELS:
            bf_PRc_masks = list[list[bool]]([])
            for lcPRcs in graph_labels_lcPRcs[label]:
                bf_PRc_masks += [
                    [
                        PRc.rot is not None and PRc.sep is not None and PRc.sep == bf
                        for PRc in lcPRcs
                    ]
                ]

            # Handle the closed line chunk where the 1st and last PRc are adjacent
            # relatively to the sampling distance and have the same separation but are at
            # the two ends of a PRc cluster
            # This should occur only when there is one line chunk for the label and one
            # cluster of PRcs for that line chunk, regarding the clustering done at the end
            # of processing stage
            if (
                len(bf_PRc_masks) == 1
                and len(bf_PRc_masks[0]) > 1
                and bf_PRc_masks[0][0]
                and bf_PRc_masks[0][-1]
                and sum(bf_PRc_masks[0]) < len(bf_PRc_masks[0])
            ):
                lcPRcs = graph_labels_lcPRcs[label][0]
                firstPRc = lcPRcs[0]
                lastPRc = lcPRcs[-1]
                lcg = ld[label].lcgl[0]
                assert lcg.lc is not None
                if isclose(lcg.lc.coords[0][0], lcg.lc.coords[-1][0]) and isclose(
                    lcg.lc.coords[0][1], lcg.lc.coords[-1][1]
                ):
                    d1 = shp.line_locate_point(lcg.lc, firstPRc.pos)
                    d2 = shp.line_locate_point(lcg.lc, lastPRc.pos)
                    d = shp.length(lcg.lc) - max(d1, d2) + min(d1, d2)
                    assert len(lcg.slc_sds) == 1
                    if d < 1.1 * list(lcg.slc_sds.values())[0]:
                        # Indices of PRc candidates with a separation equal to buffering
                        # factor
                        inds = np.nonzero(bf_PRc_masks[0])[0]
                        # Groups of continuous indices
                        group1, group2 = np.split(
                            inds, np.nonzero((np.diff(inds) > 1))[0] + 1
                        )
                        group = group2.tolist() + group1.tolist()
                        best_ind = group[len(group) // 2]
                        best_PRc = graph_labels_lcPRcs[label][0][best_ind]
                        lis[label] = Label_PR(
                            best_PRc.pos,
                            best_PRc.rot,  # pyright: ignore[reportArgumentType]
                        )
                        break

            bf_longestgroups = list[list[int]]([])
            for bf_lcPRc_mask in bf_PRc_masks:
                if sum(bf_lcPRc_mask) > 0:
                    # Indices of PRc candidates with a separation equal to buffering factor
                    inds = np.nonzero(bf_lcPRc_mask)[0]
                    # Groups of continuous indices
                    groups = np.split(inds, np.nonzero((np.diff(inds) > 1))[0] + 1)
                    # Longest group of continuous indices
                    bf_longestgroups += [max(groups, key=len).tolist()]
                else:
                    bf_longestgroups += [[]]

            if max((len(g) for g in bf_longestgroups)) > 0:
                # line chunk index with the longest group of adjacent PRcs with
                # separation == bf
                imaxlen = argmax(bf_longestgroups, keyfunc=len)
                # PRcs indices, within the line chunk PRcs, of the longest group
                lg_subinds = bf_longestgroups[imaxlen]
                # Center index of the latter group
                best_lcPRc_subind = (lg_subinds[0] + lg_subinds[-1]) // 2
                # index, within the line chunk PRcs, of the longest group center
                best_PRc = graph_labels_lcPRcs[label][imaxlen][best_lcPRc_subind]
                assert best_PRc is not None
                lis[label] = Label_PR(
                    best_PRc.pos,
                    best_PRc.rot,  # pyright: ignore[reportArgumentType]
                )
                break

    # Initialize legend content for labels than cannot be inlined properly on their curve
    legend_labels = []

    for label in list(graph_labels_lcPRcs):
        if label not in list(lis):
            # Add label to the list of labels with no non overlapping placement
            # available. To be drawn in a standard legend
            legend_labels.append(label)

    return lis, legend_labels


def solution_selection_algo1(linelikeLabels: list[str], graph_labels_PRcs: Labels_PRcs):
    #! Minimise label centers' convex hull area and distance of centers from their
    #! adjacent non overlaping solutions set boundaries
    # First identify labels with non overlapping solutions and for each ones a set of position candidates with distance to their adjacency subset boundaries

    # PCWD = namedtuple("PCWD", ["xy", "d"])  # Position Candiate with its distance to its adjacent subset boundaries
    PCWD = NamedTuple(
        "PCWD", [("xy", tuple[float, float]), ("d", float)]
    )  # Position Candidate with its distance to its adjacent subset boundaries
    PCWDLD = dict[str, list[PCWD]]
    pcwdld = PCWDLD()

    # Initialize legend content for labels than cannot be inlined properly on their curve
    legend_labels = []

    #! MLS ALTERNATIVE
    #! UNFINISHED... move from graph_labels_PRcs to graph_labels_lcPRcs or wait for the
    #! introduction of a distance to other geometries to be introduced
    # For each PRc compute its distance to the edge of the minimum seperation adjacency
    # group it belongs to
    for label in linelikeLabels:
        pcwdld[label] = []
        cont_grps_dict = list[list[int]]([])
        print(f"{SEP_LEVELS=}")
        minsep_PRc_mask = [
            PRc is not None and PRc.sep is not None and PRc.sep == min(SEP_LEVELS)
            for PRc in graph_labels_PRcs[label]
        ]
        if sum(minsep_PRc_mask) == 0:
            # Add label to the list of labels with no non overlapping placement
            # available. To be drawn in a standard legend
            legend_labels.append(label)
            # Remove it from labels entries in Labels_PRcs structure #? Maybe useless
            del graph_labels_PRcs[label]
        else:
            # Indices of PRc candidates with a separation equal to buffering factor
            inds = np.nonzero(minsep_PRc_mask)[0]
            # Groups of continuous indices
            groups = np.split(inds, np.nonzero((np.diff(inds) > 1))[0] + 1)
            cont_grps_dict = [g.tolist() for g in groups]

        for i, PRc in enumerate(graph_labels_PRcs[label]):
            if PRc is not None and PRc.sep is not None:
                # Find the iso sep group the PRc belongs to
                cont_grp = next(g for g in cont_grps_dict if i in g)
                # Compute the distance to the edges with the symplifying hypothesis that
                # PRc are evenly placed along the considered line chunk, which is not
                # the case
                dist = min(i - cont_grp[0], cont_grp[-1] - i)
                PRc.c_dte = dist
                graph_labels_PRcs[label][i] = PRc

    class Algo1(ElementwiseProblem):
        def __init__(self):
            super().__init__(
                n_var=len(pcwdld),
                n_obj=2,
                n_ieq_constr=0,
                xl=[0] * len(pcwdld),
                xu=[len(pcwdl) - 1 for pcwdl in pcwdld.values()],
                vtype=int,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            pcwdl = [list(pcwdld.values())[i][x[i]] for i in range(len(pcwdld))]
            out["F"] = [
                ConvexHull([pcwd.xy for pcwd in pcwdl]).area,
                sum([pcwd.d for pcwd in pcwdl]),
            ]

    problem = Algo1()

    algorithm = NSGA2(
        pop_size=100,
        sampling=IntegerRandomSampling(),
        crossover=SBX(prob=1.0, eta=3, vtype=float, repair=RoundingRepair()),
        mutation=PM(prob=1.0, eta=3, vtype=float, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    res = minimize(
        problem,
        algorithm,
        # ("n_gen", 1000),
        seed=1,
    )
    # print(f"Best solution found: {res.X}")
    print(f"Number of best solution: {len(res.X)}")  # type: ignore
    # print(f"Function value: {res.F}")
    # print(f"Constraint violation: {res.CV}")

    # for label in list(pcwdld):
    # res.X[1]
