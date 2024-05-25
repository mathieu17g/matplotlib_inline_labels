from .datatypes import (
    Label_Inlining_Solutions,
    Labels_lcs_adjPRcs_groups,
    Labelled_Lines_Geometric_Data_Dict,
    Label_PR,
    SepLongestSubgroups,
)

from .processing import SEP_LEVELS
from math import isclose
import shapely as shp
from shapely import LineString
import numpy as np
from .utils import Timer



def argmaxs(values, keyfunc=None):
    """Return indices of maximum values of a python iterable and the corresponding maximum
    value"""
    if keyfunc is None:
        m = max(values)
        return [i for i, v in enumerate(values) if v == m], m
    else:
        m = keyfunc(max(values, key=keyfunc))
        return [i for i, v in enumerate(values) if keyfunc(v) == m], m


# TODO: refine docstring
@Timer(name="solselect_monocrit")
def solselect_monocrit(
    linelikeLabels: list[str],
    graph_lbls_lcs_adjPRcs_grps: Labels_lcs_adjPRcs_groups,
    ld: Labelled_Lines_Geometric_Data_Dict,
) -> tuple[Label_Inlining_Solutions, list[str]]:
    """For each label, and for the highest separation label found, takes the middle of the 
    longuest continuous label position candidate list. And in case there are several, takes 
    the one with the longuest linear distance on the corresponding line chunk"""
    # Initialize best (label center) position candidate, per label dictionary
    lis = Label_Inlining_Solutions({})

    for label in (label for label in linelikeLabels if graph_lbls_lcs_adjPRcs_grps[label]):
        # Search for the longest continuous label position candidate list starting from the
        # best separation level to the least
        for sep in SEP_LEVELS:
            lcs_sep_PRc_masks = dict[int, list[list[bool]]]({})
            for lc_idx, PRc_adjgrps in graph_lbls_lcs_adjPRcs_grps[label].items():
                # Masks for each line chunk selecting the PRcs corresponding to the current
                # separation level
                lcs_sep_PRc_masks |= {
                    lc_idx: [
                        [
                            PRc.rot is not None and PRc.sep is not None and PRc.sep == sep
                            for PRc in PRc_adjgrp
                        ]
                        for PRc_adjgrp in PRc_adjgrps
                    ]
                }

            sep_longestgrps = SepLongestSubgroups()

            for sep_lcPRc_mask, lc_idx, lc_adjgrp_idx in (
                (sep_lcPRc_mask, lc_idx, lc_adjgrp_idx)
                for lc_idx, sep_lcPRc_masks in lcs_sep_PRc_masks.items()
                for lc_adjgrp_idx, sep_lcPRc_mask in enumerate(sep_lcPRc_masks)
                if sum(sep_lcPRc_mask) > 0
            ):
                lcg = ld[label].lcgl[lc_idx]
                lc = ld[label].lcgl[lc_idx].lc
                assert isinstance(lc, LineString)
                PRcs = graph_lbls_lcs_adjPRcs_grps[label][lc_idx][lc_adjgrp_idx]
                # Indices of PRc candidates with a separation equal to sep level
                inds = np.nonzero(sep_lcPRc_mask)[0]
                # Groups of continuous indices
                grps = np.split(inds, np.nonzero((np.diff(inds) > 1))[0] + 1)
                grps = [list(grp) for grp in grps]
                # If there are more than 1 group and the linechunk is closed, check
                # if first PRc of the first group and last PRc of the last group are
                # adjacent according to the adjusted sampling distance of the line
                # chunk. If yes, merge last and first groups
                if (
                    len(grps) > 1
                    and isclose(lc.coords[0][0], lc.coords[-1][0])
                    and isclose(lc.coords[0][1], lc.coords[-1][1])
                ):
                    d1 = shp.line_locate_point(lc, PRcs[grps[0][0]].pos)
                    d2 = shp.line_locate_point(lc, PRcs[grps[-1][-1]].pos)
                    d = shp.length(lc) - max(d1, d2) + min(d1, d2)
                    assert len(lcg.slc_sds) == 1
                    if d < 1.1 * list(lcg.slc_sds.values())[0]:
                        grps = [grps[-1] + grps[0]] + [
                            grp for grp in grps if grp not in [grps[0], grps[-1]]
                        ]
                # Calculate groups cardinality
                cards = [len(g) for g in grps]
                # Calculate groups length along the corresponding line chunk
                lengths: list[float] = [
                    (lc.project(PRcs[g[-1]].pos) - lc.project(PRcs[g[0]].pos)) % lc.length
                    for g in grps
                ]
                # Add longests groups of continuous indices if maximum group linear length
                # is equal to the one of previous found groups or replace them if stricly greater
                maxcard = max(cards)
                maxlength = max(lengths)
                maxc_grps_inds, maxcard = argmaxs(cards)
                maxl_grps_inds, maxlength = argmaxs(
                    list(enumerate(lengths)),
                    keyfunc=lambda il: (il[1] if il[0] in maxc_grps_inds else -1.0),
                )
                if maxcard > sep_longestgrps.card:
                    sep_longestgrps.card = maxcard
                    sep_longestgrps.length = maxlength
                    sep_longestgrps.sgd |= {
                        (lc_idx, lc_adjgrp_idx): [grps[i] for i in maxl_grps_inds]
                    }
                if maxcard == sep_longestgrps.card:
                    if maxlength > sep_longestgrps.length:
                        sep_longestgrps.length = maxlength
                        sep_longestgrps.sgd |= {
                            (lc_idx, lc_adjgrp_idx): [grps[i] for i in maxl_grps_inds]
                        }
                    if maxlength == sep_longestgrps.length:
                        sep_longestgrps.sgd = {
                            (lc_idx, lc_adjgrp_idx): [grps[i] for i in maxl_grps_inds]
                        }

            if sep_longestgrps.sgd:
                # TODO: in case of multiple longest group, do better than take the first
                first_grp_sep_inds = list(sep_longestgrps.sgd.items())[0]
                best_PRc_ind = first_grp_sep_inds[1][0][len(first_grp_sep_inds[1][0]) // 2]
                best_PRc = graph_lbls_lcs_adjPRcs_grps[label][first_grp_sep_inds[0][0]][
                    first_grp_sep_inds[0][1]
                ][best_PRc_ind]
                assert best_PRc is not None
                lis[label] = Label_PR(
                    best_PRc.pos,
                    best_PRc.rot,  # pyright: ignore[reportArgumentType]
                )
                break

    # Initialize legend content for labels than cannot be inlined properly on their curve
    legend_labels = []

    for label in list(graph_lbls_lcs_adjPRcs_grps):
        if label not in list(lis):
            # Add label to the list of labels with no non overlapping placement
            # available. To be drawn in a standard legend
            legend_labels.append(label)

    return lis, legend_labels


# //
# //def solution_selection_algo1(linelikeLabels: list[str], graph_labels_PRcs: Labels_PRcs):
# //    #! Minimise label centers' convex hull area and distance of centers from their
# //    #! adjacent non overlaping solutions set boundaries
# //    # First identify labels with non overlapping solutions and for each ones a set of position candidates with distance to their adjacency subset boundaries
# //
# //    # PCWD = namedtuple("PCWD", ["xy", "d"])  # Position Candiate with its distance to its adjacent subset boundaries
# //    PCWD = NamedTuple(
# //        "PCWD", [("xy", tuple[float, float]), ("d", float)]
# //    )  # Position Candidate with its distance to its adjacent subset boundaries
# //    PCWDLD = dict[str, list[PCWD]]
# //    pcwdld = PCWDLD()
# //
# //    # Initialize legend content for labels than cannot be inlined properly on their curve
# //    legend_labels = []
# //
# //    #! MLS ALTERNATIVE
# //    #! UNFINISHED... move from graph_labels_PRcs to graph_labels_lcPRcs or wait for the
# //    #! introduction of a distance to other geometries to be introduced
# //    # For each PRc compute its distance to the edge of the minimum seperation adjacency
# //    # group it belongs to
# //    for label in linelikeLabels:
# //        pcwdld[label] = []
# //        cont_grps_dict = list[list[int]]([])
# //        print(f"{SEP_LEVELS=}")
# //        minsep_PRc_mask = [
# //            PRc is not None and PRc.sep is not None and PRc.sep == min(SEP_LEVELS)
# //            for PRc in graph_labels_PRcs[label]
# //        ]
# //        if sum(minsep_PRc_mask) == 0:
# //            # Add label to the list of labels with no non overlapping placement
# //            # available. To be drawn in a standard legend
# //            legend_labels.append(label)
# //            # Remove it from labels entries in Labels_PRcs structure #? Maybe useless
# //            del graph_labels_PRcs[label]
# //        else:
# //            # Indices of PRc candidates with a separation equal to buffering factor
# //            inds = np.nonzero(minsep_PRc_mask)[0]
# //            # Groups of continuous indices
# //            groups = np.split(inds, np.nonzero((np.diff(inds) > 1))[0] + 1)
# //            cont_grps_dict = [g.tolist() for g in groups]
# //
# //        for i, PRc in enumerate(graph_labels_PRcs[label]):
# //            if PRc is not None and PRc.sep is not None:
# //                # Find the iso sep group the PRc belongs to
# //                cont_grp = next(g for g in cont_grps_dict if i in g)
# //                # Compute the distance to the edges with the symplifying hypothesis that
# //                # PRc are evenly placed along the considered line chunk, which is not
# //                # the case
# //                dist = min(i - cont_grp[0], cont_grp[-1] - i)
# //                PRc.c_dte = dist
# //                graph_labels_PRcs[label][i] = PRc
# //
# //    class Algo1(ElementwiseProblem):
# //        def __init__(self):
# //            super().__init__(
# //                n_var=len(pcwdld),
# //                n_obj=2,
# //                n_ieq_constr=0,
# //                xl=[0] * len(pcwdld),
# //                xu=[len(pcwdl) - 1 for pcwdl in pcwdld.values()],
# //                vtype=int,
# //            )
# //
# //        def _evaluate(self, x, out, *args, **kwargs):
# //            pcwdl = [list(pcwdld.values())[i][x[i]] for i in range(len(pcwdld))]
# //            out["F"] = [
# //                ConvexHull([pcwd.xy for pcwd in pcwdl]).area,
# //                sum([pcwd.d for pcwd in pcwdl]),
# //            ]
# //
# //    problem = Algo1()
# //
# //    algorithm = NSGA2(
# //        pop_size=100,
# //        sampling=IntegerRandomSampling(),
# //        crossover=SBX(prob=1.0, eta=3, vtype=float, repair=RoundingRepair()),
# //        mutation=PM(prob=1.0, eta=3, vtype=float, repair=RoundingRepair()),
# //        eliminate_duplicates=True,
# //    )
# //
# //    res = minimize(
# //        problem,
# //        algorithm,
# //        # ("n_gen", 1000),
# //        seed=1,
# //    )
# //    # print(f"Best solution found: {res.X}")
# //    print(f"Number of best solution: {len(res.X)}")  # type: ignore
# //    # print(f"Function value: {res.F}")
# //    # print(f"Constraint violation: {res.CV}")
# //
# //    # for label in list(pcwdld):
# //    # res.X[1]
# //
