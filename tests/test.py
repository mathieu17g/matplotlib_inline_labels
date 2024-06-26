# %%
import matplotlib.pyplot as plt
import numpy as np
import pytest
from datetime import datetime
from matplotlib.dates import UTC, DateFormatter, DayLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import loglaplace, chi2

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inline_labels import add_inline_labels


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_multiple_subplots_with_gridspec():
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    with plt.style.context("fivethirtyeight"):
        fig = plt.figure(tight_layout=True, dpi=300, figsize=(20, 15))
        gsc = GridSpec(
            3, 1, figure=fig, height_ratios=[1, 0.5, 1.5]
        )  # Gridspec with column of n lines
        gsl = []  # Initialize line subgridspec list
        axs = []  # Initialize axe list on vertical dimension
        for i in range(3):
            # Add to subgridspec list, gridspec for i-th line
            gsl.append(
                GridSpecFromSubplotSpec(
                    1, 3, subplot_spec=gsc[i], width_ratios=[1, 0.5, 1.5]
                )
            )
            # Add 2nd horizontal dimension to the axe list
            axs.append([])
            for j in range(3):
                axs[i].append(fig.add_subplot(gsl[i][0, j]))
                for a in A:
                    axs[i][j].semilogx(
                        X,
                        chi2(5).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
                        label=f"{a=}",
                    )
                # Add axe title
                axs[i][j].set_title(f"Graph[{i+1}][{j+1}]")
                # Y-axis label on fist plot of each row only
                if j == 0:
                    axs[i][j].set_ylabel("Y=chi2(5).pdf(aX) with\na in [1, 2, 5, 10, 20]")
                # For second colum set y max to 0.2
                if j == 1:
                    axs[i][j].set_ylim(top=0.2)
                # For third colum set y max to 0.1
                if j == 2:
                    axs[i][j].set_ylim(top=0.1)

        for i in range(3):
            for j in range(3):
                add_inline_labels(axs[i][j], with_overall_progress=True)

    return fig


# _ = test_multiple_subplots_with_gridspec()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_linspace():
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_linspace()
# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_warning_for_nolabels():
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x))

    with pytest.warns(UserWarning, match="No line like object with label set"):
        add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_linspace()
# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_with_overall_and_perlabel_progress_bars():
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    add_inline_labels(plt.gca(), with_overall_progress=True, with_perlabel_progress=True)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_linspace()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_linspace_with_visualdebug():
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    #! savefig_kwargs={"bbox_inches": "tight"} is necessary to properly save visual debug image
    add_inline_labels(plt.gca(), debug=True, preprocessing_curv_filter_mode="precise")
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_linspace_with_visualdebug()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_precise_preprocessing_option_with_radius_lowering():
    X = np.linspace(0, 1, 1000)
    A = [1, 2, 5, 10, 20]

    fig, ax = plt.subplots()  # figsize=(96, 64))

    for a in A:
        Y = np.sin(a * X)
        # Shortening of last curve to cut it just after a high curvature point, enables to 
        # to hit the code handling the case where there is one position candidate for which
        # the circle centered on it and of radius label's bounding box half diagonal does 
        # not intersects the curve but the circle of radius label's bounding box half width
        # does. Therefore enabling to use the maximum radius between the two radii, 
        # intersecting the curve
        if a == 20:
            Y[90:-1] = np.nan
        ax.plot(X, Y, label=f"T{a}")

    fig_debug = add_inline_labels(
        ax,
        ppf=0.5,
        preprocessing_curv_filter_mode="precise",
        fontsize="large",
    )

    return fig_debug


# _ = test_precise_preprocessing_option()

# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_precise_preprocessing_option_without_radius_lowering():
    X = np.linspace(0, 1, 1000)
    A = [1, 2, 5, 10, 20]

    fig, ax = plt.subplots()  # figsize=(96, 64))

    for a in A:
        Y = np.sin(a * X)
        # Shortening of last curve to cut it just after a high curvature point, enables to 
        # to hit the code handling the case where there is one position candidate for which
        # the ring between the circle centered on it and of radius label's bounding box half
        # diagonal and the circle of radius label's bounding box half width does not
        # intersect the curve on one side
        if a == 20:
            Y[100:-1] = np.nan
        ax.plot(X, Y, label=f"T{a}")

    fig_debug = add_inline_labels(
        ax,
        ppf=0.5,
        preprocessing_curv_filter_mode="precise",
        fontsize="large",
    )

    return fig_debug


# _ = test_precise_preprocessing_option()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_ylogspace():
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label=r"$f(x)=\exp(%s x)$" % k)

    plt.yscale("log")
    add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_ylogspace()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_xlogspace():
    x = np.linspace(0, 10)
    K = [1, 2, 4]

    for k in K:
        plt.plot(10**x, k * x, label=r"$f(x)=%s x$" % k)

    plt.xscale("log")
    # NOTE: depending on roundoff, the upper limit may be
    # 1e11 or 1e10. See PR #155.
    plt.xlim(1e0, 1e11)
    add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_xlogspace()


# %%
# @pytest.mark.xfail(sys.platform.startswith('linux'), reason="pytest-mpl does not yield same figure between test and baseline generation on Linux")
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_xylogspace():
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    x = np.geomspace(0.1, 1e1)
    K = np.arange(-5, 5, 2)

    for k in K:
        plt.plot(x, np.power(x, k), label=rf"$f(x)=x^{{{k}}}$")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(0.1, 1e1)  # Needed to avoid approximation overshoot on x lims
    plt.ylim(1e-5, 1e5)  # Needed to avoid approximation overshoot on y lims

    add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


# _ = test_xylogspace()


# %%
@pytest.mark.skip(reason="Label rotation with x or y limits modification not supported yet")
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_rotation_correction():
    # Fix axes limits and plot line
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.plot((0, 1), (0, 2), label="rescaled")

    # Now label the line and THEN rescale the axes, to force label rotation
    add_inline_labels(plt.gca())
    ax.set_ylim(0, 2)

    return fig


# _ = test_rotation_correction()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_vertical():
    x = 0.5
    plt.axvline(x, label="axvline")
    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_vertical()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_labels_range():
    x = np.linspace(0, 1)

    plt.plot(x, np.sin(x), label=r"$\sin x$")
    plt.plot(x, np.cos(x), label=r"$\cos x$")

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_labels_range()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_dateaxis_naive():
    dates = np.array([datetime(2018, 11, 1), datetime(2018, 11, 2), datetime(2018, 11, 3)])

    plt.plot(dates, [0, 5, 3], label="apples")
    plt.plot(dates, [3, 6, 2], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_dateaxis_naive()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_dateaxis_advanced():
    dates = np.array(
        [
            datetime(2018, 11, 1, tzinfo=UTC),
            datetime(2018, 11, 2, tzinfo=UTC),
            datetime(2018, 11, 5, tzinfo=UTC),
            datetime(2018, 11, 3, tzinfo=UTC),
        ]
    )

    plt.plot(dates, [0, 5, 3, 0], label="apples")
    plt.plot(dates, [3, 6, 2, 1], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_dateaxis_advanced()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_polar():
    t = np.linspace(0, 2 * np.pi, num=128)
    plt.plot(np.cos(t), np.sin(t), label="$1/1$")
    plt.plot(np.cos(t), np.sin(2 * t), label="$1/2$")
    plt.plot(np.cos(3 * t), np.sin(t), label="$3/1$")

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_polar()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_non_uniform_and_negative_spacing():
    x = [1, -2, -3, 2, -4, -3]
    plt.plot(x, [1, 2, 3, 4, 2, 1], ".-", label="apples")
    plt.plot(x, [6, 5, 4, 2, 5, 5], "o-", label="banana")

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_non_uniform_and_negative_spacing()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_errorbar():
    x = np.linspace(0, 1, 20)

    y = x**0.5
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$\sqrt{x}\pm x$", capsize=3)[0]

    y = x**3
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$x^3\pm x$", capsize=3)[0]

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_errorbar()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_negative_spacing():
    x = np.linspace(1, -1)
    y = x**2
    plt.plot(x, y, label="Test")[0]
    # Should not throw an error
    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_negative_spacing()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_label_datetime_plot():
    plt.clf()
    # data from the chinook database of iTunes music sales
    x = np.array(
        [
            "2009-01-31T00:00:00.000000000",
            "2009-02-28T00:00:00.000000000",
            "2009-03-31T00:00:00.000000000",
            "2009-04-30T00:00:00.000000000",
            "2009-06-30T00:00:00.000000000",
            "2009-09-30T00:00:00.000000000",
            "2009-10-31T00:00:00.000000000",
            "2009-11-30T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )
    y = np.array([13.86, 14.85, 28.71, 42.57, 61.38, 76.23, 77.22, 81.18])

    plt.plot_date(x, y, "-", label="USA")[0]
    plt.xticks(rotation=45)

    # should not throw an error
    plt.tight_layout()
    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_label_datetime_plot()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_auto_layout():
    X = [[1, 2], [0, 1]]
    Y = [[0, 1], [0, 1]]

    lines = []
    for i, (x, y) in enumerate(zip(X, Y)):
        lines.extend(plt.plot(x, y, label=f"i={i}"))

    add_inline_labels(plt.gca())
    return plt.gcf()


# _ = test_auto_layout()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_errorbar_with_list():
    np.random.seed(1234)
    fig, ax = plt.subplots(figsize=[10, 2])
    samples = ["a", "b"]

    x = list(np.arange(-2, 2.1, 0.1))
    ys = [list(np.random.rand(len(x))), list(np.random.rand(len(x)))]

    lines = []
    for sample, y in zip(samples, ys):
        lines.append(ax.errorbar(x, y, yerr=0.1, label=sample, capsize=3)[0])

    add_inline_labels(plt.gca(), fontsize="large")
    return fig


# _ = test_errorbar_with_list()


# %%
@pytest.mark.skip(
    reason=(
        "For a Line 2D built with axhline, x data coordinates are in Axes coordinates."
        " Cannot figure how to identify it among the Line2D of an Axes"
    )
)
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_labeling_axhline():
    fig, ax = plt.subplots()
    ax.plot([10, 12, 13], [1, 2, 3], label="plot")
    ax.axhline(y=2, label="axhline")
    # print(f"{ax.get_lines()[1].get_data()=}") #! x data returned in Axes coordinates
    fig_for_debug = add_inline_labels(plt.gca(), debug=True)
    return fig_for_debug


# _ = test_labeling_axhline()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_multiple_labels_with_overall_progress():
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    for a in A:
        plt.plot(
            X,
            loglaplace(4).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
            label=f"Line {a}",
        )
    add_inline_labels(plt.gca(), fontsize="medium", with_overall_progress=True)
    return plt.gcf()


# _ = test_multiple_labels_with_overall_progress()


# %%
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_multiple_labels_with_perlabel_progress():
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    for a in A:
        plt.plot(
            X,
            loglaplace(4).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
            label=f"Line {a}",
        )
    add_inline_labels(plt.gca(), fontsize="medium", with_perlabel_progress=True)
    return plt.gcf()


# _ = test_multiple_labels_with_perlabel_progress()


@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_fig_correctly_drawn_before_finding_label_placement():
    # Two identical subplots, with a huge title on the second should yield the
    # same label positionning and angles which is an issue when not drawing
    # the figure before launching the placement algorithm: line geometries wrongly
    # placed.
    # Issue fixed in release 0.1.7 by using fig.draw instead of fig.draw_idle
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)

    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]

    for a in A:
        ax1.semilogx(
            X,
            chi2(5).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
            label=f"Line {a}",
        )
        ax2.semilogx(
            X,
            chi2(5).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
            label=f"Line {a}",
        )

    ax2.set_title("Title", fontsize=50)

    add_inline_labels(ax1)
    add_inline_labels(ax2)

    return fig


# _ = test_fig_correctly_drawn_before_finding_label_placement()


# %% Lemniscates
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_closed_curves_lemniscates():
    fig, ax = plt.subplots()

    A = [1, 2, 5, 10, 20]
    t = np.linspace(0, 2 * np.pi, num=1000)

    for a in A:
        X = np.log10(a) / 10 + np.log10(a) * np.cos(t) / (np.sin(t) ** 2 + 1)
        Y = np.log10(a) / 10 + np.log10(a) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1)
        ax.plot(X, Y, label=f"lem {a}")

    add_inline_labels(ax, fontsize="medium")

    return fig


# _ = test_closed_curves_lemniscates()


# %% Lemniscates with 1 gap
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_closed_curves_lemniscates_with_1_gap():
    fig, ax = plt.subplots()

    A = [1, 2, 5, 10, 20]
    t = np.linspace(0, 2 * np.pi, num=1000)

    for a in A:
        X = np.log10(a) / 10 + np.log10(a) * np.cos(t) / (np.sin(t) ** 2 + 1)
        Y = np.log10(a) / 10 + np.log10(a) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1)
        Y[50:110] = np.nan
        ax.plot(X, Y, label=f"lem {a}")

    fig_for_debug = add_inline_labels(ax, debug=True, fontsize="medium")

    return fig_for_debug


# _ = test_closed_curves_lemniscates()


# %% Lemniscates with 2 gaps
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_closed_curves_lemniscates_with_2_gaps():
    fig, ax = plt.subplots()

    A = [1, 2, 5, 10, 20]
    t = np.linspace(0, 2 * np.pi, num=1000)

    for a in A:
        X = np.log10(a) / 10 + np.log10(a) * np.cos(t) / (np.sin(t) ** 2 + 1)
        Y = np.log10(a) / 10 + np.log10(a) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1)
        Y[50:110] = np.nan
        Y[505:530] = np.nan
        ax.plot(X, Y, label=f"lem {a}")

    fig_for_debug = add_inline_labels(ax, debug=True, fontsize="medium")

    return fig_for_debug


# _ = test_closed_curves_lemniscates()


# %% Lemniscates with 3 gaps and an isolated point
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_closed_curves_lemniscates_with_3_gaps_and_an_isolated_point():
    fig, ax = plt.subplots()

    A = [1, 2, 5, 10, 20]
    t = np.linspace(0, 2 * np.pi, num=1000)

    for a in A:
        X = np.log10(a) / 10 + np.log10(a) * np.cos(t) / (np.sin(t) ** 2 + 1)
        Y = np.log10(a) / 10 + np.log10(a) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1)
        Y[50:70] = np.nan
        Y[71:110] = np.nan
        Y[505:530] = np.nan
        ax.plot(X, Y, label=f"lem {a}")
        if a != 1:
            ax.scatter(
                X[70],
                Y[70],
                s=100,
                label="isolated point",
                facecolor="none",
                edgecolor="black",
            )
            annotation_text = "isolated points\non the curve" if a == 2 else ""
            ax.annotate(
                annotation_text,
                xy=(X[70], Y[70]),
                xycoords="data",
                xytext=(0.55, 0.9),
                textcoords="axes fraction",
                arrowprops=dict(
                    facecolor="black", width=1, headwidth=5, headlength=10, shrink=0.1
                ),
                horizontalalignment="right",
                verticalalignment="bottom",
            )

    add_inline_labels(ax, fontsize="medium")

    return fig


# _ = test_closed_curves_lemniscates()


# %% Almost touching circles
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_closed_curves_almost_touching_circles():
    fig, ax = plt.subplots()

    A = [1, 2, 5, 10, 20]
    t = np.linspace(0, 2 * np.pi, num=1000)

    for a in A:
        X = np.log10(a) / 1.5 + np.log10(a) * np.cos(t)
        Y = np.log10(a) / 1.5 + np.log10(a) * np.sin(t)
        ax.plot(X, Y, label=f"R=log({a})")

    ax.set_aspect("equal", adjustable="box")

    add_inline_labels(ax, fontsize="small")

    return fig


# _ = test_closed_curves_almost_touching_circles()


# %% Well separated circles
@pytest.mark.mpl_image_compare(style="default", savefig_kwargs={"bbox_inches": "tight"})
def test_closed_curves_well_separeted_circles():
    fig, ax = plt.subplots()

    A = [1, 2, 5, 10, 20]
    t = np.linspace(0, 2 * np.pi, num=1000)

    for a in A:
        X = np.log10(a) / 2 + np.log10(a) * np.cos(t)
        Y = np.log10(a) / 2 + np.log10(a) * np.sin(t)
        ax.plot(X, Y, label=f"R=log({a})")

    ax.set_aspect(1)
    # ax.set_aspect("equal", adjustable="box")

    add_inline_labels(ax, fontsize="small")

    return fig


# _ = test_closed_curves_well_separeted_circles()
