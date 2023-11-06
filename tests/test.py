# %%
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.testing import setup
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inline_labels import add_inline_labels
from datetime import datetime
from matplotlib.dates import UTC, DateFormatter, DayLocator
from scipy.stats import loglaplace
from warnings import catch_warnings


# %%
@pytest.fixture()
def setup_mpl():
    setup()
    plt.clf()


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_linspace(setup_mpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


_ = test_linspace(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_linspace_with_visualdebug(setup_mpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

    #! savefig_kwargs={"bbox_inches": "tight"} is necessary to properly save visual debug image
    add_inline_labels(plt.gca(), debug=True, fig_for_debug=plt.gcf())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


_ = test_linspace_with_visualdebug(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_ylogspace(setup_mpl):
    x = np.linspace(0, 1)
    K = [1, 2, 4]

    for k in K:
        plt.plot(x, np.exp(k * x), label=r"$f(x)=\exp(%s x)$" % k)

    plt.yscale("log")
    add_inline_labels(plt.gca())
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    return plt.gcf()


_ = test_ylogspace(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_xlogspace(setup_mpl):
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


_ = test_xlogspace(setup_mpl)


# %%
# @pytest.mark.xfail(sys.platform.startswith('linux'), reason="pytest-mpl does not yield same figure between test and baseline generation on Linux")
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_xylogspace(setup_mpl):
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


_ = test_xylogspace(setup_mpl)


# %%
@pytest.mark.skip(reason="Label rotation with x or y limits modification not supported yet")
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_rotation_correction(setup_mpl):
    # Fix axes limits and plot line
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    lines = plt.plot((0, 1), (0, 2), label="rescaled")

    # Now label the line and THEN rescale the axes, to force label rotation
    add_inline_labels(plt.gca())
    ax.set_ylim(0, 2)

    return fig


_ = test_rotation_correction(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_vertical(setup_mpl):
    x = 0.5
    plt.axvline(x, label="axvline")
    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_vertical(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_labels_range(setup_mpl):
    x = np.linspace(0, 1)

    plt.plot(x, np.sin(x), label=r"$\sin x$")
    plt.plot(x, np.cos(x), label=r"$\cos x$")

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_labels_range(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_dateaxis_naive(setup_mpl):
    dates = [datetime(2018, 11, 1), datetime(2018, 11, 2), datetime(2018, 11, 3)]

    plt.plot(dates, [0, 5, 3], label="apples")
    plt.plot(dates, [3, 6, 2], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_dateaxis_naive(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_dateaxis_advanced(setup_mpl):
    dates = [
        datetime(2018, 11, 1, tzinfo=UTC),
        datetime(2018, 11, 2, tzinfo=UTC),
        datetime(2018, 11, 5, tzinfo=UTC),
        datetime(2018, 11, 3, tzinfo=UTC),
    ]

    plt.plot(dates, [0, 5, 3, 0], label="apples")
    plt.plot(dates, [3, 6, 2, 1], label="banana")
    ax = plt.gca()
    ax.xaxis.set_major_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_dateaxis_advanced(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_polar(setup_mpl):
    t = np.linspace(0, 2 * np.pi, num=128)
    plt.plot(np.cos(t), np.sin(t), label="$1/1$")
    plt.plot(np.cos(t), np.sin(2 * t), label="$1/2$")
    plt.plot(np.cos(3 * t), np.sin(t), label="$3/1$")
    ax = plt.gca()

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_polar(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_non_uniform_and_negative_spacing(setup_mpl):
    x = [1, -2, -3, 2, -4, -3]
    plt.plot(x, [1, 2, 3, 4, 2, 1], ".-", label="apples")
    plt.plot(x, [6, 5, 4, 2, 5, 5], "o-", label="banana")
    ax = plt.gca()

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_non_uniform_and_negative_spacing(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_errorbar(setup_mpl):
    x = np.linspace(0, 1, 20)

    y = x**0.5
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$\sqrt{x}\pm x$", capsize=3)[0]

    y = x**3
    dy = x
    plt.errorbar(x, y, yerr=dy, label=r"$x^3\pm x$", capsize=3)[0]

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_errorbar(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_negative_spacing(setup_mpl):
    x = np.linspace(1, -1)
    y = x**2
    plt.plot(x, y, label="Test")[0]
    # Should not throw an error
    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_negative_spacing(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_label_datetime_plot(setup_mpl):
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


_ = test_label_datetime_plot(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_auto_layout(setup_mpl):
    X = [[1, 2], [0, 1]]
    Y = [[0, 1], [0, 1]]

    lines = []
    for i, (x, y) in enumerate(zip(X, Y)):
        lines.extend(plt.plot(x, y, label=f"i={i}"))

    add_inline_labels(plt.gca())
    return plt.gcf()


_ = test_auto_layout(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_errorbar_with_list(setup_mpl):
    np.random.seed(1234)
    fig, ax = plt.subplots(figsize=[10, 2])
    samples = ["a", "b"]
    pos = [-1, 1]

    x = list(np.arange(-2, 2.1, 0.1))
    ys = [list(np.random.rand(len(x))), list(np.random.rand(len(x)))]

    lines = []
    for sample, y in zip(samples, ys):
        lines.append(ax.errorbar(x, y, yerr=0.1, label=sample, capsize=3)[0])

    add_inline_labels(plt.gca(), fontsize="large")
    return fig


_ = test_errorbar_with_list(setup_mpl)


# %%
@pytest.mark.skip(
    reason="For a Line 2D built with axhline, x data coordinates are in Axes coordinates. Cannot figure how to identify it among the Line2D of an Axes"
)
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_labeling_axhline(setup_mpl):
    fig, ax = plt.subplots()
    ax.plot([10, 12, 13], [1, 2, 3], label="plot")
    ax.axhline(y=2, label="axhline")
    # print(f"{ax.get_lines()[1].get_data()=}") #! x data returned in Axes coordinates
    add_inline_labels(plt.gca(), debug=True, fig_for_debug=plt.gcf())
    return fig


_ = test_labeling_axhline(setup_mpl)


# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_unplaced_labels_without_warning(setup_mpl):
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    for a in A:
        plt.plot(X, loglaplace(4).pdf(a * X), label=f"Line {a}")
    with catch_warnings():
        add_inline_labels(plt.gca(), fontsize="x-large")
    return plt.gcf()


_ = test_unplaced_labels_without_warning(setup_mpl)
# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_unplaced_labels_with_warning(setup_mpl):
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    for a in A:
        plt.plot(X, loglaplace(4).pdf(a * X), label=f"Line {a}")
    with pytest.warns(UserWarning):
        add_inline_labels(plt.gca(), fontsize="x-large", nowarn=False)
    return plt.gcf()


_ = test_unplaced_labels_with_warning(setup_mpl)
# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_multiple_labels_with_overall_progress(setup_mpl):
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    for a in A:
        plt.plot(X, loglaplace(4).pdf(a * X), label=f"Line {a}")
    add_inline_labels(plt.gca(), fontsize="medium", with_overall_progress=True)
    return plt.gcf()


_ = test_multiple_labels_with_overall_progress(setup_mpl)
# %%
@pytest.mark.mpl_image_compare(savefig_kwargs={"bbox_inches": "tight"})
def test_multiple_labels_with_perlabel_progress(setup_mpl):
    #! Necessary to specify a figsize on linux, otherwise baseline generated image different from test output image
    plt.subplots(figsize=(6.4, 4.8))
    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    for a in A:
        plt.plot(X, loglaplace(4).pdf(a * X), label=f"Line {a}")
    add_inline_labels(plt.gca(), fontsize="medium", with_perlabel_progress=True)
    return plt.gcf()


_ = test_multiple_labels_with_perlabel_progress(setup_mpl)
# %%
