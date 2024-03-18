# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, loglaplace
from cProfile import Profile
from pstats import SortKey, Stats
from datetime import date
from dateutil.relativedelta import relativedelta
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inline_labels import add_inline_labels

X = np.linspace(0, 1, 1000)
A = [1, 2, 5, 10, 20]
funcs = [
    np.arctan,
    np.sin,
    loglaplace(4).pdf,  # pyright: ignore[reportAttributeAccessIssue]
    chi2(5).pdf,  # pyright: ignore[reportAttributeAccessIssue]
]

# %% Visual debug example 1
fig, ax = plt.subplots()  # figsize=(96, 64))

for a in A:
    Y = np.sin(a * X)
    # Y[10*a:10*a*3] = np.nan
    ax.plot(X, Y, label=f"Line {a}")

fig_debug = add_inline_labels(
    ax, ppf=1.5, with_perlabel_progress=True, debug=True, fontsize="small"
)

# fig_debug.savefig("example_debug.png", bbox_inches="tight")

# %% Visual debug example 2
fig, ax = plt.subplots()

for a in A:
    ax.plot(
        X,
        loglaplace(4).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
        label=f"Line {a}",
    )

fig_debug = add_inline_labels(ax, ppf=1.5, debug=True, fontsize="medium")

# %% Visual debug with x as dates with nan values at line extremities
fig, ax = plt.subplots()
RANGE = 120
X = np.array([date(2015 + i // 12, 1 + i % 12, 1) for i in range(RANGE)])
RD = [relativedelta(X[0], x) for x in X]
A = [1, 2, 5, 10, 20]
for a in A:
    Y = np.sin(a * np.array([rd.months + 12 * rd.years for rd in RD]) / RANGE)
    Y[RANGE - RANGE // 6 + 5 * a : RANGE - RANGE // 6 + 5 * a * 3] = np.nan
    ax.plot(X, Y, label=f"TS{a}")
fig_debug = add_inline_labels(ax, ppf=1, debug=True, fontsize="medium")

# %% Visual debug example 3 (lemniscates)
fig, ax = plt.subplots()

t = np.linspace(0, 2 * np.pi, num=1000)

for a in A:
    X = np.log10(a) / 10 + np.log10(a) * np.cos(t) / (np.sin(t) ** 2 + 1)
    Y = np.log10(a) / 10 + np.log10(a) * np.cos(t) * np.sin(t) / (np.sin(t) ** 2 + 1)
    ax.plot(X, Y, label=f"lem {a}")


fig_debug = add_inline_labels(ax, ppf=1, debug=True, fontsize="medium")


# %% Visual debug example 4 (almost touching circles)
fig, ax = plt.subplots()

t = np.linspace(0, 2 * np.pi, num=1000)

for a in A:
    X = np.log10(a) / 1.5 + np.log10(a) * np.cos(t)
    Y = np.log10(a) / 1.5 + np.log10(a) * np.sin(t)
    ax.plot(X, Y, label=f"R=log({a})")

ax.set_aspect("equal", adjustable="box")

fig_debug = add_inline_labels(ax, ppf=1, debug=True, fontsize="small")

# %% Visual debug example 5 (well separated circles)
fig, ax = plt.subplots()

t = np.linspace(0, 2 * np.pi, num=1000)

for a in A:
    X = np.log10(a) / 2 + np.log10(a) * np.cos(t)
    Y = np.log10(a) / 2 + np.log10(a) * np.sin(t)
    ax.plot(X, Y, label=f"R=log({a})")

ax.set_aspect(1)
# ax.set_aspect("equal", adjustable="box")

fig_debug = add_inline_labels(ax, ppf=1, debug=True, fontsize="small")

# %% Examples
fig, axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(8, 8))
axes = axes.flatten()

for a in A:
    axes[0].plot(X, np.arctan(a * X), label=f"Line {a}")
add_inline_labels(axes[0], with_overall_progress=True, fontsize="large")

for a in A:
    axes[1].plot(X, np.sin(a * X), label=f"Line {a}")
    axes[2].plot(
        X,
        loglaplace(4).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
        label=f"Line {a}",
    )
    axes[3].plot(
        X,
        chi2(5).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
        label=f"Line {a}",
    )
    axes[4].semilogx(X, np.arctan(5 * a * X), label=str(a))
    axes[5].semilogx(
        X, chi2(5).pdf(a * X), label=str(a)  # pyright: ignore[reportAttributeAccessIssue]
    )
add_inline_labels(axes[1], with_overall_progress=True, fontsize="small")
add_inline_labels(axes[2], with_overall_progress=True, fontsize="x-small")
add_inline_labels(axes[3], with_overall_progress=True, fontsize="medium")
add_inline_labels(axes[4], with_overall_progress=True, fontsize="small")
add_inline_labels(axes[5], with_overall_progress=True, fontsize="small")


# %% Profiling
fig, ax = plt.subplots()

for a in A:
    ax.semilogx(
        X,
        chi2(5).pdf(a * X),  # pyright: ignore[reportAttributeAccessIssue]
        label=f"Line {a}",
    )
ax.set_ylim(top=0.12)


with Profile() as profile:
    fig_debug = add_inline_labels(ax)
    (
        Stats(profile)
        .strip_dirs()
        .sort_stats(SortKey.TIME, SortKey.CUMULATIVE)
        .print_stats()
        .dump_stats("vdbg.prof")
    )

# %%
