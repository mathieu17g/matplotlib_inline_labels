# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, loglaplace
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from inline_labels import add_inline_labels

X = np.linspace(0, 1, 1000)
A = [1, 2, 5, 10, 20]
funcs = [np.arctan, np.sin, loglaplace(4).pdf, chi2(5).pdf] # pyright: ignore[reportAttributeAccessIssue]

# %% Visual debug example
fig, ax = plt.subplots()

for a in A:
    Y = np.sin(a * X)
    Y[100+5*a:100+5*a*3] = np.nan
    ax.plot(X, Y, label=f"Line {a}")

fig_debug = add_inline_labels(ax, ppf=1.5, debug=True, fontsize="small")

fig_debug.savefig("example_debug.png", bbox_inches="tight") # pyright: ignore[reportOptionalMemberAccess]


# %% Examples
fig, axes = plt.subplots(ncols=2, nrows=3, constrained_layout=True, figsize=(8, 8))
axes = axes.flatten()

for a in A:
    axes[0].plot(X, np.arctan(a * X), label=f"Line {a}")
add_inline_labels(axes[0], with_overall_progress=True, fontsize="large")

for a in A:
    axes[1].plot(X, np.sin(a * X), label=f"Line {a}")
    axes[2].plot(X, loglaplace(4).pdf(a * X), label=f"Line {a}") # pyright: ignore[reportAttributeAccessIssue]
    axes[3].plot(X, chi2(5).pdf(a * X), label=f"Line {a}") # pyright: ignore[reportAttributeAccessIssue]
    axes[4].semilogx(X, np.arctan(5 * a * X), label=str(a)) # pyright: ignore[reportAttributeAccessIssue]
    axes[5].semilogx(X, chi2(5).pdf(a * X), label=str(a)) # pyright: ignore[reportAttributeAccessIssue]
add_inline_labels(axes[1], with_overall_progress=True, fontsize="small")
add_inline_labels(axes[2], with_overall_progress=True, fontsize="x-small")
add_inline_labels(axes[3], with_overall_progress=True, fontsize="medium")
add_inline_labels(axes[4], with_overall_progress=True, fontsize="small")
add_inline_labels(axes[5], with_overall_progress=True, fontsize="small")

fig.savefig("example.png")


# %%
