import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import chi2, loglaplace
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from inline_labels import add_inline_labels


X = np.linspace(0, 1, 1000)
A = [1, 2, 5, 10, 20]
funcs = [
    np.arctan,
    np.sin,
    loglaplace(4).pdf,  # pyright: ignore[reportAttributeAccessIssue]
    chi2(5).pdf,  # pyright: ignore[reportAttributeAccessIssue]
]

# plt.tight_layout()
fig, ax = plt.subplots()
x = np.linspace(0, 1)
K = [1, 2, 4]

for k in K:
    plt.plot(x, np.sin(k * x), label=rf"$f(x)=\sin({k} x)$")

fig_debug = add_inline_labels(plt.gca(), ppf=0.5, debug=True)

# fig_debug.savefig("test.png", bbox_inches="tight")
plt.show()
