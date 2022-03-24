# Free support Wasserstein-2 barycenters
Implementations of the algorithms from the paper [Simple Approximative Algorithms for Free-Support Wasserstein Barycenters (von Lindheim, 2022)](https://arxiv.org/abs/2203.05267). For algorithms for the related multi-marginal optimal transport problem, see [this repository](https://github.com/jvlindheim/mot).

Using the `emd` OT solver from the [Python Optimal Transport (POT)](https://pythonot.github.io/index.html) package, which is a wrapper of [this network simplex solver](https://github.com/nbonneel/network_simplex), which, in turn, is based on an implementation in the [LEMON](http://lemon.cs.elte.hu/pub/doc/latest-svn/index.html) C++ library.

## Installation
1. Download the code or clone the Github repository with
```
git clone https://github.com/jvlindheim/free-support-barycenters.git
```
2. For the code in `bary.py`, there is the following dependencies: `numpy`, `matplotlib.pyplot`, the `cdist` function from `scipy.spatial.distance` and the `emd` function from the [POT](https://pythonot.github.io/index.html) library. You can install them e.g. using pip via
```
pip install --user numpy scipy matplotlib POT
```
If you want to run the demo notebook, you will also need to have [Jupyter Notebook or JupyterLab](https://jupyter.org/install) installed.
