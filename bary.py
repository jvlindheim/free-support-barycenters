"""
Approximative algorithms for free-support Wasserstein-2 barycenters of discrete probability distributions.

Johannes von Lindheim, 2022
https://github.com/jvlindheim/free-support-barycenters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from ot import emd

"""
Reference algorithm.
"""

def ref_bary(supports, masses, weights=None, ref_index=0, precision=7):
    """
    Computation of an approximate Wasserstein-2 barycenter using the reference algorithm.
    Convenience method combining execution of the reference measure alignment and construction
    of the correcponding barycenter.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    ref_index=0: Which of the given measures is treated as the reference measure (first measure is the default)
    precision=7: Rounded to which decimal place the weights need to sum up to one.
        
    Returns
    -------
    bary_supp: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    supports, masses, n, nis, d = prepare_data(supports, masses)
    
    # set weights uniformly if None is given
    if weights is None:
        n = len(masses)
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    assert ref_index in np.arange(n), "ref_index needs to be between 0 and n-1, where n is the number of given measures"
    
    alignment, _ = ref_alignment(supports, masses, ref_index)
    return ref_supports_from_alignment(alignment, weights, precision=7), masses[ref_index]

def ref_alignment(supports, masses, ref_index=0):
    '''
    Computes approximations of the given measures as barycentric projections from transport plans
    from a reference measure to the given input measures.
    These can be further used to compute approximate barycenters for arbitrary sets of weights.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    ref_index=0: Which of the given measures is treated as the reference measure (first measure is the default)
    
    Returns
    -------
    alignment: (n, n_ref, d)-shaped array, where n_ref is the number of support points in the reference measure.
    bary_masses: Barycenter masses corresponding to the support positions computed by 'ref_supports_from_alignment'. This is precisely the masses corresponding to the reference measure, which might have been reduced to less entries by 'prepare_data' for small or zero masses.
    '''
    
    supports, masses, n, nis, d = prepare_data(supports, masses)
    assert ref_index in np.arange(n), "ref_index needs to be between 0 and n-1, where n is the number of given measures"
    
    # compute transport plans from reference measure to other measures
    ref_mass = masses[ref_index]
    ref_supp = supports[ref_index]
    pis = [emd(ref_mass, mui_mass, cdist(ref_supp, mui_pos, metric='sqeuclidean'))
           for mui_pos, mui_mass in zip(supports, masses)]
    return np.stack([(pi/ref_mass[:, None]).dot(pos) for pi, pos in zip(pis, supports)]), masses[ref_index]

def ref_supports_from_alignment(alignment, weights=None, precision=7):
    '''
    From an alignment of the original measures with respect to a reference measure as computed by
    the function 'ref_alignment', compute barycenter support with respect to a given set of weights.
    The weights of the support points are simply given by the weights of the chosen reference
    measure and are not returned.
    
    Parameters
    ----------
    alignment: result of the function 'ref_alignment'.
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    precision=7: Rounded to which decimal place the weights need to sum up to one.
    
    Returns
    -------
    bary_supp: Barycenter support positions.
    '''
    
    if weights is None:
        n = alignment.shape[0]
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    return (weights[:, None, None]*alignment).sum(axis=0)

"""
Pairwise algorithm.
"""

def pairwise_bary(supports, masses, weights=None, compute_err_bound=False, precision=7):
    """
    Computation of an approximate Wasserstein-2 barycenter using the pairwise algorithm.
    Convenience method combining computation of all pairwise transport plans and construction
    of the correcponding barycenter.
    If barycenters for multiple sets of weights should be computed, for speed rather execute
    'pairwise_kernels' (bottleneck) only once and construct multiple barycenters using
    'pairwise_bary_from_kernels' (fast).
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    precision=7: Rounded to which decimal place the weights need to sum up to one.
        
    Returns
    -------
    bary_supp: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    """
    
    supports, masses, n, nis, d = prepare_data(supports, masses)
    # set weights uniformly if None is given
    if weights is None:
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"

    return pairwise_bary_from_kernels(supports, masses, pairwise_kernels(supports, masses),
                                      weights, compute_err_bound, precision)

def pairwise_kernels(supports, masses):
    '''
    Computes the row-normalized matrix of all pairwise optimal transports between the input measues.
    Only computes transports once for each (unordered) distinct of measures and uses the transpose
    for the transpose pair.
    Further use the result in 'pairwise_bary_from_kernels'.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    
    Returns
    -------
    kernels: Barycenter support positions.
    '''

    supports, masses, n, nis, d = prepare_data(supports, masses)
    total_supp = np.concatenate(supports, axis=0)
    total_masses = np.concatenate(masses)
    n_supp = nis.sum()
    pairwise_pis = np.zeros((n_supp, n_supp))
    nis_cum = np.concatenate([[0], np.cumsum(nis)])
    
    # write all pairwise wasserstein-2 plans to a big matrix
    for i in range(n):
        for j in range(i+1, n):
            pi_ij = emd(masses[i], masses[j], cdist(supports[i], supports[j], metric='sqeuclidean'))
            pairwise_pis[nis_cum[i]:nis_cum[i+1], nis_cum[j]:nis_cum[j+1]] = pi_ij
    pairwise_pis += pairwise_pis.T
    pairwise_pis += np.diagflat(total_masses)
    
    return pairwise_pis/total_masses[:, None]

def pairwise_bary_from_kernels(supports, masses, kernels, weights=None, compute_err_bound=False, precision=7):
    '''
    From the precomputed result from 'pairwise_kernels', 
    compute a barycenter with respect to a given set of weights.
    
    Parameters
    ----------
    supports: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    kernels: matrix of all pairwise precomputed transport kernels using 'pairwise_kernels'.
    weights=None: array of same length as number of measures, needs to sum to one.
        These are the weights in the barycenter problem, typically denoted by lambda_i.
        If None is given, use uniform weights.
    compute_err_bound=False: If set to True, additionally to the barycenter, a bound for the relative error is returned.
    precision=7: Rounded to which decimal place the weights need to sum up to one.
    
    Returns
    -------
    bary_supp: Barycenter support positions.
    bary_masses: Barycenter masses corresponding to the support positions.
    error_bound: only returned if compute_err_bound is set to True.
        This is an upper bound on the relative error <= 1, where the relative is defined as
        Psi(approx. bary)/Psi(opt. bary) - 1 with Psi being the optimization functional of the barycenter problem.
    '''
    
    supports, masses, n, nis, d = prepare_data(supports, masses)
    
    # set weights uniformly if None is given
    if weights is None:
        n = len(masses)
        weights = np.ones((n,)) / n
    assert np.round(weights.sum(), decimals=precision) == 1, "weights need to sum to one"
    
    # compute barycenter from pairwise transport kernels
    bary_masses = np.concatenate([w*mass for w, mass in zip(weights, masses)])
    bary_supp = kernels.dot(np.concatenate([w*supp for w, supp in zip(weights, supports)], axis=0))
    
    if compute_err_bound:
        total_support = np.concatenate(supports)
        nis_cum = np.concatenate([[0], np.cumsum(nis)])
        # compute error bound
        enum = (bary_masses*(((total_support-bary_supp)**2).sum(axis=1))).sum() # weighted dists from bary to input measures
        denom = 2-0.5*sum([w*(np.repeat(weights, nis)*cdist(supp, total_support, 'sqeuclidean')*kernels[nis_cum[i]:nis_cum[i+1]]*mass[:, None]).sum()
                    for i, (w, supp, mass) in enumerate(zip(weights, supports, masses))]) # pairwise W2-dists
        return bary_supp, bary_masses, enum/denom
    
    return bary_supp, bary_masses

"""
Helper functions.
"""

def prepare_data(posns, masses, min_mass=1e-10):
    '''
    Given a support positions and masses array, determine (and make security checks for) the number of measures, number
    of support points array and dimension. Also modify posns to array, if given only one array for all measures.

    Parameters
    ----------
    posns: Measure support positions list/array of length n. Can be given just a 2d-array, if the positions are always the same.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    min_mass=1e-10: Every point with a mass less than this parameter is discarded.
    min_mass=1e-10: Every point with a mass less than this parameter is discarded.
        
    Returns
    -------
    posns: Measure support positions list of length n.
    masses: Masses/weights of the given measure support points.
    n: Number of determined measures.
    nis: Number of determined support points for each measure.
    d: Dimension of the support points.    
    '''    
    # if given a list, we assume that we are given multiple measures and the length of the list is
    # the number of measures
    if isinstance(posns, list):
        n = len(posns)
        assert len(masses) == n, "masses needs have same length as pos (equal number of measures)"
    # if given a 2d-array and a list of of mass arrays, we assume that we are given a number of measures,
    # which are all supported on the same posns-array, so the number of measures n is len(masses)
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and (isinstance(masses, list) \
                                                                or (isinstance(masses, np.ndarray) and masses.ndim == 2)):
        n = len(masses)
        posns = [posns]*n
    # if given a 3d-array and a 2d array of masses, we assume that we are given a number of measures that
    # all have the same number of points
    elif isinstance(posns, np.ndarray) and posns.ndim == 3:
        n = posns.shape[0]
        assert masses.shape[0] == n, "masses needs have same length as pos (equal number of measures)"
        assert posns.shape[1] == masses.shape[1], "number of points and number of mass entries need to be the same"

    # if given a 2d-array and a 1d-array of masses, assume that we are given only one measure
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and isinstance(masses, np.ndarray) and masses.ndim == 1:
        n = 1
        assert len(posns) == len(masses), "if only one measure is given, length of posns and masses need to match"
        posns = [posns]
        masses = masses[None, :]
    else:
        raise ValueError("cannot see what the number of measures is for given parameters 'posns' and 'masses'")
    assert n >= 1, "at least one measure needs to be given"
    assert all([pos.ndim == 2 for pos in posns]), "position arrays need to be two-dimensional"
    posns = [pos[mass > min_mass] for (pos, mass) in zip(posns, masses)] # throw out points with mass <= min_mass
    masses = [mass[mass > min_mass] for mass in masses]
    nis = np.array([pos.shape[0] for pos in posns]) # number of support points for all measures
    d = posns[0].shape[1]
    return [pos for pos in posns], [mass for mass in masses], n, nis, d

def scatter_distr(posns, masses, n_plots_per_row=2, scale=4, invert=False, disk_size=6000/5, xmarkers=False, color='gray',
           alpha=0.5, xmarker_posns=None, axis_off=False, margin_fac=0.2, dpi=300,
           subtitles='', figax=None, savepath=None):
    '''
    Scatter plot function for either one or multiple discrete probability distributions.

    Parameters
    ----------
    posns: Measure support positions list/array of length 1 or n.
    masses: Masses/weights of the given measure support points (need to sum to one for each measure)
    n_plots_per_row=2: In case multiple measures are given, a figure with this number of
        subplots per column is generated.
    scale=4: Size of the figure is proportional to this parameter.
    invert=False: Whether to invert the y-axis.
    disk_size=6000/5: Size of the plotted disks per support point is proportional
        to their weight and this parameter.
    xmarkers=False: Whether to plot an x in the center of each disk (support point).
    color='gray': Color of each disk (support point).
    alpha=0.5: Transparency of each disk (support point).
    xmarker_posns=None: An additional set of x-markers can be plotted if this
        parameter is given an array of 2d positions.
    axis_off=False: Whether to turn off the coordinate system of the subplots.
    margin_fac=0.2: How much margin to leave around the minimum and maximum x/y-values
        of the support points.
    dpi=300: Resolution of figure (important for export).
    subtitles='': Array of subtitles to each subplot.
    figax=None: Tuple of the form (fig, ax, k, l). This can be used, if this function
        is only supposed to plot in the given axis array 'ax' at index k, l that
        has already been constructed.
        If None is given, a new fig and axis array are created.
    savepath=None: Saves figure to this given path.
    '''
    posns, masses, n, nis, d = prepare_data(posns, masses)
    n_plots = n
    if figax is None:
        n_rows = np.ceil(n_plots / n_plots_per_row).astype(int)
        n_cols = min(n_plots_per_row, n_plots)
        figsize = (scale*n_cols, scale*n_rows)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    else:
        fig, ax, k, l = figax

    if isinstance(subtitles, list) and len(subtitles) == 1:
        subtitles = [subtitles[0]]*n_plots
    elif isinstance(subtitles, str):
        subtitles = [subtitles]*n_plots
    else:
        assert len(subtitles) == n_plots, "length of subtitles array needs to be equal to number of plots"

    xmin, xmax, ymin, ymax = min([pos[:, 0].min() for pos in posns]), max([pos[:, 0].max() for pos in posns]), min([pos[:, 1].min() for pos in posns]), max([pos[:, 1].max() for pos in posns])

    if figax is None:
        row_inds, col_inds = range(n_rows), range(n_cols)
    else:
        row_inds, col_inds = [k], [l]
    for i in row_inds:
        for j in col_inds:
            idx = i*n_plots_per_row + j if figax is None else 0
            if idx >= n_plots or axis_off:
                ax[i, j].axis('off')
                if idx >= n_plots:
                    continue
            pos = posns[idx]
            mass = masses[idx]

            # set plot dimensions
            xmargin = margin_fac*(xmax-xmin)
            ymargin = margin_fac*(ymax-ymin)
            ax[i, j].set_xlim([xmin-xmargin, xmax+xmargin])
            ax[i, j].set_ylim([ymin-ymargin, ymax+ymargin])
            ax[i, j].set_aspect('equal')
            ax[i, j].set_title(subtitles[idx])

            # plot
            if xmarkers:
                ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='x', c='red')
            if xmarker_posns is not None:
                ax[i, j].scatter(xmarker_posns[:, 0], xmarker_posns[:, 1], marker='x', c='red')
            ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='o', s=mass*disk_size*scale, c=color, alpha=alpha)
            if invert:
                ax[i, j].set_ylim(ax[i, j].get_ylim()[::-1])

    if savepath is not None:
        plt.savefig(savepath, dpi=dpi, pad_inches=0, bbox_inches='tight')
