import jax.numpy as jnp
import jax.random as jr
from scipy.optimize import linear_sum_assignment

from ..perm_spec import PermutationSpec
from ..utils import get_permuted_param, rngmix

# From https://github.com/samuela/git-re-basin/blob/main/src/weight_matching.py
def weight_matching(rng, ps: PermutationSpec, params_a, params_b, max_iter=100, init_perm=None, masks={}):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}

    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()} if init_perm is None else init_perm
    perm_names = list(perm.keys())
    
    for iteration in range(max_iter):
        progress = False
        rng = rngmix(rng, iteration)
        for p_ix in jr.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                #print("A:", A)
                A += w_a @ w_b.T
            
            if p in masks:
                #print("Here")
                A += masks[p]
            
            #print("A:", A)
            
            ri, ci = linear_sum_assignment(A, maximize=True)
            assert (ri == jnp.arange(len(ri))).all()

            oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
            newL = jnp.vdot(A, jnp.eye(n)[ci, :])
            print(f"{iteration}/{p}: {newL - oldL}, {newL}, {oldL}")
            progress = progress or newL > oldL + 1e-12

            perm[p] = jnp.array(ci)

        if not progress:
            break

    return perm
