import logging

import jax.numpy as jnp
import jax.random as jr
from scipy.optimize import linear_sum_assignment

from ..perm_spec import PermutationSpec
from ..utils import tile_windows, nest_shuffles, get_permuted_param, rngmix

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_nested_perm(assign_mat_kq, assign_mat_v, num_heads, hidden_dim):
    """Finds permutation for keys(will be same for queries) and value weights of
    a transformer layer, by ensuring that the heads for key-queries and values
    move together.
    Args:
        assign_mat_kq:  Matrix indicating the weight matching scores(i.e., dot
                        products after permutation) for permuting weights from
                        position i to position j, along P_{n}_key axes.
        assign_mat_v:   Same as assign_mat_kq, but for permutations along P_{n}_value
                        axes.
        num_heads:      number of heads.
        hidden_dim:     hidden_dim of the model.
    Returns:
        1-D vectors indicating permutation to perform for key and value weights of
        transformer.

    Procedue:
        The score for permuting head i to head j is set as a sum of two quantites:
            1. Maximized cost obtained from solving LSAP for assigning weights
               in head i to head j, along P_key axes.
            2. Maximized cost obtained from solving LSAP for assigning weights
               in head i to head j, along P_value axes.
        Then we solve an outer LSAP for deciding which head to assign to which,
        using these scores for permuting between 2 heads.
    """
    block_size = head_size = hidden_dim // num_heads
    assignment_mat_kq = tile_windows(assign_mat_kq, num_heads, head_size)
    assignment_mat_v = tile_windows(assign_mat_v, num_heads, head_size)
    kq_shuffles = []
    v_shuffles = []
    costs = []

    for i in range(num_heads):
        kq_shuffles.append([])
        v_shuffles.append([])
        costs.append([])
        for j in range(num_heads):
            ri_kq, ci_kq = linear_sum_assignment(assignment_mat_kq[i, j], maximize=True)
            ri_v, ci_v = linear_sum_assignment(assignment_mat_v[i, j], maximize=True)
            assert (ri_kq == jnp.arange(len(ri_kq))).all()
            assert (ri_v == jnp.arange(len(ri_v))).all()
            kq_shuffles[-1].append(ci_kq)
            v_shuffles[-1].append(ci_v)
            assert (
                assignment_mat_v[i, j][ri_v, ci_v].sum()
                == assign_mat_v[i * block_size + ri_v, j * block_size + ci_v].sum()
            )
            costs[-1].append(
                (
                    assignment_mat_v[i, j][ri_v, ci_v].sum()
                    + assignment_mat_kq[i, j][ri_kq, ci_kq].sum()
                )
            )
    costs = jnp.array(costs)
    ri, ci = linear_sum_assignment(costs, maximize=True)
    # print("Maximum score:", costs[ri, ci].sum())
    assert (ri == jnp.arange(len(ri))).all()
    final_shuffles_kq = nest_shuffles(ci, kq_shuffles)
    final_shuffles_v = nest_shuffles(ci, v_shuffles)
    # print("s1+s2:=", s1+s2)
    return final_shuffles_kq, final_shuffles_v


def head_perm_weight_matching(
    rng,
    ps: PermutationSpec,
    params_a,
    params_b,
    max_iter=100,
    init_perm=None,
):
    """Find a permutation of `params_b` to make them match `params_a`."""
    perm_sizes = {
        p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()
    }

    perm = (
        {p: jnp.arange(n) for p, n in perm_sizes.items()}
        if init_perm is None
        else init_perm
    )
    perm_names = list(perm.keys())

    def get_perm_scores(
        p_ix,
    ):
        p = perm_names[p_ix]
        n = perm_sizes[p]
        A = jnp.zeros((n, n))
        for wk, axis in ps.perm_to_axes[p]:
            w_a = params_a[wk]
            w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)
            w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
            w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
            # print("A:", A)
            A += w_a @ w_b.T
        return A

    for iteration in range(max_iter):
        progress = False
        rng = rngmix(rng, iteration)
        for p_ix in jr.permutation(rngmix(rng, iteration), len(perm_names)):

            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = get_perm_scores(p_ix)

            if "value" in p:
                # Already accounted for in key
                continue

            if "key" in p:
                p_v = p.replace("key", "value")
                n_v = perm_sizes[p_v]
                A_v = get_perm_scores(perm_names.index(p_v))
                ci, ci_v = find_nested_perm(A, A_v, 12, 768)
            else:
                ri, ci = linear_sum_assignment(A, maximize=True)
                assert (ri == jnp.arange(len(ri))).all()

            oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
            newL = jnp.vdot(A, jnp.eye(n)[ci, :])
            logger.info(f"{iteration}/{p}: {newL - oldL}, {newL}, {oldL}")
            progress = progress or newL > oldL + 1e-12

            if "key" in p:
                oldL = jnp.vdot(A_v, jnp.eye(n_v)[perm[p_v]])
                newL = jnp.vdot(A_v, jnp.eye(n_v)[ci_v, :])
                logger.info(f"{iteration}/{p_v}: {newL - oldL}, {newL}, {oldL}")
                progress = progress or newL > oldL + 1e-12

                perm[p_v] = jnp.array(ci_v)

            perm[p] = jnp.array(ci)

        if not progress:
            break

    return perm
