from typing import Dict

import jax.numpy as jnp
import jax.random as jr

from .perm_spec import PermutationSpec
from transformers import FlaxAutoModelForSequenceClassification

rngmix = lambda rng, x: jr.fold_in(rng, hash(x))

def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    """Get parameter `k` from `params`, with the permutations applied."""
    w = params[k]
    for axis, p in enumerate(ps.axes_to_perm[k]):
        # Skip the axis we're trying to permute.
        if axis == except_axis:
            continue

        # None indicates that there is no permutation relevant to that axis.
        if p is not None:
            w = jnp.take(w, perm[p], axis=axis)

    return w


def apply_permutation(ps: PermutationSpec, perm, params):
    """Apply a `perm` to `params`."""
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def get_att_head_masks(
    ps: PermutationSpec, model: FlaxAutoModelForSequenceClassification
) -> Dict[str, jnp.array]:
    """Returns a dictionary with masks for dis-allowing permutations across heads"""
    diag = jnp.diag(jnp.ones(model.config.num_attention_heads))
    mask = jnp.kron(diag, jnp.ones((64, 64)))
    mask = jnp.where(mask == 0, -100000000, 0)
    masks = {}
    for perm in ps.perm_to_axes.keys():
        if "key" in perm or "value" in perm:
            masks[perm] = mask
    return masks


def flat_dict_to_nested(flat_params, sep="/"):
    """Converts a flat dictionary with keys that having sub-categories
    demarcated by the sep token, into a nested dictionary. E.g.:
    {"ax/by/z": 2, "ax/by/l": 3, "za": 5} -> {"ax" : {"by": {"z" : 2,
                                                             "l" :3
                                                             }
                                                      }
                                                "za" : 5
                                              }
    """
    if set(flat_params.keys()) == set({""}):
        assert len(flat_params) == 1
        return list(flat_params.values())[0]
    final_dict = {}
    key_wise_splits = {}

    for key, value in flat_params.items():
        top_key = key.split(sep)[0]
        if top_key not in key_wise_splits:
            key_wise_splits[top_key] = {}
        key_wise_splits[top_key][key[len(top_key) + 1 :]] = value

    for top_key, sub_dict in key_wise_splits.items():
        final_dict[top_key] = flat_dict_to_nested(sub_dict)

    return final_dict


def tile_windows(mat, num_windows, window_size):
    """Takess an nXn matrix mat, and reshapes it to
    a matrix of size (num_windows, num_windows, window_size, window_size),
    taking windows of size (window_size, window_size) with a stride of window_size.
    For e.g.:
    [
        [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12],
        [13,14,15,16]
    ] gets converted to:
    [
        [
            [
                [1,2],
                [5,6]
            ],
            [
                [3,4],
                [7,8]
            ]
        ],
        [
            [
                [9,10],
                [13,14]
            ],
            [
                [11,12],
                [15,16]
            ]
        ]
    ]
    """
    windows = []
    for i in range(num_windows):
        windows.append([])
        for j in range(num_windows):
            windows[-1].append(
                mat[
                    i * window_size : (i + 1) * window_size,
                    j * window_size : (j + 1) * window_size,
                ]
            )
    return jnp.array(windows)

def nest_shuffles(head_shuffles, within_head_shuffles):
    """
    Calculates permutaion for all weights in all heads from:
        1. permutation of heads(head_shuffles), and 
        2. shuffles to perform in head i when head i gets 
           mapped to head j(within_head_shuffles[i][j]).
    Returns:
        A 1-D array where the i-th value tells at what position
        the i-th weight needs to be permuted to.
    
    NOTE: permutation of weights is actually permutation of/along an axis
          of the weight matrix.
    """
    block_size = len(within_head_shuffles[0][0])
    #print("Block size:", block_size)
    final_shuffle = []
    
    for i, head_num in enumerate(head_shuffles):
        i_to_head_num_perm = within_head_shuffles[i][head_num]
        for j, num in enumerate(i_to_head_num_perm):
            final_shuffle.append(block_size*head_num+num)
    return jnp.array(final_shuffle)