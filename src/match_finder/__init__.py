import re
import random, string, shutil

import pandas as pd
from jax import random as jr
import jax.tree_util as jtu
from transformers import (
    FlaxAutoModelForSequenceClassification,
    AutoModelForSequenceClassification,
)

from .matching_algos.simple_match import weight_matching
from .trfrmr_perm_spec import get_trfrmr_permutation_spec
from .matching_algos.across_head_match import head_perm_weight_matching
from .utils import get_att_head_masks, apply_permutation, flat_dict_to_nested

def get_num_layers(params):
    layer_nums = []
    for key in set(params.keys()):
        match_obj = re.match(r'[^/]*/encoder/layer/(\d+)/', key)
        if match_obj is not None:
            layer_nums.append(int(match_obj.group(1)))
    return max(layer_nums)+1

def match_params(model1, model2, return_models="pt", head_perm=True, model_type="bert"):
    """Aligns params of model2 to that of model1 and returns the
    resultant models(model1 passed as it is).

    If head_perm is True, permutations of heads are also considered.
    Both model1 and model2 are converted to PyTorch if return_models is pt.

    model_type should be bert (for bert type models viz., bert-tiny, bert-small, etc.) or
    'roberta' (for roberta based models). 
    
    Bert Type models are supposed to have parameters with the following keys:
      1. bert/embeddings [with position, token_type and word embeddings within]
      2. bert/encoder with embedding layers
      3. followed by, bert/pooler/dense : a linear layer
      4. followed by, classifier/: a linear layer
    
    In roberta type models the third and fourth one is replaced with:
      3. classifier/out_proj: a linear layer.
      4. classifier/dense: a linear layer.
    """
    assert model_type in ['bert', 'roberta']
    if type(model1) is str:
        model1 = FlaxAutoModelForSequenceClassification.from_pretrained(model1)
    if type(model2) is str:
        model2 = FlaxAutoModelForSequenceClassification.from_pretrained(model2)

    params1 = pd.json_normalize(
        jtu.tree_map(lambda x: x, model1.params), sep="/"
    ).to_dict(orient="records")[0]
    params2 = pd.json_normalize(
        jtu.tree_map(lambda x: x, model2.params), sep="/"
    ).to_dict(orient="records")[0]

    model1_num_layers = get_num_layers(params1)
    model2_num_layers = get_num_layers(params1)
    assert model1_num_layers == model2_num_layers
    
    ps = get_trfrmr_permutation_spec(name = model_type, num_layers = model1_num_layers)
    masks = get_att_head_masks(ps, model1)
    rng = jr.PRNGKey(42)

    if head_perm:
        perm = head_perm_weight_matching(
            rng,
            ps,
            params1,
            params2,
        )
    else:
        perm = weight_matching(rng, ps, params1, params2, masks=masks)

    permuted_params2 = apply_permutation(
        ps,
        perm,
        params2,
    )
    nested_permuted_params2 = flat_dict_to_nested(permuted_params2)
    model2.params = nested_permuted_params2

    if return_models == "pt":
        model1_dir_name = "".join(random.choices(string.ascii_letters, k=20))
        model2_dir_name = "".join(random.choices(string.ascii_letters, k=20))
        model1.save_pretrained(model1_dir_name)
        model1 = AutoModelForSequenceClassification.from_pretrained(
            model1_dir_name, from_flax=True
        )
        model2.save_pretrained(model2_dir_name)
        model2 = AutoModelForSequenceClassification.from_pretrained(
            model2_dir_name, from_flax=True
        )
        shutil.rmtree(model1_dir_name)
        shutil.rmtree(model2_dir_name)

    return model1, model2
