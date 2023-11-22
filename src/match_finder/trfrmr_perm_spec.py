from collections import ChainMap
from .perm_spec import permutation_spec_from_axes_to_perm

layer_norm = lambda name, p_in: {f"{name}/LayerNorm/bias": (p_in,), f"{name}/LayerNorm/scale": (p_in,)}

linear_layer = lambda name, p_in, p_out: {f"{name}/kernel": (p_in, p_out), f"{name}/bias": (p_out,)}

embedding = lambda name, p_in : {f"{name}/position_embeddings/embedding": (None, p_in),
                                 f"{name}/token_type_embeddings/embedding": (None, p_in),
                                 f"{name}/word_embeddings/embedding": (None, p_in),
                                 **layer_norm(f"{name}", p_in)}

attention_self = lambda name, key_in, key_out, value_out: {**linear_layer(f"{name}/self/key", key_in, key_out),
                                                           **linear_layer(f"{name}/self/query", key_in, key_out),
                                                           **linear_layer(f"{name}/self/value", key_in, value_out)}

attention_output = lambda name, p_in, p_out : {**linear_layer(f"{name}/output/dense", p_in, p_out),
                                               **layer_norm(f"{name}/output", p_out),}

attention = lambda name, key_in, key_out, value_out : {**attention_self(f"{name}/attention", key_in, key_out, value_out),
                                                       **attention_output(f"{name}/attention", value_out, key_in),}

ffn = lambda name, p_in, p_interim : {**linear_layer(f"{name}/intermediate/dense", p_in, p_interim),
                                      **linear_layer(f"{name}/output/dense", p_interim, p_in),
                                      **layer_norm(f"{name}/output", p_in)}

trfrmr_layer = lambda name, layer_num, p_in, key_out, value_out, p_interim: {**attention(f"{name}/layer/{layer_num}", p_in, key_out, value_out),
                                                                             **ffn(f"{name}/layer/{layer_num}", p_in, p_interim)}

def get_trfrmr_permutation_spec(name="bert", num_layers=12):

    if "roberta" in name.lower():
        return permutation_spec_from_axes_to_perm(
                    dict(ChainMap(
                            *([embedding(name+"/embeddings", "P_residual"),]+
                            [
                                trfrmr_layer(name+"/encoder", i, "P_residual", f"P_key_{i}", f"P_value_{i}", f"P_interim_{i}")
                                for i in range(num_layers)
                            ]+
                            [
                                linear_layer("classifier/out_proj", "P_residual", "P_final"),
                                linear_layer("classifier/dense", "P_final", None),
                            ])
                        )
                    )
                )
    
    return permutation_spec_from_axes_to_perm(
                dict(ChainMap(
                        *([embedding(name+"/embeddings", "P_residual"),]+
                        [
                            trfrmr_layer(name+"/encoder", i, "P_residual", f"P_key_{i}", f"P_value_{i}", f"P_interim_{i}")
                            for i in range(num_layers)
                        ]+
                        [
                            linear_layer(name+"/pooler/dense", "P_residual", "P_final"),
                            linear_layer("classifier", "P_final", None),
                        ])
                    )
                )
            )    
