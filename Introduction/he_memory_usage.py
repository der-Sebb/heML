import numpy as np
import tenseal as ts
from memory_profiler import profile

#execute desired method and run file with python -m memory_profiler he_memory_usage.py

@profile
def createDeepContext():
    poly_mod_degree = 2 ** 14
    bits_scale = 40

    coeff_mod_bit_sizes=[60, bits_scale, bits_scale, bits_scale, bits_scale, 60]
    ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)

    ctx_training.global_scale = pow(2, bits_scale)
    ctx_training.generate_galois_keys()
    return ctx_training

@profile
def createShallowContext():
    poly_mod_degree = 2 ** 13
    bits_scale = 40

    coeff_mod_bit_sizes=[60, bits_scale, 60]
    ctx_training = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)

    ctx_training.global_scale = pow(2, bits_scale)
    ctx_training.generate_galois_keys()
    return ctx_training

@profile
def encryptValue(context):
    e_value = ts.ckks_vector(context, [42])
    return e_value

context = createShallowContext()
e_value = encryptValue(context)