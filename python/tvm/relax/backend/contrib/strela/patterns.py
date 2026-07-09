from tvm.relax.dpl.pattern import is_op, wildcard, is_const
from tvm.relax.transform import PatternCheckContext, FusionPattern

from ...pattern_registry import register_patterns

def _check_always_ok(context: PatternCheckContext) -> bool:
    return True

def relu_pattern() -> FusionPattern:

    input_tensor = wildcard().has_dtype("int32")
    relu = is_op("relax.nn.relu")(input_tensor)

    annotations = {
        "input": input_tensor,
        "root": relu,
    }

    return FusionPattern("strela.relu", relu, annotations, _check_always_ok)

def centered_biliear_product_pattern() -> FusionPattern:
    """This pattern is more robust than the non factorized version e.g. for
    matrix multiplication

        (W-z_w) (x-z_x) = W x - z_x sum(W) - z_w sum(x) + n z_x z_w

    which is susceptible to both commutativity and associativity of the
    operators involved. Notice also that instead of using sum in the above
    equation we could have used matrix multiplication and matrices of ones.
    Solving this idiom recognition in the general case requires an equality
    saturation approach [0].

    This pattern is still not invariant to rewriting subtraction as addition but
    given that the LHS and RHS are the same it is not susceptible to
    commutativity. The problem can be solving by canonicalizing subtraction by
    a constant as addition with that negated constant, like LLVM does [1], but
    TVM doesn't do this.

    [0]: "Latent Idiom Recognition for a Minimalist Functional Array Language using Equality Saturation"
    [1]: https://github.com/llvm/llvm-project/blob/4ffcfdfac177a7e7b3d9be20f73dd9d2890da006/llvm/lib/Transforms/InstCombine/InstCombineAddSub.cpp#L2451
    """

    # This is not a constant because operations like permutation.
    weight = wildcard().has_dtype("int8")
    input = wildcard().has_dtype("int8")
    weight_zp = is_const().has_dtype("int8")
    # Lifting the constant requirement here could allow for dynamic quantization.
    input_zp = is_const().has_dtype("int8")

    centered_input = is_op("relax.subtract")(
        is_op("relax.astype")(input).has_dtype("int32"),
        is_op("relax.astype")(input_zp).has_dtype("int32"),
    )
    centered_weight = is_op("relax.subtract")(
        is_op("relax.astype")(weight).has_dtype("int32"),
        is_op("relax.astype")(weight_zp).has_dtype("int32"),
    )
    pattern = (
        is_op("relax.matmul") | is_op("relax.nn.conv2d") | is_op("relax.multiply")
    )(
        centered_input, centered_weight
    )

    # The name of the fused operation is the topological order of the operations
    # plus the order of the arguments of the various operations (starting from
    # the root). For this pattern it is
    # fused
    #     _relax_astype_relax_astype_relax_subtract
    #     _relax_astype_relax_astype_relax_subtract
    #     _relax_(matmul|nn_conv2d|multiply)
    # Notice that the first "astype astype subtract" is the one of the input
    # and the second one is the one of the weight.
    # Due to graph homomorphism the we could match a homomorphic version of the
    # pattern e.g.
    # i = z.astype("int32")
    # c = i - i
    # y = matmul(c, c)
    # and the name the fused operation would change...


    annotations = {
        "weight": weight,
        "input": input,
        "weight_zp": weight_zp,
        "input_zp": input_zp,

        "root": pattern,
    }

    return FusionPattern(
        "strela.centered_biliear_product", pattern, annotations, _check_always_ok
    )

# This should match operations like addition and concatenation
# C = A_s/C_s * (A - A_z) + B_s/C_s * (B - B_z)
def centered_scaled_liear_sum_pattern() -> FusionPattern:
    raise RuntimeError("Not implemented yet!")

register_patterns(
    [
        relu_pattern(),
        centered_biliear_product_pattern(),
    ]
)
