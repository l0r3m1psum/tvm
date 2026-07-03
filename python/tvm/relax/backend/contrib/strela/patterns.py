from tvm.relax.dpl.pattern import is_op, wildcard, is_const
from tvm.relax.transform import PatternCheckContext, FusionPattern

from ...pattern_registry import register_patterns

def relu_pattern() -> FusionPattern:

    input_tensor = wildcard()
    relu = is_op("relax.nn.relu")(input_tensor)

    annotations = {
        "input": input_tensor,
        "root": relu,
    }

    def _check_relu(context: PatternCheckContext) -> bool:
        return True

    return FusionPattern("strela.relu", relu, annotations, _check_relu)

def qnn_linear_pattern() -> FusionPattern:
    """This pattern is fragile since it is susceptible to both commutativity and
    associativity of the operators involved. Not only that but if the patter is
    written with matrix multiplications and matrices of ones, instead of sums
    again it would not be recognized. To solve this kind of problems an approach
    like equality saturation would be needed.

    "Latent Idiom Recognition for a Minimalist Functional Array Language using
    Equality Saturation"
    """

    weight = wildcard().has_dtype("int8") # Not is_const because of permutation
    input = wildcard().has_dtype("int8")
    weight_zp = is_const().has_dtype("int32").has_shape(())
    input_zp = is_const().has_dtype("int32").has_shape(())
    n = is_const().has_dtype("int32")

    # W x - z_x sum(W) - z_w sum(x) + n z_x z_w
    weight_input = is_op("relax.matmul")(input, weight).has_dtype("int32")
    input_zp_weight = is_op("relax.multiply")(
        input_zp, is_op("relax.sum")(is_op("relax.astype")(weight))
    )
    weight_zp_input = is_op("relax.multiply")(
        weight_zp, is_op("relax.sum")(is_op("relax.astype")(input))
    )
    input_zp_weight_zp = is_op("relax.multiply")(
        n, is_op("relax.multiply")(input_zp, weight_zp)
    )
    pattern = weight_input
    pattern = pattern | is_op("relax.subtract")(weight_input, input_zp_weight)
    pattern = pattern | is_op("relax.subtract")(pattern, weight_zp_input)
    pattern = pattern | is_op("relax.add")(pattern, input_zp_weight_zp)

    annotations = {
        "weight": weight,
        "input": input,
        "weight_zp": weight_zp,
        "input_zp": input_zp,

        "weight_input": weight_input,
        "input_zp_weight": input_zp_weight,
        "weight_zp_input": weight_zp_input,
        "input_zp_weight_zp": input_zp_weight_zp,
    }

    def _check_qnn_linear(context: PatternCheckContext) -> bool:
        # Check that the pattern is realizable by expanding (W-z_w)(x-z_x)
        # (W-z_w) (x-z_x) = W x - z_x sum(W) - z_w sum(x) + n z_x z_w
        # W (x-z_x)       = W x - z_x sum(W)
        # (W-z_w) x       = W x - z_w sum(x)
        # W x             = W x
        # W x must be present, n z_x z_w is present iff both z_x sum(W) and
        # z_w sum(x) are.
        map = context.annotated_expr
        res = (
            "weight_input" in map
            and (("input_zp_weight_zp" in map)
                == ("input_zp_weight" in map and "weight_zp_input" in map)
            )
        )
        return res

    return FusionPattern("strela.qnn_linear", pattern, annotations, _check_qnn_linear)

register_patterns(
    [
        relu_pattern(),
        qnn_linear_pattern(),
    ]
)
