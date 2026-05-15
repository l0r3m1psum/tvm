from typing import ClassVar

from tvm import TVMError
from tvm.ir import Op
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax.transform import PatternCheckContext

from ...pattern_registry import register_patterns

def relu_pattern():

    def _make_relu_pattern():
        input_tensor = wildcard()
        relu = is_op("relax.nn.relu")(input_tensor)

        annotations = {
            "input": input_tensor,
            "root": relu,
        }
        return relu, annotations

    def _check_relu(context: PatternCheckContext) -> bool:
        return True

    return ("strela.relu", *_make_relu_pattern(), _check_relu)

register_patterns(
    [
        relu_pattern(),
    ]
)
