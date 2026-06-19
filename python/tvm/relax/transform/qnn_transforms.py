import tvm_ffi
from tvm import ir, relax
import numpy
import warnings

@ir.transform.module_pass(opt_level=0)
class NormalizeQDQPatterns:
    # FIXME: the stuff about homogeneous functions is a munch of crap in this context.
    """Let f be an homogeneous function of degree 1 i.e. for all scalars alpha
    f(alpha x) = alpha f(x), let q be the quantization function and dq be the
    dequantization function. This transform rewrites
        * dq(f(x)) => f(dq(x)) and
        * f(q(x)) => q(f(x)),
    until a fixed point is reached.

    This is needed to make q(dq) patterns smaller and easier to match. The f
    usually applied after dequantization are reshape or permute_dims (or
    transpose) on constants which are expected to be removed by a later constant
    folding step. The f usually applied before quantization is an activation
    which is expected to be fused by a later operator fusion step.

                dq
               /
    dq        transpose     dq        dq
      \      /                \      /
       matmul                  conv2d
         |                       |
        add-dq                  add-reshape-dq
         |                       |
        relu                 leakyrelu
         |                       |
         q                       q
    """
    def transform_module(self, mod, ctx):

        # homogeneous if the condition is not a function of alpha
        #     masked_fill, where
        # homogeneous of degree 0 if input scaled together
        #     equal, greater, greater_equal, less, less_equal, not_equal
        # homogeneous of degree 1 if input scaled together
        #     add, subtract, maximum, minimum
        # homogeneous of degree 1 or 2 if scaled 1 or two arguments
        #     einsum, linear, matmul, outer
        #     conv1d, conv2d, conv3d
        #     conv1d_transpose, conv2d_transpose, conv3d_transpose

        # TODO: This would be the dream but supporting all this stuff requires a
        # lot of work.
        homogeneous_func = (
            relax.dpl.is_op("relax.broadcast_to")
            | relax.dpl.is_op("relax.concat")
            | relax.dpl.is_op("relax.expand_dims")
            | relax.dpl.is_op("relax.flatten")
            | relax.dpl.is_op("relax.flip")
            | relax.dpl.is_op("relax.gather_elements")
            | relax.dpl.is_op("relax.gather_nd")
            | relax.dpl.is_op("relax.index_put")
            | relax.dpl.is_op("relax.index_tensor")
            | relax.dpl.is_op("relax.layout_transform")
            | relax.dpl.is_op("relax.permute_dims")
            | relax.dpl.is_op("relax.repeat")
            | relax.dpl.is_op("relax.reshape")
            | relax.dpl.is_op("relax.scatter_elements")
            | relax.dpl.is_op("relax.scatter_nd")
            | relax.dpl.is_op("relax.slice_scatter")
            | relax.dpl.is_op("relax.split")
            | relax.dpl.is_op("relax.squeeze")
            | relax.dpl.is_op("relax.stack")
            | relax.dpl.is_op("relax.tile")
            | relax.dpl.is_op("relax.dynamic_strided_slice")
            | relax.dpl.is_op("relax.strided_slice")
            | relax.dpl.is_op("relax.take")

            # TODO: probably works only if the zero point is left unchanged
            # | relax.dpl.is_op("relax.abs")
            # | relax.dpl.is_op("relax.negative")

            | relax.dpl.is_op("relax.vision.roi_align")
            | relax.dpl.is_op("relax.vision.roi_pool")

            # For reductions we have to assume -ffast-math
            # | relax.dpl.is_op("relax.max")
            # | relax.dpl.is_op("relax.min")
            # | relax.dpl.is_op("relax.mean")
            # | relax.dpl.is_op("relax.median")
            # | relax.dpl.is_op("relax.sum")
            # | relax.dpl.is_op("relax.collapse_sum_like")
            # | relax.dpl.is_op("relax.collapse_sum_to")
            # | relax.dpl.is_op("relax.cumsum")
            # | relax.dpl.is_op("relax.std")

            # Remember that scale is always positive
            | relax.dpl.is_op("relax.sort")
            | relax.dpl.is_op("relax.topk")

            # This are homogeneous of degree 0 i.e. f(alpha x) = f(x)
            | relax.dpl.is_op("relax.argmax")
            | relax.dpl.is_op("relax.argmin")
            | relax.dpl.is_op("relax.argsort")
            | relax.dpl.is_op("relax.nonzero")
            | relax.dpl.is_op("relax.bucketize")
            # | relax.dpl.is_op("relax.unique")
            | relax.dpl.is_op("relax.shape_of")
            | relax.dpl.is_op("relax.size")
            | relax.dpl.is_op("relax.tensor_to_shape")
            # TODO: not sure about those
            # | relax.dpl.is_op("relax.isfinite")
            # | relax.dpl.is_op("relax.isinf")
            # | relax.dpl.is_op("relax.isnan")
            # | relax.dpl.is_op("relax.sign")

            # TODO: this works only if the zero point is left unchanged
            # | relax.dpl.is_op("relax.nn.relu")
            # TODO: not sure about those
            # | relax.dpl.is_op("relax.nn.leakyrelu")
            # | relax.dpl.is_op("relax.nn.prelu")

            | relax.dpl.is_op("relax.nn.max_pool1d")
            | relax.dpl.is_op("relax.nn.max_pool2d")
            | relax.dpl.is_op("relax.nn.max_pool3d")

            | relax.dpl.is_op("relax.nn.batch_flatten")
            | relax.dpl.is_op("relax.nn.pad")
            | relax.dpl.is_op("relax.nn.pixel_shuffle")
            | relax.dpl.is_op("relax.nn.dropout")
        )

        homogeneous_func = lambda x: (
            relax.dpl.is_op("relax.reshape")(x, relax.dpl.wildcard())
            | relax.dpl.is_op("relax.permute_dims")(x)
            | relax.dpl.is_op("relax.nn.relu")(x)
        )

        # f(dq(x)) => dq(f(x))
        dq_pattern_input = relax.dpl.wildcard()
        dq_pattern_middle = relax.dpl.is_op("relax.dequantize")(
            dq_pattern_input, relax.dpl.wildcard(), relax.dpl.wildcard()
        )
        dq_pattern_output = homogeneous_func(dq_pattern_middle)

        # q(f(x)) = f(q(x))
        q_pattern_input = relax.dpl.wildcard()
        q_pattern_middle = homogeneous_func(q_pattern_input)
        q_pattern_output = relax.dpl.is_op("relax.quantize")(
            q_pattern_middle, relax.dpl.wildcard(), relax.dpl.wildcard()
        )

        pattern = dq_pattern_output | q_pattern_output

        def rewriter(call: relax.Call, match_map: ir.Map) -> relax.Expr:
            if dq_pattern_output in match_map:
                input = match_map[dq_pattern_input]
                middle = match_map[dq_pattern_middle]
                output = match_map[dq_pattern_output]
                raise RuntimeError("Not yet implemented...")
            else:
                assert q_pattern_output in match_map
                input = match_map[q_pattern_input]
                middle = match_map[q_pattern_middle]
                output = match_map[q_pattern_output]
                if middle.op.name == "relax.reshape":
                    raise RuntimeError("Not yet implemented...")
                elif middle.op.name == "relax.permute_dims":
                    raise RuntimeError("Not yet implemented...")
                elif middle.op.name == "relax.nn.relu":
                    raise RuntimeError("Not yet implemented...")
                else:
                    assert False
            return res

        for global_var, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_func = relax.dpl.rewrite_call(pattern, rewriter, func)
                new_func = relax.analysis.remove_all_unused(new_func)
                mod.update_func(global_var, new_func)

        return mod

def reshape_if_needed(ndim: int, const: relax.Constant, axis: int) -> relax.Constant:
    res = const
    shape_values = const.struct_info.shape.values
    if shape_values:
        if len(shape_values) != 1:
            raise ValueError("Only vectors are supported")
        shape = [1 for _ in range(ndim)]
        shape[axis] = int(shape_values[0])
        res = relax.const(const.data.numpy().reshape(shape))
    return res

@ir.transform.module_pass(opt_level=0)
class RewriteQDQPatternsToQNNOps:
    def transform_module(self, mod, ctx):

        qbilinear_dq_x = relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.is_const(), relax.dpl.is_const()
        )
        qbilinear_dq_w = relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.is_const(), relax.dpl.is_const()
        )
        qbilinear_dq_b = relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.is_const(), relax.dpl.is_const()
        )
        qbilinear_op = (
            relax.dpl.is_op("relax.nn.conv2d") | relax.dpl.is_op("relax.matmul")
        )(qbilinear_dq_x, qbilinear_dq_w)
        qbilinear_bias = relax.dpl.is_op("relax.add")(
            qbilinear_op, qbilinear_dq_b
        ) | qbilinear_op
        qbilinear_q_y = relax.dpl.is_op("relax.quantize")(
            qbilinear_bias, relax.dpl.is_const(), relax.dpl.is_const()
        )

        qadd_dq_a = relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.is_const(), relax.dpl.is_const()
        )
        qadd_dq_b = relax.dpl.is_op("relax.dequantize")(
            relax.dpl.wildcard(), relax.dpl.is_const(), relax.dpl.is_const()
        )
        qadd_op = relax.dpl.is_op("relax.add")(qadd_dq_a, qadd_dq_b)
        qadd_q_c = relax.dpl.is_op("relax.quantize")(
            qadd_op, relax.dpl.is_const(), relax.dpl.is_const()
        )

        # A single pattern must be used in this case because if this is
        # splitted in two passes of relax.dpl.rewrite_call (or using
        # relax.transform.FuseOpsByPattern with two different patterns) there
        # are some dequantize which have out degree 2 i.e are used by more than
        # one node after. If two patterns are used the dequantize node are
        # removed by the first and the second one can't use it.
        pattern = qadd_q_c | qbilinear_q_y

        def rewriter(call: relax.Call, match_map: ir.Map) -> relax.Expr:
            if qadd_q_c in match_map:
                a, a_s, a_zp = match_map[qadd_dq_a].args
                b, b_s, b_zp = match_map[qadd_dq_b].args
                c, c_s, c_zp = match_map[qadd_q_c].args

                a_axis = match_map[qadd_dq_a].attrs.axis
                a_ndim = a.struct_info.ndim
                b_axis = match_map[qadd_dq_b].attrs.axis
                b_ndim = b.struct_info.ndim
                c_axis = match_map[qadd_q_c].attrs.axis
                c_ndim = c.struct_info.ndim

                a_s = reshape_if_needed(a_ndim, a_s, a_axis)
                a_zp = reshape_if_needed(a_ndim, a_zp, a_axis)
                b_s = reshape_if_needed(b_ndim, b_s, b_axis)
                b_zp = reshape_if_needed(b_ndim, b_zp, b_axis)
                c_s = reshape_if_needed(c_ndim, c_s, c_axis)
                c_zp = reshape_if_needed(c_ndim, c_zp, c_axis)

                res = relax.op.qnn.add(a, a_s, a_zp, b, b_s, b_zp, c_s, c_zp)
            else:
                assert qbilinear_q_y in match_map
                x, x_s, x_zp = match_map[qbilinear_dq_x].args
                w, w_s, w_zp = match_map[qbilinear_dq_w].args
                y, y_s, y_zp = match_map[qbilinear_q_y].args

                x_axis = match_map[qbilinear_dq_x].attrs.axis
                x_ndim = x.struct_info.ndim
                w_axis = match_map[qbilinear_dq_w].attrs.axis
                w_ndim = w.struct_info.ndim
                y_axis = match_map[qbilinear_q_y].attrs.axis
                y_ndim = y.struct_info.ndim

                x_s = reshape_if_needed(x_ndim, x_s, x_axis)
                x_zp = reshape_if_needed(x_ndim, x_zp, x_axis)
                w_s = reshape_if_needed(w_ndim, w_s, w_axis)
                w_zp = reshape_if_needed(w_ndim, w_zp, w_axis)
                y_s = reshape_if_needed(y_ndim, y_s, y_axis)
                y_zp = reshape_if_needed(y_ndim, y_zp, y_axis)

                args = [x, x_s, x_zp, w, w_s, w_zp, y_s, y_zp]

                if qbilinear_dq_b in match_map:
                    b, b_s, b_zp = match_map[qbilinear_dq_b].args

                    b_s_old = b_s.data.numpy()
                    b_zp_old = b_zp.data.numpy()

                    b_s_new = x_s.data.numpy() * w_s.data.numpy()
                    # This comparison is safe to do even in floating point since
                    # IEEE-754 binary floating point multiplication is correct
                    # up to rounding and commutative. The only catch is that the
                    # global floating point rounding mode needs to be the same
                    # as one specified by ONNX. Hence we use allclose anyway.
                    if not numpy.allclose(b_s_old, b_s_new) or (b_zp_old != 0).any():
                        warnings.warn("requantizing bias")
                        b_old = (b.data.numpy() - b_zp_old) * b_s_old

                        b_new = b_old / b_s_new # + 0
                        assert b.data.numpy().size == b_new.size, \
                            "An unexpected broadcasting happened"

                        b = relax.const(
                            numpy.clip(
                                numpy.round(b_new),
                                numpy.iinfo("int32").min,
                                numpy.iinfo("int32").max,
                            ).astype("int32")
                        )

                    args.append(b)

                bilinear_call = match_map[qbilinear_op]
                if bilinear_call.op.name == "relax.matmul":
                    op = ir.Op.get("relax.qnn.linear")
                elif bilinear_call.op.name == "relax.nn.conv2d":
                    op = ir.Op.get("relax.qnn.conv2d")
                else:
                    assert False
                res = relax.Call(op, args, bilinear_call.attrs)
            return res

        for global_var, func in mod.functions.items():
            if isinstance(func, relax.Function):
                new_func = relax.dpl.rewrite_call(pattern, rewriter, func)
                new_func = relax.analysis.remove_all_unused(new_func)
                mod.update_func(global_var, new_func)

        return mod

from typing import List
from tvm import topi

def to_int_list(xs: tvm_ffi.Array) -> List[int]:
    return [int(x) for x in xs]

def is_broadcastable(*shapes) -> bool:
    try:
        numpy.broadcast_shapes(*shapes)
        return True
    except ValueError:
        return False

# NOTE: This is done this way instead of using clip to accommodate VTA but for
# a more generic implementation it might be better to use relax.op.clip
def clamp(data: relax.Expr, min, max) -> relax.Expr:
    res = relax.op.minimum(data, relax.const(max))
    res = relax.op.maximum(relax.const(min), res)
    return res

def const_astype(x: relax.Constant, dtype: str) -> relax.Constant:
    # TODO: check that conversion is safe to do
    return relax.const(x.data.numpy().astype(dtype))

def requantize(s: relax.Constant, x: relax.Expr, z: relax.Constant) -> relax.Expr:
    res = x
    if not (s.data.numpy() == 1).all():
        res *= s
    if not (z.data.numpy() == 0).all():
        res += const_astype(z, "float32")
    res = relax.op.round(res)
    res = clamp(res, -128., 127.).astype("int8")
    return res

# TODO: this should accept a parameter for simulated quantization...
@relax.expr_functor.mutator
class LowerQNNOpsMutator(relax.PyExprMutator):
    def visit_call_(self, call: relax.Call) -> relax.Call:
        res = call
        op_name = getattr(call.op, "name", "")
        if op_name == "relax.qnn.conv2d" or op_name == "relax.qnn.linear":
            (
                x, x_s, x_zp,
                w, w_s, w_zp,
                y_s, y_zp,
            ) = call.args[:8]
            b = call.args[8] if len(call.args) == 9 else None

            if (
                not is_broadcastable(
                    to_int_list(x.struct_info.shape.values),
                    to_int_list(x_s.struct_info.shape.values),
                    to_int_list(x_zp.struct_info.shape.values),
                ) or not is_broadcastable(
                    to_int_list(w.struct_info.shape.values),
                    to_int_list(w_s.struct_info.shape.values),
                    to_int_list(w_zp.struct_info.shape.values),
                ) or not is_broadcastable(
                    to_int_list(call.struct_info.shape.values),
                    to_int_list(y_s.struct_info.shape.values),
                    to_int_list(y_zp.struct_info.shape.values),
                )
            ):
                raise ValueError("Per-block quantization is not supported yet")

            # The result should be correctly rounded.
            m = relax.const(
                (x_s.data.numpy().astype("float64")
                    * w_s.data.numpy().astype("float64"))
                / y_s.data.numpy().astype("float64"),
                dtype="float32"
            )
            assert m.data.numpy().size == max(
                x_s.data.numpy().size, w_s.data.numpy().size, y_s.data.numpy().size
            ), "An unexpected broadcasting happened e.g. (C_out, 1, 1, 1) * (1, C_out, 1, 1) = (C_out, C_out, 1, 1)"

            x_ones = relax.op.ones_like(x)
            w_ones = relax.op.ones_like(w)
            x_zp_int = const_astype(x_zp, "int32")
            w_zp_int = const_astype(w_zp, "int32")

            x_zp_all_zero = (x_zp.data.numpy() == 0).all()
            w_zp_all_zero = (w_zp.data.numpy() == 0).all()

            if op_name == "relax.qnn.conv2d":
                kwargs = relax.op.qnn.conv2d_attrs_to_dict(call.attrs)
                kwargs["out_dtype"] = "int32"
                res = relax.op.nn.conv2d(x, w, **kwargs)
                if not x_zp_all_zero:
                    res -= x_zp_int*relax.op.nn.conv2d(x_ones, w, **kwargs)
                if not w_zp_all_zero:
                    res -= w_zp_int*relax.op.nn.conv2d(x, w_ones, **kwargs)
                if not x_zp_all_zero and not w_zp_all_zero:
                    res += x_zp_int*w_zp_int*relax.op.nn.conv2d(x_ones, w_ones, **kwargs)
            elif op_name == "relax.qnn.linear":
                rows, cols = -2, -1
                # From "Quantization and Training of Neural Networks for Efficient
                # Integer-Arithmetic-Only Inference"
                n = relax.const(topi.utils.get_const_tuple(relax.get_shape_of(x))[-1])
                res = relax.op.matmul(x, w, out_dtype="int32")
                if not x_zp_all_zero:
                    res -= x_zp_int*relax.op.sum(w.astype("int32"), axis=rows, keepdims=True)
                if not w_zp_all_zero:
                    res -= w_zp_int*relax.op.sum(x.astype("int32"), axis=cols, keepdims=True)
                if not x_zp_all_zero and not w_zp_all_zero:
                    res += n*x_zp_int*w_zp_int
            else:
                assert False

            if b:
                res += b

            res = requantize(m, res.astype("float32"), y_zp)
        elif op_name == "relax.qnn.add":
            (
                a, a_s, a_zp,
                b, b_s, b_zp,
                c_s, c_zp,
            ) = call.args

            if (
                not is_broadcastable(
                    to_int_list(a.struct_info.shape.values),
                    to_int_list(a_s.struct_info.shape.values),
                    to_int_list(a_zp.struct_info.shape.values),
                ) or not is_broadcastable(
                    to_int_list(b.struct_info.shape.values),
                    to_int_list(b_s.struct_info.shape.values),
                    to_int_list(b_zp.struct_info.shape.values),
                ) or not is_broadcastable(
                    to_int_list(call.struct_info.shape.values),
                    to_int_list(c_s.struct_info.shape.values),
                    to_int_list(c_zp.struct_info.shape.values),
                )
            ):
                raise ValueError("Per-block quantization is not supported yet")

            # C = (A_s * (A - A_z) + B_s * (B - B_z))/C_s + C_z
            # C = A_s/C_s * (A - A_z) + B_s/C_s * (B - B_z) + C_z
            c_s_float = c_s.data.numpy()
            a_m_float = a_s.data.numpy() / c_s_float
            b_m_float = b_s.data.numpy() / c_s_float
            a_m = relax.const(a_m_float)
            b_m = relax.const(b_m_float)

            lhs = a.astype("int32")
            if not (a_zp.data.numpy() == 0).all():
                lhs -= const_astype(a_zp, "int32")
            lhs = lhs.astype("float32")
            if not (a_m_float == 1).all():
                lhs *= a_m

            rhs = b.astype("int32")
            if not (b_zp.data.numpy() == 0).all():
                rhs -= const_astype(b_zp, "int32")
            rhs = rhs.astype("float32")
            if not (b_m_float == 1).all():
                rhs *= b_m

            res = lhs + rhs
            res = requantize(relax.const(1.0), res, c_zp)
        elif op_name == "relax.qnn.avg_pool2d":
            (
                x, x_s, x_zp,
                   y_s, y_zp,
            ) = call.args

            if (
                not is_broadcastable(
                    to_int_list(x.struct_info.shape.values),
                    to_int_list(x_s.struct_info.shape.values),
                    to_int_list(x_zp.struct_info.shape.values),
                ) or not is_broadcastable(
                    to_int_list(call.struct_info.shape.values),
                    to_int_list(y_s.struct_info.shape.values),
                    to_int_list(y_zp.struct_info.shape.values),
                )
            ):
                raise ValueError("Per-block quantization is not supported yet")

            # FIXME: Doing avg_pool with integer division truncates towards zero
            # as opposed to the round to nearest semantics of standard quantized
            # operations. The fix should be to lower avg_pool2d into an explicit
            # sum followed by a fixed-point multiplication.
            # NOTE: can be implemented as a convolution with a kernel of ones
            # followed by a division by N (which can be fused in the multiplier)
            m = relax.const(x_s.data.numpy()/y_s.data.numpy())
            res = relax.op.nn.avg_pool2d(
                data=x.astype("int32"),
                **relax.op.qnn.avg_pool2d_attrs_to_dict(call.attrs)
            ).astype("int32")
            if (x_zp.data.numpy() != 0).any():
                res -= const_astype(x_zp, "int32")

            res = requantize(m, res.astype("float32"), y_zp)

        if res is call:
            res = super().visit_call_(call)

        return res

@ir.transform.module_pass(opt_level=0)
class LowerQNNOps:
    def transform_module(self, mod: ir.IRModule, _ctx: ir.transform.PassContext) -> ir.IRModule:
        mutator = LowerQNNOpsMutator(mod)

        for global_var, func in mod.functions.items():
            if isinstance(func, relax.Function):
                updated_func = mutator.visit_expr(func)
                updated_func = relax.analysis.remove_all_unused(updated_func)
                mod.update_func(global_var, updated_func)

        return mod
