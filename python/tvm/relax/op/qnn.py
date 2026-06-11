"""Quantized operations. The semantic and arguments of the various operations is
modeled after ONNX and TensorFlow Lite to import models from them. In particular
to infer the output type of a quantized operation the type of the zero point (of
the output) is used e.g. QlinearConv [0].

The aim of this operators is to support legalizing this operations to implement:
  * normal quantization
  * integer-arithmetic-only
  * simulated quantization
All of this can be done by having inputs and zero points in integer formats.

Per-axis and per-block quantization is supported via the shape of the
quantization parameters e.g. if the shape of the quantized data is (M,) and we
want to have M/N quantization blocks than the quantization parameter will have
the same shape (M/N,) assuming that M%N = 0.

Broadcasting scale to zero point and vice versa is supported.

Until now we have tacitly assumed all scales and zero points to be compile time
constants to ease legalization of this operators by allowing to fuse the scales
and simplify away null zero points. To support:
  * dynamic quantization
it is necessary to relax this requirement for the input quantization parameters.

Future work involve supporting inputs and zero point in floating point formats
to allow for:
  * weight-only quantization (setting the input and output scale to one and the
    zero point to null)
  * fake quantization

Other approaches include the MLIR quant dialect [1]. It has support for
per-block quantization but it cannot support dynamic quantization being a at
type level.

[0]: https://onnx.ai/onnx/operators/onnx__QLinearConv.html#outputs
[1]: https://mlir.llvm.org/docs/Dialects/QuantDialect
"""

# NOTE: to do shape inference we use relax.BlockBuilder.normalize. This has the
# advantage of having the same behavior as TVM (by definition) but a bit
# inefficient. This method is made even a bit more inefficient by the fact that
# we have to do a weird dance of calling relax.BlockBuilder.normalize multiple
# times if the expression in not in ANF (A normal form) because
# relax.BlockBuilder.normalize emits a spurious operation in the Relax function.

# TODO: all tensors should have the same vdevice

from tvm import ir, relax, tirx

from typing import Optional, Tuple, Union

def is_int(dtype: str) -> bool: return dtype.startswith("int") or dtype.startswith("uint")
def is_float(dtype: str) -> bool: return dtype.startswith("float")
def get_tensors_sinfo(args):
    sinfo = []
    for i, arg in enumerate(args):
        if not isinstance(arg.struct_info, relax.TensorStructInfo):
            raise ValueError(f"Argument {i} must be a Tensor.")
        sinfo.append(arg.struct_info)
    return sinfo

def same_shape(s: relax.struct_info.TensorStructInfo, zp: relax.struct_info.TensorStructInfo) -> bool:
    return s.ndim == zp.ndim and (
        s.shape is None and zp.shape is None
        or s.shape.values == zp.shape.values
    )

def check_divisible(
    data: relax.struct_info.TensorStructInfo,
    qparam1: relax.struct_info.TensorStructInfo,
    qparam2: relax.struct_info.TensorStructInfo,
) -> bool:
    for qparam in (qparam1, qparam2):
        qparam_scalar = isinstance(qparam.shape, relax.ShapeExpr) and not qparam.shape.values
        if qparam_scalar:
            continue

        if data.ndim == -1 or qparam.ndim == -1:
            continue

        if data.ndim != qparam.ndim:
            return False

        if not isinstance(data.shape, relax.ShapeExpr) or not isinstance(qparam.shape, relax.ShapeExpr):
            continue

        for data_dim, qparam_dim in zip(data.shape.values, qparam.shape.values):
            if isinstance(data_dim, tirx.IntImm) and isinstance(qparam_dim, tirx.IntImm):
                q_val = int(qparam_dim)

                if int(data_dim) % q_val != 0:
                    return False

            # If either is symbolic (tir.Var), we trust it and continue. We could
            # try something like this
            # cond = tir.truncmod(data_dim, qparam_dim) == 0
            # analyzer.can_prove(cond)
            # or introduce something like
            # builder.emit(relax.op.assert_op(cond))

    return True

def infer_struct_info_qnn_add_op(call: relax.Call, ctx: relax.block_builder.BlockBuilder) -> relax.struct_info.StructInfo:
    if len(call.args) != 8:
        raise ValueError("relax.qnn.add expects exactly 8 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    a_sinfo = sinfo[0]
    a_scale_sinfo = sinfo[1]
    a_zp_sinfo = sinfo[2]
    b_sinfo = sinfo[3]
    b_scale_sinfo = sinfo[4]
    b_zp_sinfo = sinfo[5]
    c_scale_sinfo = sinfo[6]
    c_zp_sinfo = sinfo[7]

    if not (is_float(a_scale_sinfo.dtype) and
            is_float(b_scale_sinfo.dtype) and
            is_float(c_scale_sinfo.dtype)):
        raise ValueError("All scales must be float tensors.")

    if not (is_int(a_sinfo.dtype) and is_int(b_sinfo.dtype)):
        raise ValueError("All input addend must be integer tensors.")

    if not (is_int(a_zp_sinfo.dtype) and
            is_int(b_zp_sinfo.dtype) and
            is_int(c_zp_sinfo.dtype)):
        raise ValueError("All zero points must be integer tensors.")

    def get_broadcast_shape(shape1: relax.expr.ShapeExpr, shape2: relax.expr.ShapeExpr) -> relax.TensorStructInfo:
       if shape1 is None or shape2 is None:
           return None

       dummy_struct_info1 = relax.TensorStructInfo(shape1.values, dtype="float32")
       dummy_struct_info2 = relax.TensorStructInfo(shape2.values, dtype="float32")
       dummy_var1 = relax.Var("tmp1", dummy_struct_info1)
       dummy_var2 = relax.Var("tmp2", dummy_struct_info2)
       dummy_add = relax.op.add(dummy_var1, dummy_var2) # any element-wise binary op

       normalized = ctx.normalize(dummy_add)

       return normalized.struct_info

    # We avoid checking the quantization parameters because we would have to
    # broadcast them to the correct dimension and their dimension check is done
    # later in check_divisible.

    # (a_scale * (a - a_zero_point) + b_scale * (b - b_zero_point))/c_scale + c_zero_point
    out_sinfo1 = a_sinfo
    # out_sinfo1 = get_broadcast_shape(a_sinfo.shape, a_zp_sinfo.shape)
    # out_sinfo1 = get_broadcast_shape(out_sinfo1.shape, a_scale_sinfo.shape) if out_sinfo1 is not None else None
    out_sinfo2 = b_sinfo
    # out_sinfo2 = get_broadcast_shape(b_sinfo.shape, b_zp_sinfo.shape)
    # out_sinfo2 = get_broadcast_shape(out_sinfo2.shape, b_scale_sinfo.shape) if out_sinfo2 is not None else None
    out_sinfo = get_broadcast_shape(out_sinfo1.shape, out_sinfo2.shape)  if out_sinfo1 is not None and out_sinfo2 is not None else None
    # out_sinfo = get_broadcast_shape(out_sinfo.shape, c_scale_sinfo.shape) if out_sinfo is not None else None
    # out_sinfo = get_broadcast_shape(out_sinfo.shape, c_zp_sinfo.shape) if out_sinfo is not None else None
    out_shape = out_sinfo.shape if out_sinfo is not None else None

    if not (check_divisible(a_sinfo, a_scale_sinfo, a_zp_sinfo) and
            check_divisible(b_sinfo, b_scale_sinfo, b_zp_sinfo) and
            check_divisible(out_sinfo, c_scale_sinfo, c_zp_sinfo) if out_sinfo is not None else True):
        raise ValueError("All dimensions of the quantization parameters should "
            "divide the ones of the respective quantized tensor.")

    out_dtype = a_sinfo.dtype
    out_vdevice = a_sinfo.vdevice
    # Here we use the primary addends since the output should only depend on
    # their rank i.e. a has ndim=4 while a_zp has ndim=-1 make the whole
    # expression have ndim=-1
    out_ndim = max(a_sinfo.ndim, b_sinfo.ndim) \
            if out_shape is None and a_sinfo.ndim >= 0 and b_sinfo.ndim >= 0 \
            else -1
    # out_ndim = max(s.ndim for s in sinfo) \
    #     if out_shape is None and all(s.ndim >= 0 for s in sinfo) \
    #     else -1

    return relax.TensorStructInfo(out_shape, out_dtype, out_vdevice, out_ndim)

ir.register_op_attr("relax.qnn.add", "FPurity", True)
ir.register_op_attr("relax.qnn.add", "FInferStructInfo", infer_struct_info_qnn_add_op)
qnn_add_op = ir.Op.get("relax.qnn.add")
qnn_add_op.set_num_inputs(8)
qnn_add_op.add_argument("a", "Tensor", "LHS addend.")
qnn_add_op.add_argument("a_scale", "Tensor", "Scale of the LHS addend.")
qnn_add_op.add_argument("a_zero_point", "Tensor", "Zero point of the LHS addend.")
qnn_add_op.add_argument("b", "Tensor", "RHS addend.")
qnn_add_op.add_argument("b_scale", "Tensor", "Scale of the RHS addend.")
qnn_add_op.add_argument("b_zero_point", "Tensor", "Zero point of the RHS addend.")
qnn_add_op.add_argument("c_scale", "Tensor", "Scale of the result.")
qnn_add_op.add_argument("c_zero_point", "Tensor", "Zero point of the result.")

def add(
    a: relax.Expr, s_a: relax.Expr, z_a: relax.Expr,
    b: relax.Expr, s_b: relax.Expr, z_b: relax.Expr,
    s_c: relax.Expr, z_c: relax.Expr,
) -> relax.Call:
    op = ir.Op.get("relax.qnn.add")
    args = (a, s_a, z_a, b, s_b, z_b, s_c, z_c)
    return relax.Call(op, args)

def conv2d_attrs_to_dict(attrs: ir.DictAttrs) -> dict:
    strides = [int(s) for s in attrs["strides"]]
    padding = [int(p) for p in attrs["padding"]]
    dilation = [int(d) for d in attrs["dilation"]]
    groups = int(attrs["groups"])
    data_layout = str(attrs["data_layout"])
    kernel_layout = str(attrs["kernel_layout"])
    out_layout = attrs["out_layout"]
    out_layout = str(out_layout) if out_layout else data_layout
    out_dtype = "void"
    res = {
        "strides": strides,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "data_layout": data_layout,
        "kernel_layout": kernel_layout,
        "out_layout": out_layout,
        "out_dtype": out_dtype,
    }
    return res

def infer_struct_info_qnn_conv2d_op(call: relax.Call, ctx: relax.block_builder.BlockBuilder) -> relax.struct_info.StructInfo:
    if len(call.args) not in (8, 9):
        raise ValueError("relax.qnn.conv2d expects either 8 or 9 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    w_sinfo = sinfo[3]
    w_scale_sinfo = sinfo[4]
    w_zp_sinfo = sinfo[5]
    y_scale_sinfo = sinfo[6]
    y_zp_sinfo = sinfo[7]
    b_sinfo = sinfo[8] if len(call.args) == 9 else None

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(w_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
        raise ValueError("All scales must be float tensors.")

    if not (is_int(x_sinfo.dtype) and is_int(w_sinfo.dtype)):
        raise ValueError("Input and weight must be integer tensors.")

    if b_sinfo and not is_int(b_sinfo.dtype):
        raise ValueError("Bias must be an integer tensors.")

    if not (is_int(x_zp_sinfo.dtype) and
            is_int(w_zp_sinfo.dtype) and
            is_int(y_zp_sinfo.dtype)):
        raise ValueError("All zero points must be integer tensors.")

    out_shape = None
    if x_sinfo.shape is not None and w_sinfo.shape is not None:
        dummy_x = relax.Var("tmp_x", relax.TensorStructInfo(x_sinfo.shape, dtype="float32"))
        dummy_w = relax.Var("tmp_w", relax.TensorStructInfo(w_sinfo.shape, dtype="float32"))

        attrs = conv2d_attrs_to_dict(call.attrs)
        dummy_conv = relax.op.nn.conv2d(dummy_x, dummy_w, **attrs)
        normalized = ctx.normalize(dummy_conv)

        if b_sinfo is not None:
            dummy_xw = relax.Var("tmp_xw", relax.TensorStructInfo(normalized.struct_info.shape, dtype="float32"))
            dummy_b = relax.Var("tmp_b", relax.TensorStructInfo(b_sinfo.shape, dtype="float32"))
            dummy_add = relax.op.add(dummy_xw, dummy_b)
            normalized = ctx.normalize(dummy_add)

        out_shape = normalized.struct_info.shape

    if not (check_divisible(x_sinfo, x_scale_sinfo, x_zp_sinfo) and
            check_divisible(w_sinfo, w_scale_sinfo, w_zp_sinfo) and
            check_divisible(out_shape, y_scale_sinfo, y_zp_sinfo) if out_shape is not None else True):
        raise ValueError("All dimensions of the quantization parameters should "
            "divide the ones of the respective quantized tensor.")

    out_dtype = y_zp_sinfo.dtype
    out_vdevice = x_sinfo.vdevice
    out_ndim = x_sinfo.ndim if out_shape is None and x_sinfo.ndim >= 0 else -1

    return relax.TensorStructInfo(out_shape, out_dtype, out_vdevice, out_ndim)

ir.register_op_attr("relax.qnn.conv2d", "FPurity", True)
ir.register_op_attr("relax.qnn.conv2d", "FInferStructInfo", infer_struct_info_qnn_conv2d_op)

qnn_conv2d_op = ir.Op.get("relax.qnn.conv2d")
qnn_conv2d_op.set_num_inputs(9)
qnn_conv2d_op.add_argument("x", "Tensor", "Input tensor.")
qnn_conv2d_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_conv2d_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_conv2d_op.add_argument("w", "Tensor", "Weight tensor.")
qnn_conv2d_op.add_argument("w_scale", "Tensor", "Scale of the weight.")
qnn_conv2d_op.add_argument("w_zero_point", "Tensor", "Zero point of the weight.")
qnn_conv2d_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_conv2d_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")
qnn_conv2d_op.add_argument("B", "Optional[Tensor]", "Optional bias tensor.")

def conv2d(
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    w: relax.Expr, w_scale: relax.Expr, w_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    B: relax.Expr | None = None,
    strides: Union[int, Tuple[int, int]] = (1, 1),
    padding: Union[int, Tuple[int, ...]] = (0, 0),
    dilation: Union[int, Tuple[int, int]] = (1, 1),
    groups: int = 1,
    data_layout: str = "NCHW",
    kernel_layout: str = "OIHW",
    out_layout: str | None = None,
) -> relax.Call:

    op = ir.Op.get("relax.qnn.conv2d")
    args = [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point]
    if B is not None:
        args.append(B)

    attrs = ir.make_node(
        "ir.DictAttrs",
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        data_layout=data_layout,
        kernel_layout=kernel_layout,
        out_layout=out_layout,
    )

    return relax.Call(op, args, attrs)

def avg_pool2d_attrs_to_dict(attrs: ir.DictAttrs) -> dict:
    pool_size = [int(s) for s in attrs["pool_size"]]
    strides = [int(s) for s in attrs["strides"]]
    padding = [int(p) for p in attrs["padding"]]
    dilation = [int(d) for d in attrs["dilation"]]
    ceil_mode = int(attrs["ceil_mode"])
    count_include_pad = int(attrs["count_include_pad"])
    layout = str(attrs["layout"])
    out_layout = attrs["out_layout"]
    out_layout = str(out_layout) if out_layout else layout
    res = {
        "pool_size": pool_size,
        "strides": strides,
        "padding": padding,
        "dilation": dilation,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
        "layout": layout,
        "out_layout": out_layout,
    }
    return res

def infer_struct_info_qnn_avg_pool2d_op(call: relax.Call, ctx: relax.block_builder.BlockBuilder) -> relax.struct_info.StructInfo:
    if len(call.args) != 5:
        raise ValueError("relax.qnn.avg_pool2d expects 5 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    y_scale_sinfo = sinfo[3]
    y_zp_sinfo = sinfo[4]

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
        raise ValueError("All scales must be float tensors.")

    if not is_int(x_sinfo.dtype):
        raise ValueError("The input must be integer tensors.")

    if not (is_int(x_zp_sinfo.dtype) and
            is_int(y_zp_sinfo.dtype)):
        raise ValueError("All zero points must be integer tensors.")

    out_shape = None
    if x_sinfo.shape is not None:
        dummy_x_sinfo = relax.TensorStructInfo(x_sinfo.shape, dtype="float32")
        dummy_x = relax.Var("tmp_x", dummy_x_sinfo)

        attrs = avg_pool2d_attrs_to_dict(call.attrs)
        dummy_pool = relax.op.nn.avg_pool2d(dummy_x, **attrs)
        normalized = ctx.normalize(dummy_pool)
        out_shape = normalized.struct_info.shape

    if not (check_divisible(x_sinfo, x_scale_sinfo, x_zp_sinfo) and
            check_divisible(out_shape, y_scale_sinfo, y_zp_sinfo) if out_shape is not None else True):
        raise ValueError("All dimensions of the quantization parameters should "
            "divide the ones of the respective quantized tensor.")

    out_dtype = x_sinfo.dtype
    out_vdevice = x_sinfo.vdevice
    out_ndim = x_sinfo.ndim if out_shape is None and x_sinfo.ndim >= 0 else -1

    return relax.TensorStructInfo(out_shape, out_dtype, out_vdevice, out_ndim)

ir.register_op_attr("relax.qnn.avg_pool2d", "FPurity", True)
ir.register_op_attr("relax.qnn.avg_pool2d", "FInferStructInfo", infer_struct_info_qnn_avg_pool2d_op)

qnn_avg_pool2d_op = ir.Op.get("relax.qnn.avg_pool2d")
qnn_avg_pool2d_op.set_num_inputs(5)
qnn_avg_pool2d_op.add_argument("x", "Tensor", "Input tensor.")
qnn_avg_pool2d_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_avg_pool2d_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_avg_pool2d_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_avg_pool2d_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")

def avg_pool2d(
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    pool_size: int | tuple[int, int] = (1, 1),
    strides: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, ...] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    ceil_mode: bool = False,
    count_include_pad: bool = False,
    layout: str = 'NCHW',
    out_layout: str | None = None,
) -> relax.Call:
    op = ir.Op.get("relax.qnn.avg_pool2d")
    args = (x, x_scale, x_zero_point, y_scale, y_zero_point)
    attrs = ir.make_node(
        "ir.DictAttrs",
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        layout=layout,
        out_layout=out_layout,
    )
    return relax.Call(op, args, attrs)

def infer_struct_info_qnn_linear_op(call: relax.Call, ctx: relax.block_builder.BlockBuilder) -> relax.struct_info.StructInfo:
    if len(call.args) not in (8, 9):
        raise ValueError("relax.qnn.linear expects either 8 or 9 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    w_sinfo = sinfo[3]
    w_scale_sinfo = sinfo[4]
    w_zp_sinfo = sinfo[5]
    y_scale_sinfo = sinfo[6]
    y_zp_sinfo = sinfo[7]
    b_sinfo = sinfo[8] if len(call.args) == 9 else None

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(w_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
        raise ValueError("All scales must be float tensors.")

    if not (is_int(x_sinfo.dtype) and is_int(w_sinfo.dtype)):
        raise ValueError("Input and weight must be integer tensors.")

    if b_sinfo and not is_int(b_sinfo.dtype):
        raise ValueError("Bias must be an integer tensors.")

    if not (is_int(x_zp_sinfo.dtype) and
            is_int(w_zp_sinfo.dtype) and
            is_int(y_zp_sinfo.dtype)):
        raise ValueError("All input zero points must be integer tensors.")

    out_shape = None
    if x_sinfo.shape is not None and w_sinfo.shape is not None:
        dummy_x = relax.Var("tmp_x", relax.TensorStructInfo(x_sinfo.shape, dtype="float32"))
        dummy_w = relax.Var("tmp_w", relax.TensorStructInfo(w_sinfo.shape, dtype="float32"))

        # relax.on.linear expects x to be transposed...
        attrs = ir.make_node("relax.attrs.MatmulAttrs", out_dtype="void")
        dummy_linear = relax.Call(ir.Op.get("relax.matmul"), (dummy_x, dummy_w), attrs)
        normalized = ctx.normalize(dummy_linear)

        if b_sinfo is not None:
            dummy_xw = relax.Var("tmp_xw", relax.TensorStructInfo(normalized.struct_info.shape, dtype="float32"))
            dummy_b = relax.Var("tmp_b", relax.TensorStructInfo(b_sinfo.shape, dtype="float32"))
            dummy_add = relax.op.add(dummy_xw, dummy_b)
            normalized = ctx.normalize(dummy_add)

        out_shape = normalized.struct_info.shape

    if not (check_divisible(x_sinfo, x_scale_sinfo, x_zp_sinfo) and
            check_divisible(w_sinfo, w_scale_sinfo, w_zp_sinfo) and
            check_divisible(out_shape, y_scale_sinfo, y_zp_sinfo) if out_shape is not None else True):
        raise ValueError("All dimensions of the quantization parameters should "
            "divide the ones of the respective quantized tensor.")

    out_dtype = y_zp_sinfo.dtype
    out_vdevice = x_sinfo.vdevice
    out_ndim = x_sinfo.ndim if out_shape is None and x_sinfo.ndim >= 0 else -1

    return relax.TensorStructInfo(out_shape, out_dtype, out_vdevice, out_ndim)

ir.register_op_attr("relax.qnn.linear", "FPurity", True)
ir.register_op_attr("relax.qnn.linear", "FInferStructInfo", infer_struct_info_qnn_linear_op)

# beta is not present between the parameters because we expect it to be constant
# folded in B.
qnn_linear_op = ir.Op.get("relax.qnn.linear")
qnn_linear_op.set_num_inputs(9)
qnn_linear_op.add_argument("x", "Tensor", "Input tensor.")
qnn_linear_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_linear_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_linear_op.add_argument("w", "Tensor", "Weight tensor.")
qnn_linear_op.add_argument("w_scale", "Tensor", "Scale of the weight.")
qnn_linear_op.add_argument("w_zero_point", "Tensor", "Zero point of the weight.")
qnn_linear_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_linear_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")
qnn_linear_op.add_argument("B", "Optional[Tensor]", "Optional bias tensor.")

def linear(
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    w: relax.Expr, w_scale: relax.Expr, w_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    B: relax.Expr | None = None,
) -> relax.Call:
    op = ir.Op.get("relax.qnn.linear")
    args = [x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point]
    if B is not None:
        args.append(B)
    return relax.Call(op, tuple(args))

def softmax_attrs_to_dict(attrs: ir.DictAttrs) -> dict:
    axis = attrs["axis"]
    res = {
        "axis": axis,
    }
    return res

def infer_struct_info_qnn_softmax_op(call: relax.Call, ctx: relax.block_builder.BlockBuilder) -> relax.struct_info.StructInfo:
    if len(call.args) != 5:
        raise ValueError("relax.qnn.softmax expects 6 arguments.")

    sinfo = get_tensors_sinfo(call.args)

    x_sinfo = sinfo[0]
    x_scale_sinfo = sinfo[1]
    x_zp_sinfo = sinfo[2]
    y_scale_sinfo = sinfo[3]
    y_zp_sinfo = sinfo[4]

    if not (is_float(x_scale_sinfo.dtype) and
            is_float(y_scale_sinfo.dtype)):
        raise ValueError("All scales must be float tensors.")

    if not (is_int(x_sinfo.dtype)):
        raise ValueError("Input must be an integer tensor.")

    if not (is_int(x_zp_sinfo.dtype) and
            is_int(y_zp_sinfo.dtype)):
        raise ValueError("All input zero points must be integer tensors.")

    out_shape = None
    if x_sinfo.shape is not None:
        dummy_x_sinfo = relax.TensorStructInfo(x_sinfo.shape, dtype="float32")
        dummy_x = relax.Var("tmp_x", dummy_x_sinfo)

        attrs = softmax_attrs_to_dict(call.attrs)
        dummy_softmax = relax.op.nn.softmax(dummy_x, **attrs)
        normalized = ctx.normalize(dummy_softmax)
        out_shape = normalized.struct_info.shape

    if not (check_divisible(x_sinfo, x_scale_sinfo, x_zp_sinfo) and
            check_divisible(out_shape, y_scale_sinfo, y_zp_sinfo) if out_shape is not None else True):
        raise ValueError("All dimensions of the quantization parameters should "
            "divide the ones of the respective quantized tensor.")

    out_dtype = y_zp_sinfo.dtype
    out_vdevice = y_zp_sinfo.vdevice
    out_ndim = x_sinfo.ndim if out_shape is None and x_sinfo.ndim >= 0 else -1

    return relax.TensorStructInfo(out_shape, out_dtype, out_vdevice, out_ndim)

ir.register_op_attr("relax.qnn.softmax", "FPurity", True)
ir.register_op_attr("relax.qnn.softmax", "FInferStructInfo", infer_struct_info_qnn_softmax_op)

qnn_softmax_op = ir.Op.get("relax.qnn.softmax")
qnn_softmax_op.set_num_inputs(5)
qnn_softmax_op.add_argument("x", "Tensor", "Input tensor.")
qnn_softmax_op.add_argument("x_scale", "Tensor", "Scale of the input.")
qnn_softmax_op.add_argument("x_zero_point", "Tensor", "Zero point of the input.")
qnn_softmax_op.add_argument("y_scale", "Tensor", "Scale of the output.")
qnn_softmax_op.add_argument("y_zero_point", "Tensor", "Zero point of the output.")

def softmax(
    x: relax.Expr, x_scale: relax.Expr, x_zero_point: relax.Expr,
    y_scale: relax.Expr, y_zero_point: relax.Expr,
    axis: int = -1,
) -> relax.Call:
    op = ir.Op.get("relax.qnn.softmax")
    args = (x, x_scale, x_zero_point, y_scale, y_zero_point)
    attrs = ir.make_node(
        "ir.DictAttrs",
        axis=axis,
    )
    return relax.Call(op, args, attrs)

def infer_struct_info_qnn_dynamic_quantize_op(call: relax.Call, ctx: relax.block_builder.BlockBuilder) -> relax.struct_info.StructInfo:
    if len(call.args) != 1:
        raise ValueError("relax.qnn.dynamic_quantize expects exactly 1 argument.")

    sinfo = get_tensors_sinfo(call.args)
    x_sinfo = sinfo[0]

    if not is_float(x_sinfo.dtype):
        raise ValueError("The input to dynamic_quantize must be a float tensor.")

    out_dtype = call.attrs["out_dtype"]
    scale_dtype = x_sinfo.dtype
    zp_dtype = out_dtype
    vdevice = x_sinfo.vdevice

    if x_sinfo.shape is None:
        out_ndim = x_sinfo.ndim if x_sinfo.ndim >= 0 else -1
        q_sinfo = relax.TensorStructInfo(dtype=out_dtype, ndim=out_ndim, vdevice=vdevice)
    else:
        q_sinfo = relax.TensorStructInfo(x_sinfo.shape, dtype=out_dtype, vdevice=vdevice)

    axis = call.attrs["axis"]

    if axis is not None:
        if x_sinfo.ndim >= 0:
            if axis < -x_sinfo.ndim or axis >= x_sinfo.ndim:
                raise ValueError(f"axis {axis} is out of bounds for tensor of ndim {x_sinfo.ndim}")
            if axis < 0:
                axis += x_sinfo.ndim

        if x_sinfo.shape is not None and hasattr(x_sinfo.shape, "values"):
            dim = x_sinfo.shape.values[axis]
            scale_sinfo = relax.TensorStructInfo([dim], scale_dtype, vdevice)
            zp_sinfo = relax.TensorStructInfo([dim], zp_dtype, vdevice)
        else:
            scale_sinfo = relax.TensorStructInfo(None, scale_dtype, vdevice, 1)
            zp_sinfo = relax.TensorStructInfo(None, zp_dtype, vdevice, 1)
    else:
        scale_sinfo = relax.TensorStructInfo((), scale_dtype, vdevice)
        zp_sinfo = relax.TensorStructInfo((), zp_dtype, vdevice)

    return relax.TupleStructInfo([q_sinfo, scale_sinfo, zp_sinfo])

ir.register_op_attr("relax.qnn.dynamic_quantize", "FPurity", True)
ir.register_op_attr("relax.qnn.dynamic_quantize", "FInferStructInfo", infer_struct_info_qnn_dynamic_quantize_op)

qnn_dynamic_quantize_op = ir.Op.get("relax.qnn.dynamic_quantize")
qnn_dynamic_quantize_op.set_num_inputs(1)
qnn_dynamic_quantize_op.add_argument("x", "Tensor", "Input tensor.")

# TODO: a shape should be passes to allow for per-block quantization...
def dynamic_quantize(
    x: relax.Expr,
    axis: Optional[int] = None,
    out_dtype: str = "int8",
) -> relax.Call:
    """Calculates the scale and zero point for the input data x. out_dtype is
    used for both the dtype of the output and the zero_point. The scale has the
    same dtype as the input. If an axis is provided per-axis quantization is
    performed.
    """
    op = ir.Op.get("relax.qnn.dynamic_quantize")
    args = (x,)
    attrs = ir.make_node(
        "ir.DictAttrs",
        axis=axis,
        out_dtype=out_dtype,
    )
    return relax.Call(op, args, attrs)
