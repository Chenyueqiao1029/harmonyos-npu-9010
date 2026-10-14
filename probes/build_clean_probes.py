"""Clean single-op probes, redesigned after the Conv discovery.

Why redo these. The Conv result (MatMul 549 ms -> 1x1 Conv 1 ms, both numerically
verified) proved that on this NPU the *expression form* decides success, not raw
capability. The earlier "unsupported op" verdicts came from probes that were not clean:

  u6_where   reported "Select unsupported" but the real device error was a SHAPE failure:
             SelectVerify::"the dimnum of x1 and x2 must be the same, but now is 2 1"
  u4_less / u5_and  mixed in Cast and mismatched input shapes, so nothing was isolated
  dimCnt>=4  was generalised into a global ban, disproved by d34_mul running 4-D fine

So each probe here contains ONLY the operator under test:
  - one dtype throughout (fp16), explicit axes, no helper Cast
  - constant operands only where the ONNX schema requires them
  - Where gets BOTH branches in the same shape, per the review's instruction
  - ExpandDims is tested at 3->4 and 4->5 separately, since those are different cases
"""
import os

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

DIR = "/mnt/e/npu_work/hiai-llm-plan/clean-probes"
os.makedirs(DIR, exist_ok=True)
F16, F32, I32, I64, BOOL = (TensorProto.FLOAT16, TensorProto.FLOAT,
                            TensorProto.INT32, TensorProto.INT64, TensorProto.BOOL)


def vi(name, dt, shape):
    return helper.make_tensor_value_info(name, dt, shape)


def build(name, nodes, inputs, outputs, inits=None):
    g = helper.make_graph(nodes, name, inputs, outputs, initializer=inits or [])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    p = os.path.join(DIR, name + ".onnx")
    onnx.save(m, p)
    return p


def const(name, arr):
    return numpy_helper.from_array(arr, name)


def main():
    made = []

    # p01 Reshape with a CONSTANT shape -> may be foldable by the optimiser
    made.append(build("p01_reshape_const",
        [helper.make_node("Reshape", ["X", "shp"], ["Y"])],
        [vi("X", F16, [2, 3, 4])], [vi("Y", F16, [2, 12])],
        [const("shp", np.array([2, 12], dtype=np.int64))]))

    # p02 Cast fp16 -> fp32, nothing else
    made.append(build("p02_cast_f16_f32",
        [helper.make_node("Cast", ["X"], ["Y"], to=F32)],
        [vi("X", F16, [2, 3, 4])], [vi("Y", F32, [2, 3, 4])]))

    # p03 Gather along axis 0, indices as a constant (schema-legal, single op)
    made.append(build("p03_gather_axis0",
        [helper.make_node("Gather", ["D", "idx"], ["Y"], axis=0)],
        [vi("D", F16, [8, 4])], [vi("Y", F16, [2, 4])],
        [const("idx", np.array([1, 5], dtype=np.int64))]))

    # p04 ExpandDims 3-D -> 4-D, explicit axis
    made.append(build("p04_expanddims_3to4",
        [helper.make_node("Unsqueeze", ["X", "ax"], ["Y"])],
        [vi("X", F16, [2, 3, 4])], [vi("Y", F16, [1, 2, 3, 4])],
        [const("ax", np.array([0], dtype=np.int64))]))

    # p05 ExpandDims 4-D -> 5-D: the case the device log complained about
    made.append(build("p05_expanddims_4to5",
        [helper.make_node("Unsqueeze", ["X", "ax"], ["Y"])],
        [vi("X", F16, [1, 2, 3, 4])], [vi("Y", F16, [1, 1, 2, 3, 4])],
        [const("ax", np.array([0], dtype=np.int64))]))

    # p06 Where with BOTH branches the same shape -- the review's explicit fix
    made.append(build("p06_select_same_shape",
        [helper.make_node("Where", ["c", "a", "b"], ["Y"])],
        [vi("c", BOOL, [4, 8]), vi("a", F16, [4, 8]), vi("b", F16, [4, 8])],
        [vi("Y", F16, [4, 8])]))

    # p07 / p08 Cos, Sin on plain fp16 input
    made.append(build("p07_cos",
        [helper.make_node("Cos", ["X"], ["Y"])],
        [vi("X", F16, [2, 3, 4])], [vi("Y", F16, [2, 3, 4])]))
    made.append(build("p08_sin",
        [helper.make_node("Sin", ["X"], ["Y"])],
        [vi("X", F16, [2, 3, 4])], [vi("Y", F16, [2, 3, 4])]))

    # p09 Expand (BroadcastTo) from [1,4] to [3,4], constant shape
    made.append(build("p09_broadcastto",
        [helper.make_node("Expand", ["X", "shp"], ["Y"])],
        [vi("X", F16, [1, 4])], [vi("Y", F16, [3, 4])],
        [const("shp", np.array([3, 4], dtype=np.int64))]))

    # p10 LogicalAnd, both operands the same shape (the earlier version mismatched)
    made.append(build("p10_logicaland",
        [helper.make_node("And", ["A", "B"], ["Y"])],
        [vi("A", BOOL, [4, 8]), vi("B", BOOL, [4, 8])], [vi("Y", BOOL, [4, 8])]))

    # p11 MatMul -> 1x1 Conv, the confirmed-fast form, as a control
    W = (np.arange(8 * 8).reshape(8, 8).astype(np.float32) / 64.0).astype(np.float16)
    made.append(build("p11_conv1x1_control",
        [helper.make_node("Transpose", ["X"], ["Xt"], perm=[0, 2, 1]),
         helper.make_node("Unsqueeze", ["Xt", "ax"], ["Xn"]),
         helper.make_node("Conv", ["Xn", "Wc"], ["Yn"], kernel_shape=[1, 1]),
         helper.make_node("Squeeze", ["Yn", "ax"], ["Yt"]),
         helper.make_node("Transpose", ["Yt"], ["Y"], perm=[0, 2, 1])],
        [vi("X", F16, [1, 4, 8])], [vi("Y", F16, [1, 4, 8])],
        [const("Wc", W.T.reshape(8, 8, 1, 1)), const("ax", np.array([3], dtype=np.int64))]))

    for p in made:
        print("  built", os.path.basename(p))
    print("  total", len(made))


if __name__ == "__main__":
    main()
