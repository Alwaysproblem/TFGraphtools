import os
from statistics import mean
import time
import numpy as np
import libpvti as pvti
import tensorflow.compat.v1 as tf
# import tensorflow_core.python.ipu as ipu
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
from collections import defaultdict
# from tensorflow.python.ipu import ipu_compiler, loops, ipu_infeed_queue, ipu_outfeed_queue, scopes
# from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
# from tensorflow.python.ipu.ops import application_compile_op
# import yaml

os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --show_progress_bar=true --use_ipu_model'

from tensorflow.python import ipu
from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.python.ipu import utils
import sys

TF_2_NP = {
  tf.float32: np.float32,
  tf.int32: np.int32,
  tf.float16: np.float16,
  tf.int64: np.int64,
  tf.bool: np.bool
}


# Builds ipu_options
if os.path.exists(f"{sys.argv[1]}/ipu_cfg.bin"):
    with open(f"{sys.argv[1]}/ipu_cfg.bin", 'rb') as f:
        _ipu_cfg = config_pb2.IpuOptions()
        _ipu_cfg.ParseFromString(f.read())
    utils.configure_ipu_system(_ipu_cfg)
else:
# Builds ipu_options
    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1

    cfg.configure_ipu_system()
np.random.seed(1991)

ops.disable_eager_execution()
tf.disable_v2_behavior()


def run_model(model_path, tag = None):
    bs = 1

    # sess_cfg = tf.ConfigProto()
    # # sess_cfg.log_device_placement = True
    # sess_cfg.graph_options.rewrite_options.memory_optimization = (
    #     rewriter_config_pb2.RewriterConfig.OFF
    # )
    channel = pvti.createTraceChannel("Custom channel")
    with tf.Session(graph=tf.Graph()) as sess:
        meta = tf.saved_model.loader.load(sess, [tag] if tag is not None else tf.saved_model.tag_constants.SERVING, model_path)
        # output_op_names = [ f"TensorDict/StandardKvParser:{i}" for i in range(4) ]
        input_op_names = [ i.name for i in meta.signature_def["serving_default"].inputs.values()]
        # print(input_op_names)
        input_op_names.remove("all_clk_seq_1/st:0")
        input_op_names.remove("all_clk_seq_1/time:0")
        input_op_names.remove("batch_fill_attributes_for_gul_rank_item_feature:0")
        input_op_names.remove("batch_fill_attributes_for_gul_rank_item_feature_1:0")
        input_op_names.remove("LookupPkOp:0")
        output_op_names = [ i.name for i in meta.signature_def["serving_default"].outputs.values()]
        input_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in input_op_names]
        out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in output_op_names]
        inputs_name_shape_dtype = [(pl, [bs,] + pl.get_shape().as_list()[1:], pl.dtype) for pl in input_names_pl]
        print(inputs_name_shape_dtype)

        feed_dict = {
            it: np.random.randint(low=0, high=100, size=ishape).astype(TF_2_NP[idtype])
            for it, ishape, idtype in inputs_name_shape_dtype
        }
        feed_dict.update({
            sess.graph.get_tensor_by_name('all_clk_seq_1/st:0'): np.random.randint(
                low=0, high=100, size=[512 * bs, 1]).astype(TF_2_NP[tf.float32])})
        feed_dict.update({
            sess.graph.get_tensor_by_name('all_clk_seq_1/time:0'): np.random.randint(
                low=0, high=100, size=[512 * bs, 1]).astype(TF_2_NP[tf.float32])})
        feed_dict.update({
            sess.graph.get_tensor_by_name(
                "batch_fill_attributes_for_gul_rank_item_feature:0"): 
                np.ones((bs,)).astype(TF_2_NP[tf.int32])})
        feed_dict.update({
            sess.graph.get_tensor_by_name(
                "batch_fill_attributes_for_gul_rank_item_feature_1:0"): 
                (np.ones((bs,)) * 3).astype(TF_2_NP[tf.int32])})
        feed_dict.update({
            sess.graph.get_tensor_by_name('LookupPkOp:0'): np.random.randint(
                low=0, high=100, size=[1]).astype(TF_2_NP[tf.int32])})

        # o = sess.run([
        #     out_names_pl,
        #     sess.graph.get_tensor_by_name('Bitcast:0'),
        #     sess.graph.get_tensor_by_name('Unique:0'),
        #     sess.graph.get_tensor_by_name('Unique:1')], feed_dict=feed_dict)

        # out, Bitcast, unique_0, unique_1= o
        # print(f"output:{out}")
        # print(f"Bitcast:{Bitcast}")
        # print(f"unique_0:{unique_0}")
        # print(f"unique_1:{unique_1}")

        for _ in range(10):
            with pvti.Tracepoint(channel, "session.run"):
                o = sess.run(out_names_pl, feed_dict=feed_dict)

        durs = []
        for _ in range(100):
            s = time.perf_counter()
            o = sess.run(out_names_pl, feed_dict=feed_dict)
            durs.append(time.perf_counter() - s)
        print(f"{np.array(durs).mean() * 1000}ms")
    return o


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    standard_output = run_model(model_path, tag="serve")
    # print(gen_data(".", bs=4))
    print(standard_output)