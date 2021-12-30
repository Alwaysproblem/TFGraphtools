import os
from statistics import mean
import time
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow_core.python.ipu as ipu
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
from collections import defaultdict
# from tensorflow.python.ipu import ipu_compiler, loops, ipu_infeed_queue, ipu_outfeed_queue, scopes
# from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
# from tensorflow.python.ipu.ops import application_compile_op
# import yaml

os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --executable_cache_path=cachedir --show_progress_bar=true'
# os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --executable_cache_path=/tmp/cachedir/ --show_progress_bar=true --use_ipu_model'

from tensorflow.python import ipu

from tensorflow.compiler.plugin.poplar.driver import config_pb2
from tensorflow.python.ipu import utils
import sys


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

def gen_data(model_path, bs = 1):
    with open(f"{model_path}/raw_data.dat") as dat_file:
        dat_content = dat_file.read().strip().split('[dat]')

    input_str_list = []
    for s in dat_content:
        if s:
            #s = s.replace('\n', "")
            s = '[dat]' + s
            input_str_list.append(s)

    input_strs = [ "".join(input_str_list[i: i + bs]) for i in range(0, len(dat_content), bs) if "".join(input_str_list[i: i + bs])]

    return input_strs


def run_model(model_path, tag = None):
    bs = 1

    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )
    # input_str = gen_data()

    with tf.Session(graph=tf.Graph()) as sess:
        meta = tf.saved_model.loader.load(sess, [tag] if tag is not None else tf.saved_model.tag_constants.SERVING, model_path)
        # output_op_names = [ f"TensorDict/StandardKvParser:{i}" for i in range(4) ]
        output_op_names = [ i.name for i in meta.signature_def["serving_default"].outputs.values()]
        out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in output_op_names]
        # data_npz_dict = defaultdict(list)
        for d in gen_data(model_path, bs):
            o = sess.run(out_names_pl, feed_dict={sess.graph.get_tensor_by_name("TensorDict/batch:0"): (d,)})
        #     for i in range(4):
        #         data_npz_dict[f"tensordict_standardkvparser_{i}_args"].append(o[i])
        # np.savez("raw_data_wo_kv_load.npz", **{ k: np.concatenate(v) for k, v in data_npz_dict.items()})
    return o

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    standard_output = run_model(model_path, tag="serve")
    print(standard_output)