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

# os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --show_progress_bar=true --use_ipu_model'

from tensorflow.python import ipu

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


def run_model(model_path, tag=None, bs=1, kvparser_idx=None):
    kvparser_idx = kvparser_idx or []
    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )

    with tf.Session(graph=tf.Graph()) as sess:
        meta = tf.saved_model.loader.load(sess, [tag] if tag is not None else tf.saved_model.tag_constants.SERVING, model_path)
        output_op_names = [ f"TensorDict/StandardKvParser:{i}" for i in kvparser_idx ]
        # output_op_names = [ i.name for i in meta.signature_def["serving_default"].outputs.values()]
        out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in output_op_names]
        data_npz_dict = defaultdict(list)
        for d in gen_data(model_path, bs):
            o = sess.run(out_names_pl, feed_dict={sess.graph.get_tensor_by_name("TensorDict/batch:0"): (d,)})
            for i, j in enumerate(kvparser_idx):
                data_npz_dict[f"tensordict_standardkvparser_{j}_args"].append(o[i])
        np.savez("raw_data_wo_kv_load.npz", **{ k: np.concatenate(v) for k, v in data_npz_dict.items()})
    return o

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("the kvparser data extraction")
    parser.add_argument("-m", "--model", dest="model_path", required=True, type=str, action='store')
    parser.add_argument("-bs", "--batch-size", dest="batch_size", type=int, default=1, action='store')
    parser.add_argument("-kidx", "--kvparser-idx", dest="kvparser_idx", type=int, default=1, action='store',
                        help="the wanted output index of kvparser", default=[0], type=int, nargs='+')

    args = parser.parse_args()
    standard_output = run_model(args.model_path, tag="serve", bs=args.batch_size, kvparser_idx=args.kvparser)