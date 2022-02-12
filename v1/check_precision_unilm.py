#%%
import os
from statistics import mean
import time
import numpy as np
import tensorflow.compat.v1 as tf
# import tensorflow_core.python.ipu as ipu
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
# from tensorflow.python.ipu import ipu_compiler, loops, ipu_infeed_queue, ipu_outfeed_queue, scopes
# from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
# from tensorflow.python.ipu.ops import application_compile_op
import yaml

# os.environ['TF_POPLAR_FLAGS'] = '--max_compilation_threads=40 --show_progress_bar=true --use_ipu_model'

import tensorflow_core.python.ipu as ipu
# Builds ipu_options
cfg = ipu.config.IPUConfig()
cfg.auto_select_ipus = 2

cfg.configure_ipu_system()

np.random.seed(1991)

ops.disable_eager_execution()
tf.disable_v2_behavior()

def load_tf_graph(frozen_graph_filename, tag = tf.saved_model.tag_constants.SERVING):
    if not os.path.isdir(frozen_graph_filename):
        model_path = os.path.dirname(frozen_graph_filename)
    else:
        model_path = frozen_graph_filename

    with tf.Session(graph=tf.Graph()) as sess:
        meta_graph = tf.saved_model.loader.load(sess, [tag] if tag else [], model_path)
        graph = tf.get_default_graph()
        return graph, meta_graph


def gen_data():
    input_str = (
        """[dat]"""
        """input_ids=276:101,3300,671,1372,6054,4281,679,2207,2552,6649,6822,671,1366,1282,5101,3918,4638,759,7027,102,1,122,1372,6054,4281,679,2207,2552,2957"""
        """,6822,749,671,1366,1282,5101,3918,4638,3369,759,7027,7481,117,2124,4635,1921,2518,677,4260,758,5101,3241,677,4717,6230,3198,1348,3998,678,676,2"""
        """input_mask=119:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"""
        """token_type_ids=119:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"""
        """unique_id=36:8d32328c-a32d-11eb-a0c2-b8599f4d8d8a"""
    )
    return input_str


def run_model(model_path, output_op_names, tag = None):

    input_str = gen_data()
    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
        rewriter_config_pb2.RewriterConfig.OFF
    )

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, [tag] if tag is not None else tf.saved_model.tag_constants.SERVING, model_path)
        out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in output_op_names]
        o = sess.run(out_names_pl, feed_dict={sess.graph.get_tensor_by_name("TensorDict/batch:0"): (input_str,)})
    
    return o


def check_same(s, f, threshold = 0.01):

    def equals(i, j):
        if i.dtype == j.dtype:
            if i.dtype in (np.float, np.float64, np.float16, np.float32):
                return (i - j) < threshold
            else:
                return i == j

    return list(map(lambda x, y: np.all(equals(x, y)), s, f))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--config-file", action="store", type=str, default="outop.yml", dest="config_yaml")
    args = parser.parse_args()
    with open(args.config_yaml) as ff:
        config_yaml = yaml.load(ff, Loader=yaml.FullLoader)
        if (isinstance(config_yaml['CheckOutputConfig'][0]["threshold"], str) 
                and "e" in config_yaml['CheckOutputConfig'][0]["threshold"].lower()):
            config_yaml['CheckOutputConfig'][0]["threshold"] = eval(config_yaml['CheckOutputConfig'][0]["threshold"])
        if config_yaml["modelURL"]["standard"][1]["tag"].lower() == "none":
            config_yaml["modelURL"]["standard"][1]["tag"] = None
        if config_yaml["modelURL"]["needfix"][1]["tag"].lower() == "none":
            config_yaml["modelURL"]["needfix"][1]["tag"] = None

    model_stand = config_yaml["modelURL"]["standard"][0]["name"]
    model_needfix = config_yaml["modelURL"]["needfix"][0]["name"]

    model_stand_tag = config_yaml["modelURL"]["standard"][1]["tag"]
    model_needfix_tag = config_yaml["modelURL"]["needfix"][1]["tag"]

    output_op_list = config_yaml["outputNode"]

    check_threhold = config_yaml['CheckOutputConfig'][0]["threshold"]

    standard_output = run_model(model_stand, output_op_list, tag=model_stand_tag)
    needfix_output = run_model(model_needfix, output_op_list, tag=model_needfix_tag)

    print(standard_output)
    print(needfix_output)

    print(f"the check same: {check_same(standard_output, needfix_output, check_threhold)}")
