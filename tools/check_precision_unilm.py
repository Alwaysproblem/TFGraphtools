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
cfg.auto_select_ipus = 1

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
"""[dat]
query=0:马桶堵了衣服
url_pos=0:https://baijiahao.baidu.com/s?id=1590751554878831778
atitle_pos=0:马桶堵了用衣架疏通怎么找小孔^A怎么用衣架疏通马桶^A马桶堵了用衣架疏通图^A衣架通马桶^A衣架水瓶马桶^A马桶堵了可以用衣架通吗^A马桶堵塞了
title_pos=0:家里马桶堵了别急着修，一个塑料瓶加衣架就搞定，赶紧在家...
feature_pos=0:0.575354814529
url_neg=0:https://baijiahao.baidu.com/s?id=1572907682834742
atitle_neg=0:马桶被一大团卫生纸堵住^A没有扔东西马桶突然堵了^A关于客房马桶工人不能使用的警告^A马桶堵住了家政公司的人会把马桶拆开吗^A通马桶工人工资^A马桶堵崩溃的图片^A怎么把贝壳从马桶掏出^A马桶的东西都掏了还是堵^A上完卫生间公司马桶赌起来了^A马桶突然很顺畅突然就堵了什么情况^A通马桶工人能否查出原因
title_neg=0:回家发现马桶堵了，工人从里面掏出一团东西，看完后我瞬间...
feature_neg=0:-1.35283219814
input_words_query=0:101,7716,3446,1843,749,6132,3302,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
segment_ids_query=0:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
input_words_pos_title=0:101,2157,7027,7716,3446,1843,749,1166,2593,4708,934,8024,671,702,1848,3160,4486,1217,6132,3373,2218,3018,2137,8024,6628,5165,1762,2157,119,119,119,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
segment_ids_pos_title=0:0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
input_words_neg_title=0:101,1726,2157,1355,4385,7716,3446,1843,749,8024,2339,782,794,7027,7481,2959,1139,671,1730,691,6205,8024,4692,2130,1400,2769,4746,7313,119,119,119,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
segment_ids_neg_title=0:0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
input_words_pos=0:101,7716,3446,1843,749,6132,3302,102,2157,7027,7716,3446,1843,749,1166,2593,4708,934,8024,671,702,1848,3160,4486,1217,6132,3373,2218,3018,2137,8024,6628,5165,1762,2157,119,119,119,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
segment_ids_pos=0:0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
input_words_neg=0:101,7716,3446,1843,749,6132,3302,102,1726,2157,1355,4385,7716,3446,1843,749,8024,2339,782,794,7027,7481,2959,1139,671,1730,691,6205,8024,4692,2130,1400,2769,4746,7313,119,119,119,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
segment_ids_neg=0:0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
label_pos=0:1
label_neg=0:0
pos_predict=0:-1.28940212727
neg_predict=0:-4.04050159454
query_bert_id_str=0:7716,3446,1843,749,6132,3302
title_pos_bert_id_str=0:2157,7027,7716,3446,1843,749,1166,2593,4708,934,8024,671,702,1848,3160,4486,1217,6132,3373,2218,3018,2137,8024,6628,5165,1762,2157,119,119,119
title_neg_bert_id_str=0:1726,2157,1355,4385,7716,3446,1843,749,8024,2339,782,794,7027,7481,2959,1139,671,1730,691,6205,8024,4692,2130,1400,2769,4746,7313,119,119,119"""
)
    # input_str = (
    #     """[dat]"""
    #     """input_ids=276:101,3300,671,1372,6054,4281,679,2207,2552,6649,6822,671,1366,1282,5101,3918,4638,759,7027,102,1,122,1372,6054,4281,679,2207,2552,2957"""
    #     """,6822,749,671,1366,1282,5101,3918,4638,3369,759,7027,7481,117,2124,4635,1921,2518,677,4260,758,5101,3241,677,4717,6230,3198,1348,3998,678,676,2"""
    #     """input_mask=119:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"""
    #     """token_type_ids=119:0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1"""
    #     """unique_id=36:8d32328c-a32d-11eb-a0c2-b8599f4d8d8a"""
    # )
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
