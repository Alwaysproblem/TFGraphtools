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
        output_op_names = [ i.name for i in meta.signature_def["serving_default"].outputs.values()]
        out_names_pl = [ sess.graph.get_tensor_by_name(o_name) for o_name in output_op_names]

        durations = []

        for d in gen_data(model_path, bs):
            start = time.time()
            o = sess.run(out_names_pl, feed_dict={sess.graph.get_tensor_by_name("TensorDict/batch:0"): (d,)})
            duration = time.time() - start
            durations.append(duration)


        print("Average statistics excluding the first half iterations.")
        print("-------------------------------------------------------------------------------------------")
        durations = durations[len(durations)//2:]
        print("Average latency: {:.5f}".format(np.mean(durations)))

    # total_start = time.time()
    # for iter_count in range(args.num_iterations):
    #   start = time.time()
    #   predictions = session.run(predict_ops, feed_dict={
    #     input_ids: np.random.randint(low=0, high=21128, size=[args.batch_size, args.sequence_length]),
    #     input_mask: np.ones((args.batch_size, args.sequence_length)),
    #     token_type_ids: np.zeros((args.batch_size, args.sequence_length)),
    #   })
    # total_duration = time.time() - total_start

    # print("Throughput at bs={}, mode={}, num_ipus={}: {} samples/sec.".format(args.batch_size, args.mode, args.num_ipus, args.batch_size * args.num_iterations/total_duration))

    return o

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    standard_output = run_model(model_path, tag="serve")
    # print(gen_data(".", bs=4))