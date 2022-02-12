import os
from statistics import mean
import time
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_core.python.ipu as ipu
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import ops
from tensorflow.python.ipu import ipu_compiler, loops, ipu_infeed_queue, ipu_outfeed_queue, scopes
from tensorflow.compiler.plugin.poplar.ops import gen_application_runtime
from tensorflow.python.ipu.ops import application_compile_op
# import matplotlib.pyplot as plt

from threading import Thread
from queue import Queue

np.random.seed(1991)

PIPELINE_STAGE = 2
NUM_ITER = 100

input_table = {
    "LookupPkOp:0": ((425,), np.int32),
    "all_clk_seq_1/st:0": ((512, 1), np.float32),
    "all_clk_seq_1/time:0": ((512, 1), np.float32),
    # "batch_fill_attributes_for_gul_rank_item_feature:0": ((425,), np.int32),
    # "batch_fill_attributes_for_gul_rank_item_feature_1:0": ((425,), np.int32),
    "Unique:1": ((425,), np.int32),
    "embedding/item_cate_id_d_shared_embedding_2:0": ((425, 32), np.float32),
    "embedding/item_id_d_shared_embedding_2:0": ((425, 64), np.float32),
    "embedding/item_seller_id_d_shared_embedding_2:0": ((425, 32), np.float32),
    "embedding/ui_page_shared_embedding:0": ((1, 8), np.float32),
    "input_from_feature_columns/concat:0": ((1, 592), np.float32),
    "input_from_feature_columns/concat_1:0": ((425, 824), np.float32),
    "input_from_feature_columns/concat_3:0": ((1, 1), np.float32),
    "input_from_feature_columns/concat_4:0": ((425, 288), np.float32),
    "input_from_feature_columns/concat_5:0": ((1, 1), np.float32),
    "input_from_feature_columns/concat_7:0": ((178, 1), np.float32),
    "seq_input_from_feature_columns/concat:0": ((1, 48, 324), np.float32),
    "seq_input_from_feature_columns/concat_1:0": ((1, 512, 144), np.float32),
    "seq_input_from_feature_columns/concat_2:0": ((178, 48, 136), np.float32),
}

def load_tf_graph(frozen_graph_filename, with_meta_graph = True,
                   tag = tf.saved_model.tag_constants.SERVING):
    if with_meta_graph:
        if not os.path.isdir(frozen_graph_filename):
            model_path = os.path.dirname(frozen_graph_filename)
        else:
            model_path = frozen_graph_filename

        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph = tf.saved_model.loader.load(sess, [tag] if tag else [], model_path)
            graph = tf.get_default_graph()
            return graph, meta_graph

def gen_data(bs, graph):
    data_dict = np.load("data.npz")
    feed_dict = { graph.get_tensor_by_name(i if i != "Unique:1" else "Unique:0"): data_dict[i] for i in input_table }
    return feed_dict

def run_tput(model_path, bs=1, pipline_stages=PIPELINE_STAGE, iterations=NUM_ITER):

    graph, meta = load_tf_graph(model_path)

#%%
    # inp_names= sorted([ i.name for i in meta.signature_def["serving_default"].inputs.values()])
    out_names= [ i.name for i in meta.signature_def["serving_default"].outputs.values()]

    out_names_pl = [ graph.get_tensor_by_name(o_name) for o_name in out_names]

    sess_cfg = tf.ConfigProto()
    # sess_cfg.log_device_placement = True
    sess_cfg.graph_options.rewrite_options.memory_optimization = (
    rewriter_config_pb2.RewriterConfig.OFF
    )

    q = Queue()

    with tf.Session(graph=graph, config=sess_cfg) as session:

        feed_dict = gen_data(bs, graph)
        _ = session.run(out_names_pl, feed_dict=feed_dict)

        def runner(bs, graph, feed_dict, session, sleep_time = 0.5):
            durations = []
            # feed_dict = gen_data(bs, graph)
            for _ in range(iterations):
                start = time.time()
                results = session.run(out_names_pl, feed_dict=feed_dict)
                stop = time.time()
                durations.append((start, stop, ))

            q.put(durations, timeout=10)

        # sleep_times = [0.01] * pipline_stages
        sleep_times = [0.1, 0.158, 0.02, 0.05] * pipline_stages
        thp = [ Thread(target=runner, args=(bs, graph, feed_dict, session, sleep_times[i])) for i in range(pipline_stages)]

        s = time.time()
        for idx, th in enumerate(thp):
            th.start()
            print(f"Thread {idx} started.")

        for idx, th in enumerate(thp):
            th.join()
            print(f"Thread {idx} join.")

        durations_from_th = []
        while not q.empty():
            durations_from_th += q.get()

        latency_list = np.array([ y - x for x, y in durations_from_th])
        latency = np.mean(latency_list)
        mid_latency = np.percentile(latency_list, 50)
        P75_latency = np.percentile(latency_list, 75)
        P90_latency = np.percentile(latency_list, 90)
        P99_latency = np.percentile(latency_list, 99)
        (_, idx, counts) = np.unique(latency_list, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_latency = latency_list[index]
        min_start = min([ x for x, _ in durations_from_th])
        max_stop = max([ y for _, y in durations_from_th])
        # tput = bs * pipline_stages / latency
        tput = bs * pipline_stages * iterations / (max_stop - min_start)

        # test_results.append([bs, latency, tput])

    # bs, latency, tput = test_results[0]

    print(f"batch size: {bs}\n"
          f"latency: {latency}\n"
          f"throughput: {tput}\n"
          f"mid: {mid_latency}\n"
          f"P75: {P75_latency}\n"
          f"P90: {P90_latency}\n"
          f"P99: {P99_latency}\n"
          f"mode: {mode_latency}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("the kvparser data extraction")
    parser.add_argument("-m", "--model", dest="model_path", required=True, type=str, action='store')
    parser.add_argument("-bs", "--batch-size", dest="batch_size", type=int, default=1, action='store')
    parser.add_argument("-ps", "--pipline-stages", dest="pipline_stages", type=int, default=2, action='store')
    parser.add_argument("-i", "--iterations", dest="iterations", type=int, default=100, action='store')

    args = parser.parse_args()
    run_tput(args.model_path, args.batch_size, pipline_stages=args.pipline_stages, iterations=args.iterations)