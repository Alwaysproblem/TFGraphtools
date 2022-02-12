import tensorflow.compat.v1 as tf
from tensorflow.saved_model import tag_constants
import os
import numpy as np

import tensorflow.python.saved_model as saved_model
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.framework import graph_util


tf.disable_v2_behavior()

def build_graph(inputs):
    o = inputs

    for i in range(5):
        with tf.variable_scope(f"unit_{i}"):
            o = tf.layers.dense(inputs=o, units=2, activation=tf.nn.relu, kernel_initializer=tf.initializers.truncated_normal(
                mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32
            ))

    return o


def main():
    
    i = np.random.randint(10, size=[3, 2])
    print(i)


    # from tensorflow.core.protobuf import rewriter_config_pb2
    # sess_cfg = tf.ConfigProto()
    # sess_cfg.graph_options.rewrite_options.memory_optimization = (
    #     rewriter_config_pb2.RewriterConfig.OFF)

    with tf.Graph().as_default():
        inputs = tf.placeholder(shape=[3, 2], dtype=tf.float32)
        model = build_graph(inputs)
        init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init_op)
            # tf.summary.FileWriter("logs/", sess.graph)
            oo = sess.run(model, feed_dict={inputs: i})
            print(oo)

            constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(add_shapes=True), ["unit_4/dense/Relu"])

            with tf.gfile.GFile("save_freeze/saved_model.pb", "wb") as f:
                f.write(constant_graph.SerializeToString())

            # inputs_des = {
            #     "inputs": inputs,
            # }
            # outputs_des = {"prediction": model}
            # # tf.saved_model.simple_save(
            # #     sess, 'save/', inputs_des, outputs_des
            # # )
            # builder = saved_model.builder.SavedModelBuilder("save_builder/")

            # signature = predict_signature_def(inputs=inputs_des,
            #                                 outputs=outputs_des)
            # # using custom tag instead of: tags=[tag_constants.SERVING]
            # builder.add_meta_graph_and_variables(sess=sess,
            #                                     tags=[tag_constants.SERVING],
            #                                     signature_def_map={'predict': signature})
            # builder.save()


def p():
    return __file__

def k():
    return os.getcwd()

if __name__ == '__main__':
    
    # cfg = ipu.utils.create_ipu_config()
    # cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    # cfg = ipu.utils.auto_select_ipus(cfg, 1)
    # ipu.utils.configure_ipu_system(cfg)

    main()