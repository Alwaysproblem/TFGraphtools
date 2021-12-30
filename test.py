import tensorflow.compat.v1 as tf
from tensorflow.saved_model import tag_constants
import os
import numpy as np
# from tensorflow.python.ipu import outlined_function as ipu_function
# from tensorflow.python import ipu
# from functools import wraps

tf.disable_v2_behavior()

# def function_decorator(use_ipu_function, func=None):
#     """Wrapper which uses @ipu.function when enabled, and not otherwise"""
#     def decorated(inner_func):
#         if use_ipu_function:
#             return ipu_function(inner_func)
#         return inner_func

#     if func is not None:
#         return decorated(func)
#     return decorated

def build_graph(inputs):
    # o = inputs
    def unit(inputs, scope):
        # @function_decorator(use_ipu_function=True)
        def unit_net(inputs):
            with tf.variable_scope("unit_%d" %scope):
                w = tf.get_variable("weights", shape = [2, 2], initializer=tf.ones_initializer())
                b = tf.get_variable("bias", shape = [2], initializer=tf.ones_initializer())

                o = tf.matmul(inputs, w) + b
            return o
        o = unit_net(inputs)
        # o = ipu.pipelining_ops.recomputation_checkpoint(o)
        return o

    def activations(inputs):
        # v = tf.get_variable("c", [1], initializer=tf.ones_initializer())
        return tf.nn.relu(inputs)


    with tf.variable_scope("units") as vs:
        for i in range(5):
            inputs = unit(inputs, i)
    
    with tf.variable_scope(vs,
                        auxiliary_name_scope=False) as vs1:
        with tf.name_scope(vs1.original_name_scope):
            inputs = activations(inputs)

    return inputs

def infer_loop():
    pass

def main():
    inputs = tf.placeholder(shape=[3, 2], dtype=tf.float32)
    model = build_graph(inputs)

    i = np.random.randint(10, size=[3, 2])
    print(i)

    init_op = tf.global_variables_initializer()
    # from tensorflow.core.protobuf import rewriter_config_pb2
    sess_cfg = tf.ConfigProto()
    # sess_cfg.graph_options.rewrite_options.memory_optimization = (
    #     rewriter_config_pb2.RewriterConfig.OFF)
    sess = tf.Session(config=sess_cfg)
    sess.run(init_op)
    # tf.summary.FileWriter("logs/", sess.graph, session = sess)
    oo = sess.run(model, feed_dict={inputs: i})
    print(oo)

    # Saving
    inputs_des = {
        "inputs": inputs,
    }
    outputs_des = {"prediction": model}
    tf.saved_model.simple_save(
        sess, 'save/', inputs_des, outputs_des
    )

    # with tf.Session() as sess:
    #     sess.run(init_op)
    #     tf.summary.FileWriter("logs/", sess.graph)
    #     oo = sess.run(model, feed_dict={inputs: i})
    #     print(oo)

if __name__ == '__main__':
    
    # cfg = ipu.utils.create_ipu_config()
    # cfg = ipu.utils.set_ipu_model_options(cfg, compile_ipu_code=False)
    # cfg = ipu.utils.auto_select_ipus(cfg, 1)
    # ipu.utils.configure_ipu_system(cfg)

    main()