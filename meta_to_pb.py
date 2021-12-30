import tensorflow as tf
import os
import shutil
import numpy as np
from tensorflow.python.saved_model import builder

batch_size = 1
max_sequence_length = 128

def meta_to_saved_model(meta_model_input_path, savedModel_path, output_node_names = ['bert/pooler/dense/Tanh']):
    meta_path = f'{meta_model_input_path}/bert_model.ckpt.meta' # Your .meta file
    saved_model_builder = builder.SavedModelBuilder(savedModel_path)
    with tf.Session() as sess:
        # Restore the graph
        saver = tf.train.import_meta_graph(meta_path)

        # Load weights
        saver.restore(sess, tf.train.latest_checkpoint('uncased_L-24_H-1024_A-16'))

        # sess.graph_def.node[0].attr["shape"].shape.dim[1].size = max_sequence_length
        # sess.graph_def.node[1].attr["shape"].shape.dim[1].size = max_sequence_length
        # sess.graph_def.node[2].attr["shape"].shape.dim[1].size = max_sequence_length
        # sess.graph_def.node[0].attr["shape"].shape.dim[0].size = -1
        # sess.graph_def.node[1].attr["shape"].shape.dim[0].size = -1
        # sess.graph_def.node[2].attr["shape"].shape.dim[0].size = -1

    #     old_graph_def = tf.get_default_graph().as_graph_def()
    #     old_graph_def.node[0].attr["shape"].shape.dim[1].size = max_sequence_length
    #     old_graph_def.node[1].attr["shape"].shape.dim[1].size = max_sequence_length
    #     old_graph_def.node[2].attr["shape"].shape.dim[1].size = max_sequence_length
    #     old_graph_def.node[0].attr["shape"].shape.dim[0].size = -1
    #     old_graph_def.node[1].attr["shape"].shape.dim[0].size = -1
    #     old_graph_def.node[2].attr["shape"].shape.dim[0].size = -1

    # with tf.Graph().as_default() as new_graph:
    #     input_map_in_use = {
    #         "Placeholder:0": tf.placeholder(tf.int32, shape=[None, max_sequence_length], name = "input_ids"),
    #         "Placeholder_1:0": tf.placeholder(tf.int32, shape=[None, max_sequence_length], name = "input_mask"),
    #         "Placeholder_2:0": tf.placeholder(tf.int32, shape=[None, max_sequence_length], name = "segment_ids"),
    #     }
    #     out = tf.import_graph_def(old_graph_def, name="", input_map=input_map_in_use, return_elements=output_node_names)
    #     print()
    #     # Freeze the graph
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph_def,
            output_node_names)

        # output_graph_def.node[0].name = "input_ids"
        # output_graph_def.node[1].name = "input_mask"
        # output_graph_def.node[2].name = "segment_ids"
        # output_graph_def.node[0].attr["shape"].shape.dim[1].size = max_sequence_length
        # output_graph_def.node[1].attr["shape"].shape.dim[1].size = max_sequence_length
        # output_graph_def.node[2].attr["shape"].shape.dim[1].size = max_sequence_length
        # output_graph_def.node[0].attr["shape"].shape.dim[0].size = -1
        # output_graph_def.node[1].attr["shape"].shape.dim[0].size = -1
        # output_graph_def.node[2].attr["shape"].shape.dim[0].size = -1



    with tf.Graph().as_default():
        # input_map_in_use = {
        #     "Placeholder:0": tf.placeholder(tf.int32, shape=[None, max_sequence_length], name = "input_ids"),
        #     "Placeholder_1:0": tf.placeholder(tf.int32, shape=[None, max_sequence_length], name = "input_mask"),
        #     "Placeholder_2:0": tf.placeholder(tf.int32, shape=[None, max_sequence_length], name = "segment_ids"),
        # }
        # out = tf.import_graph_def(output_graph_def, name="", input_map=input_map_in_use, return_elements=output_node_names)
        out = tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:

            input_ids_pl = sess.graph.get_tensor_by_name("input_ids:0")
            input_mask_pl = sess.graph.get_tensor_by_name("input_mask:0")
            segment_ids_pl = sess.graph.get_tensor_by_name("segment_ids:0")
            # out_1 = sess.graph.get_tensor_by_name("bert/encoder/Reshape_25:0")

            input_ids_pl.set_shape((None, max_sequence_length))
            input_mask_pl.set_shape((None, max_sequence_length))
            segment_ids_pl.set_shape((None, max_sequence_length))
            # out_1.set_shape((None, max_sequence_length, 1024))

            output_node_pl = [sess.graph.get_tensor_by_name(f"{t}:0") for t in output_node_names]
            o = sess.run(output_node_pl, feed_dict={
                input_ids_pl: np.random.randint(0, 10, size=(batch_size, max_sequence_length)).astype(np.int32),
                input_mask_pl: np.random.randint(0, 10, size=(batch_size, max_sequence_length)).astype(np.int32),
                segment_ids_pl: np.random.randint(0, 10, size=(batch_size, max_sequence_length)).astype(np.int32),
            })
            input_placeholder_info = [tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(i)) for i in ("input_ids:0", "input_mask:0", "segment_ids:0")]
            output_node_pl_info = [ tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(f"{tens}:0")) for tens in output_node_names ]
            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={ ip.name.split(':')[0]: ip for ip in input_placeholder_info },
                    outputs={ res.name.split(':')[0]: res for res in output_node_pl_info },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                )
            )

        # with tf.Session() as sess:
            saved_model_builder.add_meta_graph_and_variables(
                sess,
                # [tag] if tag else [],
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
                },)
        saved_model_builder.save()


if __name__ == '__main__':
    meta_model_input_path = "uncased_L-24_H-1024_A-16"
    savedModel_path = "google-bert-L"
    if os.path.exists(savedModel_path):
        shutil.rmtree(savedModel_path, ignore_errors=True)
    meta_to_saved_model(meta_model_input_path, savedModel_path)