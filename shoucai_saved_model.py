import tensorflow.compat.v1 as tf
import os
from tensorflow.python.saved_model import builder


def convert_graph_def_to_graph(graph_def):
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
  return graph


def load_tf_graph(frozen_graph_filename, with_meta_graph=True,
          tag=tf.saved_model.tag_constants.SERVING):
  if with_meta_graph:
    if not os.path.isdir(frozen_graph_filename):
      model_path = os.path.dirname(frozen_graph_filename)
    else:
      model_path = frozen_graph_filename

    with tf.Session(graph=tf.Graph()) as sess:
      tf.saved_model.loader.load(sess, [tag] if tag else [], model_path)
      graph = tf.get_default_graph()
      return graph
  else:
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
    graph = convert_graph_def_to_graph(graph_def)
    return graph

def main():
    inputs = [
        'LookupPkOp',
        'all_clk_seq_1/st',
        'all_clk_seq_1/time',
        'batch_fill_attributes_for_gul_rank_item_feature',
        'batch_fill_attributes_for_gul_rank_item_feature_1',
        'embedding/item_cate_id_d_shared_embedding_2',
        'embedding/item_id_d_shared_embedding_2',
        'embedding/item_seller_id_d_shared_embedding_2',
        'embedding/ui_page_shared_embedding',
        'input_from_feature_columns/concat',
        'input_from_feature_columns/concat_1',
        'input_from_feature_columns/concat_3',
        'input_from_feature_columns/concat_4',
        'input_from_feature_columns/concat_5',
        'input_from_feature_columns/concat_7',
        'seq_input_from_feature_columns/concat',
        'seq_input_from_feature_columns/concat_1',
        'seq_input_from_feature_columns/concat_2',
    ]

    outputs = ['CTR_Mark_Output/rank_predict']
    g = load_tf_graph("model/shoucai/saved_model.pb", with_meta_graph=False)

    saved_model_builder = builder.SavedModelBuilder("models/shoucai_cpu")
    with tf.Session(graph=g) as sess:
        # inputs_tensors = [sess.graph.get_tensor_by_name(f"{inp_name}:0") for inp_name in inputs]
        # output_tensors = [sess.graph.get_tensor_by_name(f"{out_name}:0") for out_name in outputs]

        input_placeholder_info = [tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(f"{i}:0")) for i in inputs]
        output_node_pl_info = [ tf.saved_model.utils.build_tensor_info(sess.graph.get_tensor_by_name(f"{tens}:0")) for tens in outputs ]
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={ ip.name.split(':')[0]: ip for ip in input_placeholder_info },
                outputs={ res.name.split(':')[0]: res for res in output_node_pl_info },
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
            )
        )
        saved_model_builder.add_meta_graph_and_variables(
            sess,
            # [tag] if tag else [],
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            },)
    saved_model_builder.save()

if __name__ == "__main__":
    main()