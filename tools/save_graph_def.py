import tensorflow as tf
import shutil

def saved_graph_def(output_dir, graph_def, meta_graph):
  with tf.Graph().as_default():
    tf.import_graph_def(graph_def, name="")
    with tf.Session() as sess:
      shutil.rmtree(output_dir, ignore_errors=True)
      builder = tf.saved_model.builder.SavedModelBuilder(output_dir)
      builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map=meta_graph.signature_def,
      )
      builder.save()