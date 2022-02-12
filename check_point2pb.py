import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
 
def freeze_graph(ckpt, output_graph):
    output_node_names = 'cls/predictions/output_bias'
    # saver = tf.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    saver = tf.compat.v1.train.import_meta_graph(ckpt+'.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
 
    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=output_node_names.split(',')
        )
        with tf.gfile.GFile(output_graph, 'wb') as fw:
            fw.write(output_graph_def.SerializeToString())
        print ('{} ops in the final graph.'.format(len(output_graph_def.node)))
 
ckpt = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
pb   = 'bert_model.pb'

if __name__ == '__main__':
    freeze_graph(ckpt, pb)