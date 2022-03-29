#%%
import tensorflow.compat.v1 as tf
import logging
#%%
reader = tf.train.NewCheckpointReader("/mnt/scratch001/custeng-cn-scratch/yongxiy/Desktop/unilm-latest/text-generation-inference-ipu_diverse_beam_search/ckpt/model.ckpt")
# %%
var_to_map = reader.get_variable_to_dtype_map()
# %%
var_to_shape_map = reader.get_variable_to_shape_map()
# %%
def add_variable(saved_variables, old_tensor, new_tensor):
    logging.info(f"{old_tensor} -> {new_tensor}")
    saved_variables.append(new_tensor)

saved_variables = []
with tf.Graph().as_default():
    sess = tf.Session()
    for name, dtype in var_to_map.items():
        if dtype == tf.float32:
            tensor_value = tf.Variable(tf.cast(reader.get_tensor(name), dtype=tf.float16), name = name)
            add_variable(saved_variables, name, tensor_value)
        else:
            add_variable(saved_variables, name, reader.get_tensor(name))
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=saved_variables)
    saver.save(sess, "ckpt/model.ckpt")