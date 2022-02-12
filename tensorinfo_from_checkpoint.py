import os
from tensorflow.python import pywrap_tensorflow
 
# code for finall ckpt
checkpoint_path = "uncased_L-12_H-768_A-12/bert_model.ckpt"
# Read data from checkpoint file
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
# Print tensor name and values
for key in var_to_shape_map:
    if key.startswith("cls"):
        print("tensor_name: ", key)
    # print(reader.get_tensor(key))