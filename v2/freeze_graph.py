# %%
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
# Get keras model

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)

#%%
# model.save("mnist_model")

# Convert Keras model to ConcreteFunction    
full_model = tf.function(lambda inputs: model(inputs))
full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])
# Get frozen ConcreteFunction    
frozen_func = convert_variables_to_constants_v2(full_model)

#%%
graph_def = frozen_func.graph.as_graph_def()

#%%
import tensorflow.compat.v1 as tf1
# %%
with tf1.Session() as sess:
    tf.import_graph_def(graph_def, name='')
    inputs_pl = sess.graph.get_tensor_by_name('inputs:0')
    outputs_pl = sess.graph.get_tensor_by_name('sequential/dense_1/Softmax:0')
    print(sess.run(outputs_pl, feed_dict={inputs_pl: x_test[:1]}))
    builder = tf1.saved_model.builder.SavedModelBuilder("tf1_mnist_model")
    prediction_signature = (
        tf1.saved_model.signature_def_utils.build_signature_def(
            inputs={ 
                "inputs": tf1.saved_model.utils.build_tensor_info(inputs_pl),
            },
            outputs={ "Softmax": tf1.saved_model.utils.build_tensor_info(outputs_pl) },
            method_name=tf1.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )
    builder.add_meta_graph_and_variables(
        sess,
        [tf1.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": prediction_signature,
        },
    )
    builder.save()
# %%
