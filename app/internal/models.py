import os
import tensorflow as tf

_model_path = os.path.abspath('bin/models/Best_EfficientNetB4_V2_image_input.keras')
ImageModel = tf.keras.models.load_model(_model_path)
