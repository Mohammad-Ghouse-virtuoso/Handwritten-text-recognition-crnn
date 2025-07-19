from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained model
model_path = "complete_model.keras"
act_model = load_model(model_path, compile=False)

# Character list for decoding
char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=(32, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_text(image_path):
    image = preprocess_image(image_path)
    prediction = act_model.predict(image)
    decoded = tf.keras.backend.ctc_decode(
        prediction,
        input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
        greedy=True
    )[0][0].numpy()
    result_text = "".join([char_list[int(p)] for p in decoded if int(p) != -1])
    return result_text
