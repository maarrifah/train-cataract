import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model/modelnew.h5')

def predict_cataract(model, image_path, threshold=0.5):
    img = tf.keras.utils.load_img(image_path, target_size=(120, 120))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array / 255.0, axis=0)

    prediction = model.predict(img_array)[0][0]

    if 1 - prediction >= threshold:
        condition = 'Cataract'
        prediction_score = 1 - prediction
    else:
        condition = 'Normal'
        prediction_score = 1 - prediction

    print(f"Prediction score: {prediction_score:.4f}")
    print(f"The eye condition is: {condition}")

    return {"condition": condition, "prediction_score": prediction_score}
