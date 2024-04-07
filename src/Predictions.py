from DataProcessor import convert_image_to_array, normalize_set_of_images, get_class_labels

import tensorflow as tf

if __name__ == "__main__":
    loaded_model = tf.keras.models.load_model('../Models/alexnet-model.h5')
    img = convert_image_to_array('../Images/Tests/pepper_bell_healthy.JPG')
    img_list = [img]
    normalized = normalize_set_of_images(img_list)
    prediction = loaded_model.predict(normalized).argmax()
    labels = get_class_labels()
    print("Predicted class: " + labels[prediction])
