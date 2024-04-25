
import cv2
import gradio
import tensorflow as tf
import os
import numpy as np  

from keras.utils.image_utils import img_to_array
from PIL import Image

model_path = f"{os.getcwd()}/app/model.h5"

current_path = os.path.dirname(os.path.abspath(__file__))

if 'soranaaa' in current_path:
    directory_root = '/Users/soranaaa/Documents/ubb/third-year-sem2/ai/plants/PlantVillage'
elif 'Faculty materials' in current_path:
    directory_root = r'D:\Faculty materials\3rd year\2nd sem\AI in climate change\plants\PlantVillage'
else:
    directory_root = r'D:\.Uni\AI\plants\PlantVillage'

print('[INFO] Directory root:', directory_root)


def normalize_set_of_images(image_list):
    np_image_list = np.array(image_list, dtype=np.float16) / 255.0

    return np_image_list

def get_class_labels():
    class_labels = []
    try:
        root_dir = os.listdir(directory_root)
        root_dir = [disease_folder for disease_folder in root_dir if disease_folder != ".DS_Store"]

        for disease_folder in root_dir:
            class_labels.append(disease_folder)
    except Exception as e:  
        print(f"Error : {e}")
        class_labels = []  # Return an empty set in case of an error

    return class_labels



def load_model():
    # Load the model
    return tf.keras.models.load_model(model_path)



def process_webcam_image(image):
    # Convert array to an image
    image = Image.fromarray(image)
    
    # Calculate the center crop dimensions
    width, height = image.size
    new_width, new_height = 224, 224
    left = int((width - new_width)/2)
    top = int((height - new_height)/2)
    right = int((width + new_width)/2)
    bottom = int((height + new_height)/2)
    
    # Center crop the image
    cropped_image = image.crop((left, top, right, bottom))
    
    # Convert cropped PIL Image to numpy array for the model
    model_image = img_to_array(cropped_image)
    # Normalize the image
    model_image = model_image.astype('float32') / 255.0
    # Add a batch dimension
    model_image = np.expand_dims(model_image, axis=0)

    global model

    # Make a prediction
    prediction = model.predict(model_image).argmax()

    # Get the label of the prediction
    labels = get_class_labels()

    return labels[prediction], cropped_image


if __name__ == "__main__":
    # save_model()
    model = load_model()

    app = gradio.Interface(
        fn = process_webcam_image,
        inputs = gradio.Image(sources = ["webcam"]),
        outputs = ["text", "image"],
        live = True,
        title = "Detectare Plante Bolnave",
    )

    app.launch(
        # ssl_keyfile = "app/key.pem",
        # ssl_certfile = "app/cert.pem",
        # ssl_verify = False,
        share=True
    )
    