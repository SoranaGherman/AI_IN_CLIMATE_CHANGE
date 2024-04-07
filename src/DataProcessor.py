import cv2
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from keras.utils.image_utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from os import listdir


current_path = os.path.dirname(os.path.abspath(__file__))

if 'soranaaa' in current_path:
    directory_root = '/Users/soranaaa/Documents/ubb/third-year-sem2/ai/plants/PlantVillage'
elif 'Faculty materials' in current_path:
    directory_root = r'D:\Faculty materials\3rd year\2nd sem\AI in climate change\plants\PlantVillage'
else:
    directory_root = r'D:\.Uni\AI\plants\PlantVillage'

print('[INFO] Directory root:', directory_root)

default_image_size = (224, 224)  
MAX_IMAGES = 750


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is None :
            return np.array([])
        
        image = cv2.resize(image, default_image_size)   
        return img_to_array(image)
    except Exception as e:
        print(f"Error : {e}")
        return None


def extract_dataset():
    image_list, label_list = [], []
    try:
        root_dir = listdir(directory_root)
        root_dir = [disease_folder for disease_folder in root_dir if disease_folder != ".DS_Store"]
            
        for plant_disease_folder in root_dir:
            print(f"[INFO] Processing {plant_disease_folder} ...")
            plant_disease_image_list = listdir(f"{directory_root}/{plant_disease_folder}/")
            plant_disease_image_list = [disease_folder for disease_folder in plant_disease_image_list if disease_folder != ".DS_Store"]

            for image in plant_disease_image_list[:MAX_IMAGES]:
                image_directory = f"{directory_root}/{plant_disease_folder}/{image}"
                if image_directory.endswith(".jpg") == True or image_directory.endswith(".JPG") == True:
                    converted_image = convert_image_to_array(image_directory)
                    if converted_image is not None:
                        image_list.append(converted_image)
                        label_list.append(plant_disease_folder)

        return image_list, label_list   
    
    except Exception as e:
        print(f"Error : {e}")


def augument_dataset():
    aug = ImageDataGenerator(
        rotation_range = 25, 
        width_shift_range = 0.1,
        height_shift_range = 0.1, 
        shear_range = 0.2, 
        zoom_range = 0.2,
        horizontal_flip = True, 
        fill_mode="nearest"
    )

    return aug


def normalize_set_of_images(image_list):
    np_image_list = np.array(image_list, dtype=np.float16) / 255.0

    return np_image_list


def get_label_binarizer(label_list):
    label_binarizer = LabelBinarizer()
    image_labels = label_binarizer.fit_transform(label_list)
    d = dict(enumerate(label_binarizer.classes_))
    return image_labels, d


def histogram(label_list):

    labels = label_list

    plt.figure(figsize=(10, 8))
    sns.countplot(y=labels)
    plt.title('Distribuția claselor în setul de date')
    plt.xlabel('Numărul de imagini')
    plt.ylabel('Clase')
    plt.show()


def get_class_labels():
    class_labels = []
    try:
        root_dir = listdir(directory_root)
        root_dir = [disease_folder for disease_folder in root_dir if disease_folder != ".DS_Store"]

        for disease_folder in root_dir:
            class_labels.append(disease_folder)
    except Exception as e:  
        print(f"Error : {e}")
        class_labels = []  # Return an empty set in case of an error

    return class_labels


if __name__ == "__main__":
    # get_class_labels()
    image_list, label_list = extract_dataset()
    labels_binarizer = get_label_binarizer(label_list)

    image_labels_encoded, image_labels_classes = get_label_binarizer(label_list)
    print(image_labels_encoded)
    print(image_labels_classes)
    histogram(label_list)