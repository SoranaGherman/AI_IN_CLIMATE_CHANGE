from os import listdir
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
# from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
 
directory_root = '/Users/soranaaa/Documents/ubb/third-year-sem2/ai/plants'
# directory_root = r'D:\Faculty materials\3rd year\2nd sem\AI in climate change\plants'
default_image_size = (224, 224)  


def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None :
            image = cv2.resize(image, default_image_size)   
            return img_to_array(image)
        else :
            return np.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def extract_dataset():
    image_list, label_list = [], []
    try:
        root_dir = listdir(directory_root)
        for directory in root_dir :
            if directory == ".DS_Store" :
                root_dir.remove(directory)

        for plant_folder in root_dir :
            plant_disease_folder_list = listdir(f"{directory_root}/{plant_folder}")
            
            for disease_folder in plant_disease_folder_list :
                if disease_folder == ".DS_Store" :
                    plant_disease_folder_list.remove(disease_folder)

            for plant_disease_folder in plant_disease_folder_list:
                print(f"[INFO] Processing {plant_disease_folder} ...")
                plant_disease_image_list = listdir(f"{directory_root}/{plant_folder}/{plant_disease_folder}/")
                    
                for single_plant_disease_image in plant_disease_image_list :
                    if single_plant_disease_image == ".DS_Store" :
                        plant_disease_image_list.remove(single_plant_disease_image)

                for image in plant_disease_image_list[:200]:
                    image_directory = f"{directory_root}/{plant_folder}/{plant_disease_folder}/{image}"
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
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, 
    zoom_range=0.2,horizontal_flip=True, 
    fill_mode="nearest")

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



if __name__ == "__main__":
    image_list, label_list = extract_dataset()
    labels_binarizer = get_label_binarizer(label_list)

    image_labels_encoded, image_labels_classes = get_label_binarizer(label_list)
    histogram(label_list)