from process_dataset.get_dataset import extract_dataset, normalize_set_of_images, get_label_binarizer, augument_dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import math
import numpy as np

def split_train_test_dataset():
    print("[INFO] Spliting data to train, test")
    image_list, label_list = extract_dataset()
    np_image_list = normalize_set_of_images(image_list)
    image_labels, image_labels_classes = get_label_binarizer(label_list)

    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.2, random_state = 42)
    return x_train, x_test, y_train, y_test, len(image_labels_classes)

def train():
    base_model =  tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape= (224, 224, 3),
        include_top=False,
        weights='imagenet',
    )

    x_train, x_test, y_train, y_test, n_classes = split_train_test_dataset()


    # Add a global spatial average pooling layer
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(512, activation='relu')(out)
    out = Dense(512, activation='relu')(out)
    predictions = Dense(n_classes, activation='softmax')(out)
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=["accuracy"]) 

    model.summary()

    aug = augument_dataset()

    epochs = 10
    batch_size = 32

    # we train the model

    history = model.fit(
        aug.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs, 
        verbose=1
    )


    # we plot the accuracy and loss 

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    #Train and validation accuracy
    plt.plot(epochs, acc, 'b', label='Training accurarcy')
    plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
    plt.title('Training and Validation accurarcy')
    plt.legend()

    plt.figure()
    #Train and validation loss
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

    # we save the model 
    model.save("plant_disease_15_class_mobilenet.keras")


if __name__ == "__main__":
    train()