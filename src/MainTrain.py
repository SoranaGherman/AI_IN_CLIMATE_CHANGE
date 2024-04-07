import uuid
import tensorflow as tf
import matplotlib.pyplot as plt

from DataProcessor import extract_dataset, normalize_set_of_images, get_label_binarizer, augument_dataset, convert_image_to_array, get_class_labels
from sklearn.model_selection import train_test_split
from keras.models import Model

from ModelsBulder import build_alexnet, build_densetnet, build_mobilenet


MODELS_BASE_PATH = "../Models"

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def split_train_test_dataset():
    print("[INFO] Spliting data to train, test")
    image_list, label_list = extract_dataset()
    np_image_list = normalize_set_of_images(image_list)
    image_labels, image_labels_classes = get_label_binarizer(label_list)

    x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size = 0.2, random_state = 42)

    return x_train, x_test, y_train, y_test, len(image_labels_classes)


def train(
    x_train,
    x_test,
    y_train,
    y_test,
    model: Model,
    epochs = 100,
    batch_size = 32,
    model_prefix = "alexnet"
):
    run_id = uuid.uuid4()

    model.compile(loss='categorical_crossentropy', optimizer="adam",metrics=["accuracy"]) 

    model.summary()

    aug = augument_dataset()
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{MODELS_BASE_PATH}/{model_prefix}-{run_id}.h5', verbose = 1, save_best_only = True)

    print("[INFO] Training the model...")

    history = model.fit(
        aug.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs, 
        callbacks=[checkpoint],
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


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test, n_classes = split_train_test_dataset()

    # model = build_alexnet(num_classes = n_classes)
    model = load_model(f'{MODELS_BASE_PATH}/alexnet-3f4f1c10-f4ad-4fb3-ba3a-34b7e9aa5ce4.h5')

    train(x_train, x_test, y_train, y_test, model, model_prefix = "alexnet", epochs = 25)

    