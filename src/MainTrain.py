import uuid
import tensorflow as tf
import matplotlib.pyplot as plt

from DataProcessor import extract_dataset, normalize_set_of_images, get_label_binarizer, augument_dataset
from sklearn.model_selection import train_test_split
from keras.models import Model

from ModelsBuilder import build_alexnet, build_densetnet, build_efficientnet, build_inception, build_mobilenet, build_xception


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
    file_name = "alexnet"
):
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    model.summary()

    aug = augument_dataset()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{MODELS_BASE_PATH}/{file_name}.h5', verbose=1, save_best_only=True)

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    print("[INFO] Training the model...")

    # Set the learning rate lower
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

    history = model.fit(
        aug.flow(x_train, y_train, batch_size=batch_size),
        validation_data=(x_test, y_test),
        epochs=epochs,
        callbacks=[checkpoint, reduce_lr],
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

    # model = build_xception(num_classes = n_classes)
    model = load_model(f'{MODELS_BASE_PATH}/Good/xception-2.h5')

    train(x_train, x_test, y_train, y_test, model, file_name = "xception-2", epochs = 50)
