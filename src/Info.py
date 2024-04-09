from ModelsBuilder import build_alexnet, build_densetnet, build_efficientnet, build_inception, build_mobilenet, build_xception

import tensorflow as tf

if __name__ == "__main__":

    model = build_xception()

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])

    model.summary()