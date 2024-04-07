import tensorflow as tf

from keras.layers import GlobalAveragePooling2D, Dense, Conv2D, BatchNormalization, Flatten, Dropout
from keras.models import Model


def build_alexnet(input_shape=(224, 224, 3), num_classes=13):
    model = tf.keras.models.Sequential()

    # Layer 1
    model.add(tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=input_shape, padding='valid'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Layer 2
    model.add(tf.keras.layers.Conv2D(256, (5, 5), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2)))

    # Layer 3
    model.add(tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # Layer 4
    model.add(tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    # Layer 5
    model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))  # Adjust pool size to (2, 2) to maintain dimensionality


    # Flatten and fully connected layers
    model.add(tf.keras.layers.Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='root'))

    return model


def build_densetnet(input_shape=(224, 224, 3), num_classes=13):
    densenet = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False)

    densenet.trainable = False

    input = tf.keras.Input(shape=input_shape)
    x = Conv2D(3, (3, 3), padding='same')(input)
    
    x = densenet(x)
    
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # multi output
    output = Dense(num_classes, activation = 'softmax', name='root')(x)

    # model
    model = Model(input, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model


def build_resenet(input_shape=(224, 224, 3), num_classes=13):
    # Load the ResNet50 model pre-trained on ImageNet data
    base_model = tf.keras.applications.ResNet50(
        input_shape = input_shape,
        weights='imagenet', 
        include_top=False
    )

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of ResNet
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # New FC layer, random init
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # New softmax layer

    # Define the new model
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def build_mobilenet(input_shape=(224, 224, 3), num_classes=13):

    base_model =  tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape= input_shape,
        include_top = False,
        weights = 'imagenet',
    )

    # Add a global spatial average pooling layer
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(512, activation = 'relu')(out)
    out = Dense(512, activation = 'relu')(out)
    predictions = Dense(num_classes, activation = 'softmax')(out)
    model = Model(inputs = base_model.input, outputs = predictions)

    for layer in base_model.layers:
        layer.trainable = False

    return model