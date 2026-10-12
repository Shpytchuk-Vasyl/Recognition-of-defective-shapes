"""MobileNetV2 (ImageNet) + GAP + Dense - детектор дефектів, як у базовій роботі."""
import keras

from shapes import IMG_SIZE


def build_model(weights='imagenet'):
    base = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights=weights)
    base.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype='float32', name='image')
    x = keras.layers.Rescaling(1 / 127.5, offset=-1)(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation='sigmoid', name='defect')(x)
    return keras.Model(inputs, outputs, name='defect_classifier'), base


def load_trained(path):
    model, _ = build_model(weights=None)
    model.load_weights(path)
    return model
