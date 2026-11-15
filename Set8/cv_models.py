import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 10

def build_small_cnn():
    """Small VGG-style CNN (2 conv blocks). Input: 32x32x3 in 0-255."""
    inp = keras.Input(shape=(32, 32, 3), name="image")
    x = layers.Rescaling(1.0 / 255, name="rescale")(inp)
    for i, f in enumerate([32, 64]):
        x = layers.Conv2D(f, 3, padding="same", activation="relu", kernel_initializer="he_uniform")(x)
        x = layers.Conv2D(f, 3, padding="same", activation="relu", kernel_initializer="he_uniform")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(0.25)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_initializer="he_uniform", name="features")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return keras.Model(inp, out, name="small_cnn")

def build_inception_transfer(pretrained=True, input_size=96):
    """InceptionV3 (ImageNet) + new head. Input: 32x32x3 in 0-255.
    Resizes to input_size and applies Inception preprocessing ([-1, 1]) inside the model."""
    inp = keras.Input(shape=(32, 32, 3), name="image")
    x = layers.Resizing(input_size, input_size, name="resize")(inp)
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="inception_preprocess")(x)
    base = keras.applications.InceptionV3(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=(input_size, input_size, 3), name="inception_v3")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)   # 2048-dim vector
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation="relu", name="features")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)
    return keras.Model(inp, out, name="inception_transfer")

BUILDERS = {"small_cnn": build_small_cnn, "inception_transfer": build_inception_transfer}
