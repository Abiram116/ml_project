import json
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize to [0,1]
    x_train = (x_train / 255.0).astype("float32")
    x_test = (x_test / 255.0).astype("float32")
    # Model accepts (28,28) and reshapes internally, but augmentation needs channel dim
    x_train_c = x_train[..., None]
    x_test_c = x_test[..., None]
    return (x_train, y_train, x_train_c), (x_test, y_test, x_test_c)


def build_model():
    # Match website preprocessing: (28, 28, 1)
    inputs = layers.Input(shape=(28, 28, 1))
    x = inputs

    # Strong but compact CNN
    def conv_block(x, filters):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        return x

    x = conv_block(x, 32)
    x = conv_block(x, 32)
    x = layers.MaxPooling2D()(x)

    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = layers.MaxPooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = conv_block(x, 128)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_datasets(x_train_c, y_train, x_test_c, y_test, batch_size=256):
    # Light augmentation
    aug = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomZoom(0.05),
    ])

    def ds(x, y, training):
        d = tf.data.Dataset.from_tensor_slices((x, y))
        if training:
            d = d.shuffle(10000)
        d = d.batch(batch_size)
        if training:
            d = d.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
        d = d.prefetch(tf.data.AUTOTUNE)
        return d

    train_ds = ds(x_train_c, y_train, True)
    test_ds = ds(x_test_c, y_test, False)
    return train_ds, test_ds


def main():
    set_seed(42)
    (x_train, y_train, x_train_c), (x_test, y_test, x_test_c) = load_data()

    train_ds, test_ds = get_datasets(x_train_c, y_train, x_test_c, y_test)
    model = build_model()

    # Callbacks
    ckpt_path = "mnist_best.keras"
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-5),
        ModelCheckpoint(ckpt_path, monitor="val_accuracy", save_best_only=True),
    ]

    history = model.fit(
        x_train_c, y_train,
        epochs=20,
        batch_size=256,
        validation_data=(x_test_c, y_test),
        callbacks=callbacks,
        verbose=2,
    )

    # Evaluate and save
    # Ensure models/ exists and save artifacts there
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    test_loss, test_acc = model.evaluate(x_test_c, y_test, verbose=0)
    model.save(os.path.join(models_dir, "mnist_model.h5"))

    # Persist metrics + history for the website
    metrics = {"test_accuracy": float(test_acc), "test_loss": float(test_loss)}
    with open(os.path.join(models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    hist = {
        "loss": [float(v) for v in history.history.get("loss", [])],
        "accuracy": [float(v) for v in history.history.get("accuracy", [])],
        "val_loss": [float(v) for v in history.history.get("val_loss", [])],
        "val_accuracy": [float(v) for v in history.history.get("val_accuracy", [])],
    }
    with open(os.path.join(models_dir, "training_history.json"), "w") as f:
        json.dump(hist, f, indent=2)

    print("Training complete. Test accuracy:", test_acc)
    print("Saved model to models/mnist_model.h5 and metrics to models/metrics.json, models/training_history.json")


if __name__ == "__main__":
    main()
