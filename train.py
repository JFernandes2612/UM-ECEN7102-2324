import tensorflow as tf
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import typing
from sklearn.utils import class_weight
import itertools

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_ds = tf.keras.utils.image_dataset_from_directory(
    "images/labeled/",
    validation_split=0.2,
    subset="training",
    label_mode="binary",
    seed=0,
    image_size=(64, 192),
    batch_size=32)
print(train_ds)


def calculate_class_weights(dataset: tf.data.Dataset) -> typing.Dict[int, np.ndarray]:
    all_class_names = np.concatenate([y for _, y in dataset], axis=0).tolist()
    class_weights = class_weight.compute_class_weight('balanced', classes=[
                                                      0.0, 1.0], y=list(itertools.chain.from_iterable(all_class_names)))

    print(enumerate(class_weights))

    return dict(enumerate(class_weights))


class_weights_dict = calculate_class_weights(train_ds)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "images/labeled/",
    validation_split=0.2,
    subset="validation",
    label_mode="binary",
    seed=0,
    image_size=(64, 192),
    batch_size=32)
print(val_ds)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1, input_shape=(64, 192, 3)),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
    tf.keras.layers.Rescaling(1./255, input_shape=(64, 192, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy(),
             tf.keras.metrics.Precision(),
             tf.keras.metrics.Recall()])

print(model.summary())

early_stoping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5)
epochs = 1000


class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.timetaken = datetime.now()

    def on_train_end(self, logs={}):
        self.total_time = (datetime.now() -
                           self.timetaken).total_seconds()


timetaken = timecallback()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stoping, timetaken],
    class_weight=class_weights_dict
)

model.save("models/default_model.tf")

scores = model.evaluate(val_ds)
print(f"Time taken: {timetaken.total_time}s")
print(f"Model Accuracy: {scores[1]*100}%")
print(f"Model Precision: {scores[2]*100}%")
print(f"Model Recall: {scores[3]*100}%")
print(
    f"Model F1 Score: {((2 * scores[2] * scores[3]) /(scores[2] + scores[3]))*100}%")
history.history['f1'] = []
history.history['val_f1'] = []
for index, val in enumerate(history.history['precision']):
    precision = val
    val_precision = history.history['val_precision'][index]
    recall = history.history['recall'][index]
    val_recall = history.history['val_recall'][index]
    history.history['f1'].append(
        2 * precision * recall / (precision + recall))
    history.history['val_f1'].append(
        2 * val_precision * val_recall / (val_precision +
                                          val_recall))

pd.DataFrame({key: val for key, val in
              history.history.items() if key in [
                  'categorical_accuracy',
                  'val_categorical_accuracy', 'precision',
                  'val_precision', 'recall', 'val_recall',
                  'f1', 'val_f1']}).plot(figsize=(8, 5))

plt.show()
