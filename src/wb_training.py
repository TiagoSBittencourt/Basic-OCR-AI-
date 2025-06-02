import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import json

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1])  # achata 28x28 para 784
    label = tf.one_hot(label, 10)
    return image, label

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
    shuffle_files=True
)

batch_size = 64

ds_train = ds_train.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds_train, epochs=5, validation_data=ds_test)
model.evaluate(ds_test)

# Extrair pesos e bias das camadas
theta1, b1 = model.layers[0].get_weights()  # Dense 128
theta2, b2 = model.layers[1].get_weights()  # Dense 10

# Criar dicionário no formato desejado
nn_params = {
    "theta1": theta1.tolist(),
    "theta2": theta2.tolist(),
    "b1": b1.tolist(),
    "b2": b2.tolist()
}

# Salvar em JSON
with open("src/neural_network.json", "w") as f:
    json.dump(nn_params, f)