import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, send_file
from PIL import Image

app = Flask(__name__)

def create_autoencoder():
    input_img = tf.keras.layers.Input(shape=(32, 32, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    decoded = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = tf.keras.models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def train_autoencoder():
    # Load data
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Preprocess and normalize data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2)), mode='constant')  # resize images to 32x32
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2)), mode='constant')    # resize images to 32x32

    # Add noise to images
    noise_factor = 0.5
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    # Create a stacked autoencoder
    autoencoder = create_autoencoder()

    # Train the model
    autoencoder.fit(x_train_noisy, x_train,
                    epochs=50,
                    batch_size=128,
                    shuffle=True,
                    validation_data=(x_test_noisy, x_test))

    # Save the model
    model_path = os.path.join(os.getcwd(), 'model', 'stacked_autoencoder_model.h5')
    autoencoder.save(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    model = create_autoencoder()
    model_path = os.path.join(os.getcwd(), 'model', 'stacked_autoencoder_model.h5')
    model.load_weights(model_path)

    file = request.files['file']
    img = Image.open(file.stream).convert('L')  # convert to grayscale
    img = img.resize((32, 32))  # Resize the image to match the input shape of the model
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.
    img_array = np.reshape(img_array, (1, 32, 32, 1))

    denoised_image = model.predict(img_array)
    denoised_image = (denoised_image * 255).astype(np.uint8)
    denoised_image = Image.fromarray(denoised_image.reshape((32, 32)))
    denoised_image_path = 'static/denoised_image.png'
    denoised_image.save(denoised_image_path)

    return send_file(denoised_image_path, mimetype='image/png')

if __name__ == '__main__':
    if not os.path.exists('model'):
        os.makedirs('model')
        train_autoencoder()
    else:
        if not os.path.exists(os.path.join(os.getcwd(), 'model', 'stacked_autoencoder_model.h5')):
            train_autoencoder()
    app.run(debug=True)
