import os
import shutil
import zipfile
import cv2
import imghdr
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import  Dropout
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


app = Flask(__name__)

# Define the functions for extracting and sorting images
def extract_images(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    data_path = os.path.join(extract_path, 'data')
    if os.path.exists(data_path):
        sort_images(data_path)

def sort_images(extract_path):
    data_path = os.path.join(extract_path, 'data')
    defective_path = os.path.join(data_path, 'defective')
    not_defective_path = os.path.join(data_path, 'not_defective')

    os.makedirs(defective_path, exist_ok=True)
    os.makedirs(not_defective_path, exist_ok=True)

    for filename in os.listdir(extract_path):
        if filename.startswith('cast_def'):
            shutil.move(os.path.join(extract_path, filename), os.path.join(defective_path, filename))
        elif filename.startswith('cast_ok'):
            shutil.move(os.path.join(extract_path, filename), os.path.join(not_defective_path, filename))


# Remove dodgy image
    data_dir = data_path
    image_exts = ['jpeg','jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir): 
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try: 
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts: 
                    print('Image not in ext list {}'.format(image_path))
                    os.remove(image_path)
            except Exception as e: 
                print('Issue with image {}'.format(image_path))
                # os.remove(image_path)

    # Load Data
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

    # Scale Data
    data = data.map(lambda x,y: (x/255, y))
    data.as_numpy_iterator().next()

    # Split Data
    train_size = int(len(data)*.8)
    val_size = int(len(data)*.1)
    test_size = int(len(data)*.1)

    print(train_size)
    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size+val_size).take(test_size)

    # Build Deep Learning Model
    print(train)
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, (3,3), 1, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.summary()

    # Train
    logdir='logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    hist = model.fit(train, epochs=5, validation_data=val, callbacks=[tensorboard_callback])

   
    # Evaluate
    pre = Precision()
    re = Recall()
    acc = BinaryAccuracy()

    for batch in test.as_numpy_iterator(): 
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
        print(pre.result(), re.result(), acc.result())
        
    model.save(os.path.join('models','imageclassifier.h5'))
    
def predict_single_image(image_path, model_path):
    # Load the saved model
    new_model = load_model(model_path)

    try:
        # Read and resize the image
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError("Error reading the image. Please check the image path.")

        resize = tf.image.resize(img, (256, 256))

        # Predict class and confidence
        yhat = new_model.predict(np.expand_dims(resize/255, 0))
        confidence = yhat[0][0].item()

        # Print prediction and confidence
        if yhat > 0.8: 
            return 'not_defective', confidence
        else:
            return 'defective', 1 - confidence

    except Exception as e:
        print(f"Error processing the image: {e}")
        return None, None
    
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        zip_path = f"uploads/{uploaded_file.filename}"
        uploaded_file.save(zip_path)
        
        # Create a directory to extract images
        extract_path = f"extracted_images/{os.path.splitext(uploaded_file.filename)[0]}"
        os.makedirs(extract_path, exist_ok=True)
        
        # Extract images
        extract_images(zip_path, extract_path)
        
        # Process the images
        results = []
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(('jpeg', 'jpg', 'bmp', 'png')):
                    image_path = os.path.join(root, file)
                    model_path = r'C:\Users\Divya.mehra\OneDrive - ITRadiant Solution Pvt Ltd\Desktop\anamolly\models\imageclassifier.h5'
                    class_prediction, confidence = predict_single_image(image_path, model_path)
                    results.append({'image_path': image_path, 'prediction': class_prediction, 'confidence': confidence})  # Convert confidence to float

        return jsonify(results)
    return jsonify({'error': 'No file uploaded'}), 400

if __name__ == '__main__':
    app.run(host=0.0.0.0,port=3000)




