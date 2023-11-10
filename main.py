import os
import shutil
import zipfile
import cv2
import imghdr
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.layers import  Dropout
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

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
    hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])

    # Plot Performance
    fig = plt.figure()
    plt.plot(hist.history['loss'], color='teal', label='loss')
    plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

    fig = plt.figure()
    plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
    plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()

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

        # Print prediction and confidence
        if yhat > 0.5: 
            return 'not_defective', yhat[0][0]
        else:
            return 'defective', 1 - yhat[0][0]

    except Exception as e:
        print(f"Error processing the image: {e}")
        return None, None



#if __name__ == '__main__':
    # zip_path = input("Enter the path of the zip file: ")   # Replace with your actual zip file path
    # extract_path = 'extracted_images'
    # model_path = r'C:\Users\Divya\Desktop\Anamoly_detection_1\Anamonly_detection1\models\imageclassifier.h5'

    # # Extract and sort images
    # extract_images(zip_path, extract_path)
    
    # results = []

    # for root, dirs, files in os.walk(extract_path):
    #     for file in files:
    #         if file.lower().endswith(('jpeg', 'jpg', 'bmp', 'png')):
    #             image_path = os.path.join(root, file)

    #             # Test the model on the user-provided image
    #             class_prediction, confidence = predict_single_image(image_path, model_path)

    #             if class_prediction is not None:
    #                 results.append((image_path, class_prediction, confidence))

    # # Save results to an Excel file
    # df = pd.DataFrame(results, columns=['Image Path', 'Prediction', 'Confidence'])
    # df.to_excel(r'C:\Users\Divya\Desktop\Anamoly_detection_1\Anamonly_detection1\predictions.xlsx', index=False)