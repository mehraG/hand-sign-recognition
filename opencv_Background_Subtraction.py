from tensorflow import keras
import numpy as np
import cv2 as cv

# load the saved model
modelFileName = "model.44-0.34.hdf5"
loaded_model = keras.models.load_model(modelFileName)

cap = cv.VideoCapture(0)
_, first_frame = cap.read()
first_frame = cv.flip(first_frame, 1)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv.flip(frame, 1)

    diff = cv.subtract(first_frame, frame)
    diff_gray = cv.cvtColor(diff,cv.COLOR_BGR2GRAY)
    _, mask = cv.threshold(diff_gray, 21, 255, cv.THRESH_BINARY)
    res = cv.bitwise_and(frame,frame,mask=mask)
    
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,255,0) ,1)
    # Extracting the ROI
    roi = res[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv.resize(roi, (128, 128))
    roi = cv.flip(roi, 1)
    cv.imshow("test", roi)
    # Batch of 1
    image_arrays = [roi]
    image_arrays = np.array(image_arrays)

    # We need to format the input to match the training data
    # The data generator loaded the values as floating point numbers
    # and normalized the pixel values, so...
    img_features = image_arrays.astype('float32')
    img_features /= 255
    # These are the classes our model can predict
    classnames = ['E','F','I','L','V']
    # Predict the class of each input image
    predictions = loaded_model.predict(img_features)
    confidence = sorted(predictions[0], reverse=1)
    confidence = confidence[0]
    # The prediction for each image is the probability for each class, e.g. [0.8, 0.1, 0.2]
    # So get the index of the highest probability
    class_idx = np.argmax(predictions[0])
    result = classnames[class_idx]
    
    cv.putText(frame, str(confidence), (10, 100), cv.FONT_HERSHEY_PLAIN, 3,(153,107,78), 2)

    if confidence<0.65: result = "Pta nhi kya h."
    elif confidence<0.9: result = "F"
    
    # Displaying the predictions
    cv.putText(frame, result, (10, 70), cv.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 5)    

    cv.imshow('Frame', frame)
    cv.imshow('diff_gray', diff_gray)

    keyboard = cv.waitKey(30) & 0xFF
    if keyboard == ord('q') or keyboard == 27:
        break
    elif keyboard == ord('b'):
        _, first_frame = cap.read()
        first_frame = cv.flip(first_frame, 1)

cap.release()
cv.destroyAllWindows()
