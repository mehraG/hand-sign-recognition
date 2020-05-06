# Firstly, install opencv if you're getting errors.
# Open cmd and type command: pip install opencv-python

#install tensorflow and keras in cmd
# pip install --upgrade tensorflow
# pip install keras

from tensorflow import keras
import cv2
import numpy as np
'''
# Loading the model
json_file = open("handsign-classifierminloss.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = keras.models.model_from_json(model_json)
# load weights into new model
loaded_model.load_weights("handsign-classifierminloss.h5")
print("Loaded model from disk")
'''
# load the saved model
modelFileName = "model4F.01-0.50.hdf5"
loaded_model = keras.models.load_model(modelFileName) 

cap = cv2.VideoCapture(0)
_, first_frame = cap.read()
first_frame = cv2.flip(first_frame, 1)

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    diff = cv2.subtract(first_frame,frame)
    diff_gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(diff_gray, 21, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(frame,frame,mask=mask)
    # Got this from collect-data.py
    # Coordinates of the ROI
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
    
    # Extracting the ROI
    roi = res[y1:y2, x1:x2]
    
    # Resizing the ROI so it can be fed to the model for prediction
    roi = cv2.resize(roi, (128, 128))
    roi = cv2.flip(roi, 1)
    cv2.imshow("test", roi)
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
    
    cv2.putText(frame, str(confidence), (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)
    
    # Displaying the predictions
    cv2.putText(frame, result, (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (100, 100, 100), 5)    
    cv2.imshow("Frame", frame)
    cv2.imshow("diff_gray", diff_gray)
    
    keyboard = cv2.waitKey(30) & 0xFF
    if keyboard == ord('q') or keyboard == 27:
        break
    elif keyboard == ord('b'):
        _, first_frame = cap.read()
        first_frame = cv2.flip(first_frame, 1)        
 
cap.release()
cv2.destroyAllWindows()
