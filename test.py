import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

model_file = 'model.h5'

try:
    # Load the trained model
    model = load_model(model_file)

    # Dictionary to map class indices to class labels
    class_to_label = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic'}

    # Function to preprocess the image
    def preprocess_image(image, target_size):
        img = cv2.resize(image, target_size)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        return np.expand_dims(img_array, axis=0)

    # Function to predict the class of an image
    def predict_image_class(image_array, model):
        prediction = model.predict(image_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_to_label[predicted_class_index]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence

    # Capture video from the camera
    cap = cv2.VideoCapture(2)

    while True:
        ret, frame = cap.read()
        
        # Display the captured frame
        cv2.imshow('Camera', frame)
        
        # Preprocess the frame for prediction
        image_array = preprocess_image(frame, target_size=(32, 32))
        
        # Predict the class of the frame
        predicted_class, confidence = predict_image_class(image_array, model)
        
        # Display the prediction
        cv2.putText(frame, f'Prediction: {predicted_class} ({confidence:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Camera', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Error loading the model: {e}")
