import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('gendermodel.h5')

# Define labels
labels = ['women', 'men']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Load the face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Loop to capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the frame to grayscale (needed for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each face detected
    for (x, y, w, h) in faces:
        # Extract the face region
        face = frame[y:y+h, x:x+w]

        # Resize the face to the input size expected by the model
        resized_face = cv2.resize(face, (150, 150))
        
        # Normalize pixel values between 0 and 1
        normalized_face = resized_face / 255.0
        
        # Expand dimensions to match model input
        input_data = np.expand_dims(normalized_face, axis=0)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Debug information
        print(f'Prediction probabilities: {prediction}')
        
        # If using sigmoid, classify based on threshold
        if len(prediction[0]) == 1:
            predicted_label = labels[int(prediction > 0.5)]
        else:
            predicted_label = labels[np.argmax(prediction)]

        # Draw a rectangle around the face and put text
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Prediction: {predicted_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with predictions
    cv2.imshow('Gender Detection', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windowsqq
cap.release()
cv2.destroyAllWindows()
