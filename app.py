import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Load your pre-trained Keras model
model = load_model('models/sign-lang.h5')

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Video Capture (webcam)
cap = cv2.VideoCapture(0)

# Define dictionary to map class indices to sign language labels
sign_language_labels = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X'
}

# Main loop
while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    # If frame was not read, break the loop
    if not ret:
        break
    
    # Convert frame to RGB (MediaPipe requires RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the bounding box of the detected hand
            x_min, y_min = np.min([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)
            x_max, y_max = np.max([(lm.x, lm.y) for lm in hand_landmarks.landmark], axis=0)

            # Convert bounding box coordinates to pixel values
            x_min, y_min = int(x_min * frame.shape[1]), int(y_min * frame.shape[0])
            x_max, y_max = int(x_max * frame.shape[1]), int(y_max * frame.shape[0])

            # Crop the region around the hand
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Resize the hand region to 28x28 pixels and convert to grayscale
            hand_region_resized = cv2.resize(hand_region, (28, 28))
            hand_region_gray = cv2.cvtColor(hand_region_resized, cv2.COLOR_BGR2GRAY)

            # Convert the grayscale image to the format required by the model
            hand_region_input = hand_region_gray.reshape(1, 28, 28, 1).astype('float32')

            # Predict the sign language gesture using the model
            prediction = model.predict(hand_region_input)

            # Process the prediction (e.g., find the index of the maximum probability)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Get the label of the predicted class
            predicted_label = sign_language_labels.get(predicted_class, 'Unknown')
            
            # Display the predicted label on the frame
            cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            # Draw landmarks and connections on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
    
    # Display the frame
    cv2.imshow('Sign Language Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
