import cv2
import mediapipe as mp
from keras.models import load_model
import numpy as np
import pandas as pd
import time
from gtts import gTTS
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the pre-trained Keras model
model = load_model('models/mnist2.h5')

# Initialize MediaPipe hands with higher confidence thresholds
mphands = mp.solutions.hands
hands = mphands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils

s = ""
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

# Video source (0 for default webcam)
url = 0

# Initialize video capture
cap = cv2.VideoCapture(url)
_, frame = cap.read()
h, w, c = frame.shape
print(h, w)

# Initialize timing variables
start_time = None
threshold_seconds = 1.0  # Set the threshold time in seconds
timer_start_time = None  # Start the timer

while True:
    ret, frame = cap.read()
    if not ret:
        break

    analysisframe = frame
    framergbanalysis = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2RGB)
    resultanalysis = hands.process(framergbanalysis)
    hand_landmarksanalysis = resultanalysis.multi_hand_landmarks

    if hand_landmarksanalysis:
        for handLMsanalysis in hand_landmarksanalysis:
            x_min = w
            x_max = 0
            y_min = h
            y_max = 0

            for landmarks in handLMsanalysis.landmark:
                x, y = int(landmarks.x * w), int(landmarks.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y

            # Ensure bounding box is within the frame dimensions
            y_min = max(0, y_min - 20)
            y_max = min(h, y_max + 20)
            x_min = max(0, x_min - 20)
            x_max = min(w, x_max + 20)

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 3)
            mp_drawing.draw_landmarks(frame, handLMsanalysis, mphands.HAND_CONNECTIONS)

            try:
                analysisframe = cv2.cvtColor(analysisframe, cv2.COLOR_BGR2GRAY)
                analysisframe = analysisframe[y_min:y_max, x_min:x_max]
                analysisframe = cv2.resize(analysisframe, (28, 28))
                flat_image = analysisframe.flatten()
                datan = pd.DataFrame(flat_image).T
                pixeldata = datan.values
                pixeldata = pixeldata / 255
                pixeldata = pixeldata.reshape(-1, 28, 28, 1)

                # Prediction
                prediction = model.predict(pixeldata)
                predarray = np.array(prediction[0])

                letter_prediction_dict = {letterpred[i]: predarray[i] for i in range(len(letterpred))}
                letter, probability = "", 0

                for key, value in letter_prediction_dict.items():
                    if value > probability:
                        probability = value
                        letter = key

                current_time = time.time()

                # Check if prediction is 1.0 and start the timer
                if probability == 1.0:
                    if timer_start_time is None:
                        timer_start_time = current_time
                    elapsed_time = current_time - timer_start_time
                    if elapsed_time >= threshold_seconds:
                        print(f"Predicted Letter: {letter} with probability: {probability}")
                        s += letter
                        timer_start_time = None  # Reset timer after adding letter
                else:
                    timer_start_time = None  # Reset timer if prediction is not 1.0

                # Show the timer if hand is detected and prediction is 1.0
                if timer_start_time is not None:
                    elapsed_time = current_time - timer_start_time
                    timer_display = "Timer: {:.2f}".format(elapsed_time)
                    cv2.putText(frame, timer_display, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Format the letter and probability for display
                letter_display = "{} prob:{}".format(letter, probability)
                font = cv2.FONT_HERSHEY_SIMPLEX
                position = (x_max, y_min)  # Specify the (x, y) coordinates where you want to place the text
                font_scale = round(h / 400)  # Font scale

                font_color = (255, 255, 255)  # Font color in BGR format (white in this example)
                font_thickness = round(h / 200)  # Font thickness

                # Draw the text on the frame
                cv2.putText(frame, letter_display, position, font, font_scale, font_color, font_thickness)

            except cv2.error as e:
                pass

    # Display the frame
    cv2.imshow("Sign Language Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 13:  # Enter key
        try:
            if not s:
                raise ValueError("Empty string")
            # Convert the accumulated string to speech using gTTS
            tts = gTTS(s, lang='en')
            tts.save("output.mp3")

            # Play the audio file using pygame
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()

            # Keep the program running until the audio finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

        except ValueError as e:
            # Handle the empty string case
            tts = gTTS("No text to speak", lang='en')
            tts.save("output.mp3")

            # Play the audio file using pygame
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()

            # Keep the program running until the audio finishes playing
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

cap.release()
cv2.destroyAllWindows()
print(s)
