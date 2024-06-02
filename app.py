import os
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
model = load_model('models/mnist2_model.h5')

# Initialize MediaPipe hands
mphands = mp.solutions.hands
hands = mphands.Hands()
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
            
            y_min -= 20
            y_max += 20
            x_min -= 20
            x_max += 20

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

                # Check the timing mechanism
                current_time = time.time()
                if probability == 1.0:
                    if start_time is None:
                        start_time = current_time
                    elif current_time - start_time >= threshold_seconds:
                        print(f"Predicted Letter: {letter} with probability: {probability}")
                        s += letter
                        start_time = None  # Reset start_time after adding letter
                else:
                    start_time = None  # Reset timer if prediction is not 1

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
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Convert the accumulated string to speech using gTTS
tts = gTTS(s, lang='en')
tts.save("output.mp3")

# Play the audio file using pygame
pygame.mixer.music.load("output.mp3")
pygame.mixer.music.play()

# Keep the program running until the audio finishes playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

print(s)
