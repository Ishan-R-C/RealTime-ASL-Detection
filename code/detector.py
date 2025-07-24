import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import os
from collections import deque, Counter
import threading
import queue
from gtts import gTTS
from io import BytesIO
import pygame
import time

#load models
model = tf.keras.models.load_model("asl_model.h5") #detects from A-Z
co_model = tf.keras.models.load_model("co_model.h5") #specializes in C & O distinction

tts_queue = queue.Queue()
pygame.init()
pygame.mixer.init()

#text-to-speech setup
def speak(text, language='en'):
    mp3_fo = BytesIO()
    tts = gTTS(text, lang=language)
    tts.write_to_fp(mp3_fo)
    mp3_fo.seek(0)
    return mp3_fo

def tts_worker():
    while True:
        text = tts_queue.get() #waits until a new word is available
        if text:
            try:
                sound = speak(text)
                pygame.mixer.music.load(sound, 'mp3')
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
            except Exception as e:
                print(f"Error in TTS: {e}")
        tts_queue.task_done()

prediction_buffer = deque(maxlen=10) #holds last 10 predictions
final_text = ""

DATA_DIR = "processed"
class_names = sorted(os.listdir(DATA_DIR)) #prediction classes taken
cl_nm = ['C', 'O']

#mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

#removing specks and noise from image
def remove_noise(img, min_area=100):
    inverted_img = cv2.bitwise_not(img)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_img, connectivity=8)

    cleaned = np.zeros_like(img)

    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cv2.bitwise_not(cleaned)

#process image of hand before prediction
def preprocess_hand(crop):
    img_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    img_clean = remove_noise(img_thresh, min_area=80)
    resized = cv2.resize(img_clean, (128, 128))
    normalized = resized / 255.0
    reshaped = normalized.reshape(1, 128, 128, 1)
    return reshaped

#Crops out the hand from the image
def get_square_box(landmarks, shape, padding=20):
    h, w, _ = shape
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]

    x_min = max(min(x_coords) - padding, 0)
    x_max = min(max(x_coords) + padding, w)
    y_min = max(min(y_coords) - padding, 0)
    y_max = min(max(y_coords) + padding, h)

    box_w = x_max - x_min
    box_h = y_max - y_min
    box_size = max(box_w, box_h)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    x1 = max(cx - box_size // 2, 0)
    y1 = max(cy - box_size // 2, 0)
    x2 = min(cx + box_size // 2, w)
    y2 = min(cy + box_size // 2, h)

    return x1, y1, x2, y2

cap = cv2.VideoCapture(0) #webcam initialized

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x1, y1, x2, y2 = get_square_box(hand_landmarks.landmark, frame.shape)
            cropped_hand = frame[y1:y2, x1:x2]

            if cropped_hand.size > 0:
                input_img = preprocess_hand(cropped_hand)
                predictions = model.predict(input_img)
                pred_idx = np.argmax(predictions)
                confidence = predictions[0][pred_idx]
                predicted_class = class_names[pred_idx]
                if predicted_class in ['C', 'O']:
                    co_pred = co_model.predict(input_img)
                    co_idx = np.argmax(co_pred)
                    confidence = co_pred[0][co_idx]
                    predicted_class = cl_nm[co_idx]
                prediction_buffer.append(predicted_class)

                #remove batch and channel dimensions, scale to 0-255, convert to uint8
                display_img = (input_img[0] * 255).astype(np.uint8) 
                cv2.imshow("Cropped Hand", display_img) #optional to see what the model is seeing

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{predicted_class} ({confidence*100:.1f}%)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 0, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if len(prediction_buffer) == 10:
        most_common, count = Counter(prediction_buffer).most_common(1)[0]
        if count > 7:  # Add a stability threshold
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if most_common == "Space":
                words = final_text.strip().split()
                if words:
                    last_word = words[-1]
                    tts_queue.put(last_word)
                final_text += " "
            elif most_common == "Delete":
                final_text = final_text[:-1]
            else:
                final_text += most_common
            prediction_buffer.clear()  # Reset buffer to avoid repeated addition
    cv2.putText(frame, f"Text: {final_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)  # thicker black outline
    cv2.putText(frame, f"Text: {final_text}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

    with open("asl_output.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    cv2.imshow("ASL Interpreter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tts_queue.put(None)  # Signal the TTS thread to exit
tts_thread.join()    # Wait for it to finish