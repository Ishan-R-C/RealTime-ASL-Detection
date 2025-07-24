import cv2
import mediapipe as mp 
import uuid
import os

#set up mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 1) #change max_num_hands for multiple hands
mp_draw = mp.solutions.drawing_utils

label = 'A' #change label to the alphabet you are collecting images for
save_dir = os.path.join("dataset", label)
os.makedirs(save_dir, exist_ok = True)

capture = cv2.VideoCapture(0) #webcam accessed at 0 index

def get_padded_box(landmarks, image_shape, padding = 20):
    #function to create a padded box around the hand using mediapipe for a clear image
    h, w, _ = image_shape 

    #pixel coordinates
    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]

    #initial box (within image boundaries) is found
    x_min, x_max = max(min(x_coords) - padding, 0), min(max(x_coords) + padding, w)
    y_min, y_max = max(min(y_coords) - padding, 0), min(max(y_coords) + padding, h)

    box_w = x_max - x_min #width
    box_h = y_max - y_min #height
    box_size = max(box_w, box_h) #finds max of both values

    #coords of box center found
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    #four coords of the square box calculated
    x1 = max(cx - box_size // 2, 0)
    y1 = max(cy - box_size // 2, 0)
    x2 = min(cx + box_size // 2, w)
    y2 = min(cy + box_size // 2, h)

    return x1, y1, x2, y2

while True:
    success, frame = capture.read()
    if not success: #checks if frame captured successfully
        break
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #frame converted from BGR to RGB (mediapipe supports RGB)
    
    #mediapipe landmarks found and drawn on your hand
    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x1, y1, x2, y2 = get_padded_box(hand_landmarks.landmark, frame.shape)
            cv2.rectangle(frame, (x1,y1), (x2,y2),(0,255,0),2)
    
    cv2.putText(frame, f"Ready for: {label}",(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255),2)
    cv2.imshow("WebCam",frame)

    key = cv2.waitKey(1)
    if key == ord('b'): #on pressing 'b', 10 frames of your hand are captured
        print(f"Capturing 10 cropped images for '{label}'")
        captured = 0
        while captured < 10:
            success, frame = capture.read()
            if not success:
                continue
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    x1,y1,x2,y2 = get_padded_box(hand_landmarks.landmark, frame.shape)
                    cropped = frame[y1:y2, x1:x2] #your hand gets cropped and taken out of the frame

                    if cropped.size != 0: 
                        file_name = f"{uuid.uuid4().hex}.jpg"
                        cv2.imwrite(os.path.join(save_dir,file_name),cropped) #cropped hand saved in database
                        captured += 1
                        print(f"Saved {captured}/10")

                    cv2.imshow("Cropped Hand", cropped) #shows the frame captured

            cv2.waitKey(100)

    elif key == 27: #while loop breaks on pressing ESC
        break

capture.release()
cv2.destroyAllWindows()