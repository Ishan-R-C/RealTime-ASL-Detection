import os
import cv2
import numpy as np

#initialize parameters
label = 'A' #change label for each subfolder
INPUT = f'dataset/{label}'
OUTPUT = f'processed/{label}'
TARGET_SIZE = (128,128)
THRESHOLD = 130

os.makedirs(OUTPUT, exist_ok = True)
image_files = [f for f in os.listdir(INPUT) if f.lower().endswith(('.png','.jpg','.jpeg'))]

def remove_noise(img, min_area=100):
    #function to remove random specks and noise from the image

    inverted_img = cv2.bitwise_not(img)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_img, connectivity=8)
    cleaned = np.zeros_like(img)

    for i in range(1, num_labels): 
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    return cv2.bitwise_not(cleaned)


for idx, file_name in enumerate(image_files):
    input_path = os.path.join(INPUT, file_name)
    output_path = os.path.join(OUTPUT, file_name)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Skipping Invalid File: {file_name}")
        continue

    img_resized = cv2.resize(img, TARGET_SIZE, interpolation = cv2.INTER_LINEAR) #resized to 128x128
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) #grayscaled
    img_thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) #threshold used to find edges

    img_clean = remove_noise(img_thresh, min_area=80) #image cleaned of any noise

    cv2.imwrite(output_path, img_clean) #image saved
    print(f"[{idx + 1}/{len(image_files)}] Processed and Saved: {file_name}") 

print("All images processed and saved to: ",OUTPUT)