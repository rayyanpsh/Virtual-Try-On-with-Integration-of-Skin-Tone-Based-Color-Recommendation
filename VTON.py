import cv2
import numpy as np
import mediapipe as mp
import os
import time
import pickle
import re
import math
import random
import tensorflow as tf
# Initialize MediaPipe Pose, Hands, and Face Mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Load the trained KNN model for size recommendation
with open("D:/VTON/Model/knn_size_recommender.pkl", "rb") as file:
    knn = pickle.load(file)

# Define clothing folders based on skin tone
CLOTHING_FOLDERS = {
    "Light": "D:/VTON/shirt/Light",
    "Medium": "D:/VTON/shirt/Medium",
    "Dark": "D:/VTON/shirt/Dark"
}
model = tf.keras.models.load_model('skin_tone_model.keras')
def get_clothing_folder(skin_tone):
    """
    Returns the appropriate clothing folder based on the detected skin tone.
    :param skin_tone: str, detected skin tone ("Light", "Medium", "Dark")
    :return: str, path to the clothing folder
    """
    return CLOTHING_FOLDERS.get(skin_tone, "D:/VTON/shirt/images")

def overlay_image(background, overlay, mask, top_left):
    x, y = top_left
    h, w = overlay.shape[:2]
    bg_height, bg_width = background.shape[:2]

    # Calculate regions to overlay
    overlay_x_start = max(0, -x)
    overlay_y_start = max(0, -y)
    overlay_x_end = min(w, bg_width - x)
    overlay_y_end = min(h, bg_height - y)

    bg_x_start = max(0, x)
    bg_y_start = max(0, y)
    bg_x_end = min(bg_width, x + w)
    bg_y_end = min(bg_height, y + h)

    # Extract regions
    overlay_region = overlay[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
    mask_region = mask[overlay_y_start:overlay_y_end, overlay_x_start:overlay_x_end]
    bg_region = background[bg_y_start:bg_y_end, bg_x_start:bg_x_end]

    # Ensure regions are not empty
    if overlay_region.size > 0 and bg_region.size > 0:
        # Normalize mask to [0, 1] and expand to 3 channels
        mask_region = mask_region[:, :, np.newaxis] / 255.0
        mask_region = np.repeat(mask_region, 3, axis=2)

        # Blend using the mask
        bg_region[:] = (1 - mask_region) * bg_region + mask_region * overlay_region

    return background

def draw_arrows(frame, w, h):
    cv2.arrowedLine(frame, (40, h // 2), (10, h // 2), (0, 0, 255), 5, tipLength=0.5)
    cv2.arrowedLine(frame, (w - 40, h // 2), (w - 10, h // 2), (0, 0, 255), 5, tipLength=0.5)

def calculate_top_left_position(landmarks, w, h, transformed_clothing):
    left_shoulder = np.array([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h])
    right_shoulder = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h])
    left_hip = np.array([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h])
    right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h])

    # Calculate shoulder midpoint
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2

    # Calculate hip midpoint
    hip_midpoint = (left_hip + right_hip) / 2

    # Calculate torso height and width
    torso_height = np.linalg.norm(shoulder_midpoint - hip_midpoint)
    torso_width = np.linalg.norm(left_shoulder - right_shoulder)

    # Adjust top-left position based on torso dimensions
    top_left = (
        int(max(0, shoulder_midpoint[0] - torso_width / 2 -70)),
        int(max(0, shoulder_midpoint[1] - torso_height / 10 - 60))
    )

    return top_left, torso_width, torso_height

def transform_clothing(clothing, torso_width, torso_height):
    # Scale clothing to fit torso dimensions
    scale_width = int(torso_width * 2.0)  # Add some padding
    scale_height = int(torso_height * 1.8)  # Add some padding
    resized_clothing = cv2.resize(clothing, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
    return resized_clothing

def predict_size(shoulder_distance, torso_height):
    features = np.array([[shoulder_distance, torso_height]])
    predicted_size = knn.predict(features)[0]  # Predict the size using the KNN model
    if predicted_size == 0:
        return 'Large'
    elif predicted_size == 1:
        return 'Medium'
    else:
        return 'Small'

def detect_swipe_gesture(hand_landmarks, frame_width):
    wrist_x = hand_landmarks.landmark[0].x * frame_width
    swipe_threshold = frame_width // 6  # Reduced threshold for better detection
    if wrist_x < swipe_threshold:
        return "swipe_left"
    elif wrist_x > (frame_width - swipe_threshold):
        return "swipe_right"
    return None

def analyze_skin_tone(frame, landmarks):
    # Extract the forehead region for skin tone analysis
    x1, y1 = int(landmarks[10][0]), int(landmarks[10][1])  # Forehead point
    x2, y2 = int(landmarks[152][0]), int(landmarks[152][1])  # Chin point

    # Check if coordinates are within the frame bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    #Ensure the ROI is valid
    if x1 >= x2 or y1 >= y2:
        return "Invalid ROI", (0, 0, 0)

    # Define region of interest (ROI) for skin tone analysis
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Empty ROI", (0, 0, 0)

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Calculate the average color in the region
    avg_color = np.mean(hsv_roi, axis=(0, 1))

    hue = avg_color[0]  # Hue is used for determining skin tone
    if hue < 20:
        return "Light", (255, 220, 177)
    elif hue < 40:
        return "Medium", (209, 177, 122)
    else:
        return "Dark", (100, 67, 30)

def load_clothing_images(clothing_folder, mask_folder):
    """
    Load clothing images and masks from the specified folders.
    :param clothing_folder: str, path to the folder containing clothing images
    :param mask_folder: str, path to the folder containing mask images
    :return: tuple of lists, (clothing_images, clothing_masks)
    """
    clothing_images = []
    clothing_masks = []
    
    print(f"Loading clothing images from: {clothing_folder}")
    print(f"Loading masks from: {mask_folder}")
    
    for fname in os.listdir(clothing_folder):
        if fname.endswith('.jpg') and not fname.startswith('.'):  # Look for .jpg images, exclude hidden files
            # Remove the .jpg extension to construct the mask filename
            base_name = os.path.splitext(fname)[0]  # Removes .jpg
            img_path = os.path.join(clothing_folder, fname)
            
            # Handle special characters in filenames
            mask_fname = re.sub(r'[^a-zA-Z0-9_]', '_', base_name) + "_mask.jpg"  # Replace special chars with _
            mask_path = os.path.join(mask_folder, mask_fname)  # Construct mask filename
            
            # Debug: Print image and mask paths
            print(f"Image: {img_path}")
            print(f"Mask: {mask_path}")
            
            # Debug: Check if files exist
            if not os.path.exists(img_path):
                print(f"Image file does not exist: {img_path}")
                continue  # Skip this image
            if not os.path.exists(mask_path):
                print(f"Mask file does not exist: {mask_path}")
                continue  # Skip this image
            
            clothing_img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load color image
            clothing_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Load mask as grayscale
            
            if clothing_img is None:
                print(f"Failed to load image: {img_path}")
                continue  # Skip this image
            if clothing_mask is None:
                print(f"Failed to load mask: {mask_path}")
                continue  # Skip this image
            
            if clothing_img is not None and clothing_mask is not None:
                clothing_images.append(clothing_img)
                clothing_masks.append(clothing_mask)
            else:
                print(f"Error loading image or mask: {fname}")
    
    print(f"Loaded {len(clothing_images)} clothing images and {len(clothing_masks)} masks.")
    return clothing_images, clothing_masks

# Define recommended colors for each skin tone
RECOMMENDED_COLORS = {
    "Light": ["Light Purple", "Rose", "Grey"],
    "Medium": ["Olive", "Blue", "Maroon"],
    "Dark": ["White", "Black", "Red", "Yellow", "Green"]
}

def main():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
         mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
         mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        
        last_gesture_time = time.time()
        gesture_cooldown = 2
        
        # Initialize variables for clothing and skin tone
        clothing_images = []
        clothing_masks = []
        current_clothing_index = 0
        skin_tone = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(frame_rgb)
            hands_results = hands.process(frame_rgb)
            face_results = face_mesh.process(frame_rgb)

            # Detect skin tone if not already detected
            if skin_tone is None and face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    landmarks = [(face_landmarks.landmark[i].x * w,
                                  face_landmarks.landmark[i].y * h,
                                  face_landmarks.landmark[i].z * w) for i in range(468)]
                    
                    # Perform skin tone analysis
                    skin_tone, color = analyze_skin_tone(frame, landmarks)
                    print(f"Detected Skin Tone: {skin_tone}")
                    
                    # Load clothing images and masks based on skin tone
                    clothing_folder = get_clothing_folder(skin_tone)
                    mask_folder = os.path.join(clothing_folder, "masks")
                    clothing_images, clothing_masks = load_clothing_images(clothing_folder, mask_folder)
                    print(f"Loaded {len(clothing_images)} clothing images and {len(clothing_masks)} masks for {skin_tone} skin tone.")

            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    gesture = detect_swipe_gesture(hand_landmarks, w)
                    if gesture and time.time() - last_gesture_time > gesture_cooldown:
                        if gesture == "swipe_left":
                            current_clothing_index = (current_clothing_index - 1) % len(clothing_images)
                        elif gesture == "swipe_right":
                            current_clothing_index = (current_clothing_index + 1) % len(clothing_images)
                        last_gesture_time = time.time()
                        
                        # Debug: Print the current clothing index
                        print(f"Current Clothing Index: {current_clothing_index}")

            if pose_results.pose_landmarks and len(clothing_images) > 0:
                landmarks = pose_results.pose_landmarks.landmark
                top_left, torso_width, torso_height = calculate_top_left_position(landmarks, w, h, clothing_images[current_clothing_index])

                # Predict clothing size using the trained KNN model
                predicted_size = predict_size(torso_width, torso_height)

                transformed_clothing = transform_clothing(clothing_images[current_clothing_index], torso_width, torso_height)
                transformed_mask = cv2.resize(clothing_masks[current_clothing_index], 
                                             (transformed_clothing.shape[1], transformed_clothing.shape[0]), 
                                             interpolation=cv2.INTER_NEAREST)
                
                frame = overlay_image(frame, transformed_clothing, transformed_mask, top_left)
                
                # Display the predicted size on the try-on interface
                cv2.putText(frame, f"Recommended Size: {predicted_size}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display skin tone and recommended colors
            if skin_tone is not None:
                # Display skin tone
                cv2.putText(frame, f"Skin Tone: {skin_tone}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display recommended colors
                recommended_colors = RECOMMENDED_COLORS.get(skin_tone, [])
                if recommended_colors:
                    colors_text = "Recommended Colors: " + ", ".join(recommended_colors)
                    cv2.putText(frame, colors_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            draw_arrows(frame, w, h)
            cv2.imshow('Virtual Try-On', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()