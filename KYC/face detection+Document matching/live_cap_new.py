import cv2
import mediapipe as mp
import time
from scipy.spatial import distance
from collections import deque
import random

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to initialize Mediapipe face mesh
def init_face_mesh():
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=2)  # Set max_num_faces to 2 to detect multiple faces

# Function to get facial landmarks for eyes
def get_eye_landmarks(face_landmarks, frame_shape, left_eye_idxs, right_eye_idxs):
    left_eye = [(int(face_landmarks.landmark[idx].x * frame_shape[1]), 
                 int(face_landmarks.landmark[idx].y * frame_shape[0])) for idx in left_eye_idxs]
    
    right_eye = [(int(face_landmarks.landmark[idx].x * frame_shape[1]), 
                  int(face_landmarks.landmark[idx].y * frame_shape[0])) for idx in right_eye_idxs]

    return left_eye, right_eye

# Function to detect head tilt direction
def detect_head_tilt(face_landmarks):
    nose_y = face_landmarks.landmark[1].y
    left_eye_y = face_landmarks.landmark[33].y
    right_eye_y = face_landmarks.landmark[362].y

    nose_x = face_landmarks.landmark[1].x
    left_eye_x = face_landmarks.landmark[33].x
    right_eye_x = face_landmarks.landmark[362].x

    tilt_direction = ""

    avg_eye_y = (left_eye_y + right_eye_y) / 2
    avg_eye_x = (left_eye_x + right_eye_x) / 2

    if nose_y < avg_eye_y - 0.005:
        tilt_direction = "Up"
    elif nose_y > avg_eye_y + 0.08:
        tilt_direction = "Down"
    
    if nose_x < avg_eye_x - 0.01:
        tilt_direction += " Right"
    elif nose_x > avg_eye_x + 0.06:
        tilt_direction += " Left"

    return tilt_direction.strip()

# Function to generate a random direction
def generate_random_direction():
    directions = ["Up", "Down", "Left", "Right"]
    return random.choice(directions)

# Function to process the video frame and detect blinks
def process_frame(face_mesh, frame, left_eye_idxs, right_eye_idxs, ear_history, blink_counter, total_blinks, last_blink_time):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks is not None and len(results.multi_face_landmarks) == 1:  # Proceed only if exactly 1 face detected
        face_landmarks = results.multi_face_landmarks[0]
        left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape, left_eye_idxs, right_eye_idxs)
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        ear_history.append(ear)
        smoothed_ear = sum(ear_history) / len(ear_history)

        if len(ear_history) > 5:
            initial_ear = sum(ear_history) / len(ear_history)
            adjusted_threshold = initial_ear * 0.75
        else:
            adjusted_threshold = 0.21  # Default for the first few frames

        if smoothed_ear < adjusted_threshold:
            blink_counter += 1
            last_blink_time = time.time()
        else:
            if blink_counter >= 3:  # If blink lasts for 3 consecutive frames
                total_blinks += 1
            blink_counter = 0

        return total_blinks, blink_counter, last_blink_time, face_landmarks, False  # Return face_landmarks for further processing
    else:
        # Handle case when no face or multiple faces detected
        if results.multi_face_landmarks is None:
            return total_blinks, blink_counter, last_blink_time, None, False  # No face detected
        else:
            return total_blinks, blink_counter, last_blink_time, None, len(results.multi_face_landmarks) > 1  # Check for multiple faces


# Main function to capture video stream and detect both blinks and head tilts
# Main function to capture video stream and detect both blinks and head tilts
# Function to capture video stream and detect both blinks and head tilts
def blink_and_tilt_detection(frame_placeholder):
    blink_counter = 0
    total_blinks = 0
    last_blink_time = time.time()
    last_change_time = time.time()  # Track the last time a change (blink or tilt) occurred
    ear_history = deque(maxlen=5)
    cap = cv2.VideoCapture(0)
    
    face_mesh = init_face_mesh()

    left_eye_idxs = [33, 160, 158, 133, 153, 144]  # Mediapipe left eye indices
    right_eye_idxs = [362, 385, 387, 263, 373, 380]  # Mediapipe right eye indices
    matched_count = 0  # Head tilt match counter

    random_direction = generate_random_direction()
    consecutive_correct_tilt = 0  # Counter for consecutive correct tilts

    # Capture the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to capture the first frame.")
        cap.release()
        cv2.destroyAllWindows()
        return False, None  # Return False if frame capture failed

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        original_frame = frame.copy()  # Keep the original frame for processing

        total_blinks, blink_counter, last_blink_time, face_landmarks, multiple_faces_detected = process_frame(
            face_mesh, frame, left_eye_idxs, right_eye_idxs, ear_history, blink_counter, total_blinks, last_blink_time
        )

        # Check if 15 seconds have passed without any change
        if time.time() - last_change_time > 15:
            print("Timeout: No change detected for 15 seconds. Returning False.")
            cap.release()
            cv2.destroyAllWindows()
            return False, first_frame  # Timeout due to inactivity, return the first frame

        if multiple_faces_detected:
            # Show message if multiple faces are detected
            cv2.putText(frame, "Multiple Faces Detected. Pausing...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif face_landmarks is None:  # No face detected
            cv2.putText(frame, "No Face Detected", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            if face_landmarks:
                # Draw blue box around the detected face
                h, w, _ = frame.shape
                face_bbox = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in range(468)]
                x_min = min([p[0] for p in face_bbox])
                x_max = max([p[0] for p in face_bbox])
                y_min = min([p[1] for p in face_bbox])
                y_max = max([p[1] for p in face_bbox])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

                # Check head tilt
                tilt_direction = detect_head_tilt(face_landmarks)
                cv2.putText(frame, f'Tilt Direction: {tilt_direction}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f'Match Tilt: {random_direction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check if tilt direction matches the random direction
                if random_direction in tilt_direction:
                    consecutive_correct_tilt += 1
                    if consecutive_correct_tilt >= 2:  # Require two consecutive correct tilts
                        matched_count += 1
                        consecutive_correct_tilt = 0  # Reset the counter after a match
                        random_direction = generate_random_direction()  # Generate new direction after match
                        cv2.putText(frame, "Correct Tilt!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        last_change_time = time.time()  # Reset last change time after correct tilt
                else:
                    consecutive_correct_tilt = 0  # Reset if the direction does not match

            # Display total blinks
            cv2.putText(frame, f"Blinks: {total_blinks}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update last change time if a blink occurs
            if total_blinks > 0 and total_blinks > matched_count:
                last_change_time = time.time()

        # Update the Streamlit placeholder with the current frame
        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        

        # Return True and the first frame when both blink count is 3 or more, and head tilt matches twice or more
        if total_blinks >= 3 and matched_count >= 2:
            print("Both blink and tilt criteria satisfied. Returning True.")
            cap.release()
            cv2.destroyAllWindows()
            return True, first_frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return False, first_frame  # Default return False and the first frame if no criteria met



        