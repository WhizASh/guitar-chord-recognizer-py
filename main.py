"""
realtime_chord_recognizer.py
Complete guitar chord recognition system with OpenCV + MediaPipe
Fixed mirroring consistency between calibration and runtime.
"""
import cv2
import collections
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import json
import os
import argparse

# -------------- 1. Constants -----------------
CHORD_FINGERS = {
    "Cmaj": {(3, 3), (2, 2), (1, 1)},           # (string, fret)
    "Emaj": {(5, 2), (4, 2), (3, 1)},
    "Emin": {(5, 2), (4, 2)},
    "Amaj": {(4, 2), (3, 2), (2, 2)},
    "Amin": {(3, 2), (4, 2), (2, 1)},
    "Fmaj": {(4, 3), (3, 2), (2, 1), (1, 1)}    # easy mini-barre version
}

url = "http://192.168.0.103:8080/shot.jpg"

BUFFER_LEN = 4  # frames to smooth
CALIBRATION_FILE = "guitar_calibration.json"
MIRROR = True  # Set to False if you prefer non-mirrored view

# Default calibration (adjust for your setup)
string_y = [180, 200, 220, 240, 260, 280]  # 6 guitar strings (top to bottom)
fret_x = [120, 160, 200, 240, 280, 320, 360]  # fret positions (left to right)

# -------------- 2. Utility Functions -----------------
def maybe_flip(frame):
    """Apply consistent mirroring to frame if enabled."""
    return cv2.flip(frame, 1) if MIRROR else frame

# -------------- 3. Calibration Functions -----------------
def save_calibration(string_y, fret_x, filename=CALIBRATION_FILE):
    """Save calibration data to JSON file."""
    data = {"string_y": string_y, "fret_x": fret_x}
    with open(filename, 'w') as f:
        json.dump(data, f)
    print(f"Calibration saved to {filename}")

def load_calibration(filename=CALIBRATION_FILE):
    """Load calibration data from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        return data["string_y"], data["fret_x"]
    return None, None

def collect_points(title, n_points, instruction):
    """Interactive calibration - click points on screen with consistent mirroring."""
    pts = []
    def onclick(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            print(f"Point {len(pts)}: ({x}, {y})")
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(title)
    cv2.setMouseCallback(title, onclick)
    
    print(f"\n{instruction}")
    print(f"Click {n_points} points. Press ESC when done.")
    
    while len(pts) < n_points:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply consistent mirroring during calibration
        frame = maybe_flip(frame)
        
        # Show existing points
        for i, (px, py) in enumerate(pts):
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (px+10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Points: {len(pts)}/{n_points}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(title, frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cv2.destroyWindow(title)
    cap.release()
    
    if 'string' in title.lower():
        return [p[1] for p in sorted(pts, key=lambda p: p[1])]  # Sort by Y
    else:
        return [p[0] for p in sorted(pts, key=lambda p: p[0])]  # Sort by X

def calibrate_guitar():
    """Complete calibration routine."""
    print("=== Guitar Calibration ===")
    print("Position your guitar so the neck is clearly visible in the camera.")
    input("Press Enter when ready...")
    
    # Calibrate strings
    string_y = collect_points("Calibrate Strings", 6, 
                             "Click on each guitar string from TOP to BOTTOM (low E to high E)")
    
    # Calibrate frets
    fret_x = collect_points("Calibrate Frets", 7,
                           "Click on fret lines from LEFT to RIGHT (nut, 1st fret, 2nd fret, etc.)")
    
    save_calibration(string_y, fret_x)
    return string_y, fret_x

# -------------- 4. Helper Functions -----------------
def draw_guitar_neck(frame, string_y, fret_x):
    """Overlay guitar neck grid on the video frame."""
    h, w, _ = frame.shape
    
    # Draw strings (horizontal lines)
    for i, y in enumerate(string_y):
        cv2.line(frame, (0, int(y)), (w, int(y)), (0, 255, 255), 1)  # Yellow
        cv2.putText(frame, f"S{i+1}", (5, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    
    # Draw frets (vertical lines)
    for i, x in enumerate(fret_x):
        cv2.line(frame, (int(x), 0), (int(x), h), (255, 0, 255), 1)  # Magenta
        cv2.putText(frame, f"F{i}", (int(x), 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

def draw_hand_landmarks(frame, landmarks):
    """Draw hand landmarks with fingertips highlighted."""
    h, w, _ = frame.shape
    
    # Draw all landmarks
    for idx, landmark in enumerate(landmarks):
        px, py = int(landmark.x * w), int(landmark.y * h)
        
        # Highlight fingertips in red, others in blue
        if idx in [4, 8, 12, 16, 20]:  # Fingertips
            cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)  # Red
            cv2.putText(frame, str(idx), (px+8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        else:
            cv2.circle(frame, (px, py), 3, (255, 0, 0), -1)  # Blue

def chord_from_fingers(fingers):
    """Determine chord from detected finger positions."""
    best_match = "Unknown"
    max_matches = 0
    
    for name, required in CHORD_FINGERS.items():
        matches = len(required.intersection(fingers))
        if matches > max_matches and matches >= len(required) * 0.7:  # 70% match threshold
            max_matches = matches
            best_match = name
    
    return best_match

def draw_debug_info(frame, fingers, smoothed, label, finger_hist):
    """Draw debugging information on frame."""
    y_offset = 140
    cv2.putText(frame, f"Raw fingers: {len(fingers)}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, f"Smoothed: {len(smoothed)}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y_offset += 20
    cv2.putText(frame, f"History: {len(finger_hist)}", (10, y_offset), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# -------------- 5. Main Application -----------------
def main():
    global string_y, fret_x, MIRROR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Guitar Chord Recognition System')
    parser.add_argument('--no-mirror', action='store_true', 
                       help='Disable mirroring (use raw camera view)')
    args = parser.parse_args()
    
    if args.no_mirror:
        MIRROR = False
    
    print("Guitar Chord Recognition System")
    print(f"Mirror mode: {'ON' if MIRROR else 'OFF'}")
    print("1. Use existing calibration")
    print("2. Calibrate guitar position")
    choice = input("Choose option (1/2): ").strip()
    
    if choice == "2":
        string_y, fret_x = calibrate_guitar()
    else:
        # Try to load existing calibration
        loaded_strings, loaded_frets = load_calibration()
        if loaded_strings and loaded_frets:
            string_y, fret_x = loaded_strings, loaded_frets
            print("Loaded existing calibration")
        else:
            print("No calibration found, using defaults")
    
    # Initialize MediaPipe
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    VisionRunningMode = vision.RunningMode
    
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=1,
        running_mode=VisionRunningMode.VIDEO
    )
    
    try:
        hand_landmarker = HandLandmarker.create_from_options(options)
    except FileNotFoundError:
        print("Error: hand_landmarker.task file not found!")
        print("Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        return
    
    # Initialize camera and tracking
    cap = cv2.VideoCapture(0)
    finger_hist = collections.deque(maxlen=BUFFER_LEN)
    
    print("\nStarting chord recognition...")
    print("Controls:")
    print("- ESC: Quit")
    print("- 'c': Recalibrate")
    print("- 'r': Reset history")
    print("- 'm': Toggle mirror mode")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply consistent mirroring
        frame = maybe_flip(frame)
        
        # Process with MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        result = hand_landmarker.detect_for_video(mp_image, timestamp)
        
        fingers = set()
        
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            h, w, _ = frame.shape
            
            # Draw hand landmarks
            draw_hand_landmarks(frame, lm)
            
            # Detect finger positions
            for idx in [4, 8, 12, 16, 20]:  # Fingertip landmarks
                px, py = int(lm[idx].x * w), int(lm[idx].y * h)
                
                # Find closest string and fret
                if string_y and fret_x:
                    s = min(range(len(string_y)), key=lambda i: abs(py - string_y[i]))
                    f = max([j for j, x in enumerate(fret_x) if px >= x], default=0)
                    fingers.add((s + 1, f))  # 1-indexed strings
            
            finger_hist.append(fingers)
        
        # Apply smoothing
        if finger_hist:
            # Use intersection of recent frames for stability
            smoothed = set.intersection(*list(finger_hist)[-min(2, len(finger_hist)):])
        else:
            smoothed = set()
        
        # Recognize chord
        label = chord_from_fingers(smoothed)
        
        # Draw guitar neck overlay
        draw_guitar_neck(frame, string_y, fret_x)
        
        # Draw main chord label
        color = (0, 255, 0) if label != "Unknown" else (0, 0, 255)
        cv2.putText(frame, f"Chord: {label}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Draw debug information
        draw_debug_info(frame, fingers, smoothed, label, finger_hist)
        
        # Show confidence indicator
        confidence = len(smoothed) / 4.0 * 100  # Rough confidence based on fingers detected
        cv2.putText(frame, f"Confidence: {confidence:.0f}%", (10, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show mirror status
        mirror_text = "MIRROR ON" if MIRROR else "MIRROR OFF"
        cv2.putText(frame, mirror_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        cv2.imshow("Guitar Chord Recognizer", frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('c'):  # Calibrate
            cv2.destroyAllWindows()
            string_y, fret_x = calibrate_guitar()
        elif key == ord('r'):  # Reset
            finger_hist.clear()
            print("History reset")
        elif key == ord('m'):  # Toggle mirror
            MIRROR = not MIRROR
            print(f"Mirror mode: {'ON' if MIRROR else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
