import cv2
import numpy as np
import os
import time
import pickle
from picamera2 import Picamera2

# --- CONFIGURATION ---
ENCODINGS_FILE      = 'encodings.pickle'
DETECTOR_MODEL      = 'face_detection_yunet_2022mar.onnx'
RECOGNIZER_MODEL    = 'face_recognition_sface_2021dec.onnx'
CONFIDENCE_THRESHOLD = 0.8    # YuNet detection confidence
MATCH_THRESHOLD      = 0.363  # SFace cosine similarity threshold

# Load AI Models
detector  = cv2.FaceDetectorYN.create(
    DETECTOR_MODEL, "", (320, 320), CONFIDENCE_THRESHOLD, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

# Load pre-computed embeddings database
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.loads(f.read())
known_feats = data["feats"]
known_names = data["names"]
print(f"Loaded {len(known_names)} known face embeddings.")

# Setup Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()
time.sleep(2)  # Warm-up

print("Running. Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame = cv2.flip(frame, 1)       # Mirror for UX

    height, width, _ = frame.shape
    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)   # YuNet inference

    if faces is not None:
        for face in faces:
            box           = face[0:4].astype(int)
            aligned_face  = recognizer.alignCrop(frame, face)
            feat          = recognizer.feature(aligned_face)

            name      = "Unknown"
            max_score = 0.0

            for i, known_feat in enumerate(known_feats):
                score = recognizer.match(
                    known_feat, feat, cv2.FaceRecognizerSF_FR_COSINE)
                if score > MATCH_THRESHOLD and score > max_score:
                    max_score = score
                    name = known_names[i]

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{name} ({int(max_score*100)}%)"
            cv2.rectangle(frame, (x, y+h-28), (x+w, y+h), color, cv2.FILLED)
            cv2.putText(frame, label, (x+4, y+h-6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)

    cv2.imshow('Pi 4 Face Recognition System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()