# --- PHASE 2: ENCODE FACES (after photo capture) ---
detector  = cv2.FaceDetectorYN.create(DETECTOR_MODEL, "", (320,320), 0.8, 0.3, 5000)
recognizer = cv2.FaceRecognizerSF.create(RECOGNIZER_MODEL, "")

known_feats = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        img = cv2.imread(os.path.join(KNOWN_FACES_DIR, filename))
        if img is None: continue

        height, width, _ = img.shape
        detector.setInputSize((width, height))
        _, faces = detector.detect(img)

        if faces is not None:
            aligned_face = recognizer.alignCrop(img, faces[0])
            feat         = recognizer.feature(aligned_face)
            known_feats.append(feat)
            # Extract clean name (strip '_N' suffix from filename)
            base_name = os.path.splitext(filename)[0]
            known_names.append(base_name.split('_')[0])

# Serialize to pickle
data = {"feats": known_feats, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print(f"SUCCESS! Database updated: {len(known_names)} embeddings stored.")