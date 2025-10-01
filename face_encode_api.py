import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
from bson.objectid import ObjectId
from dotenv import load_dotenv
import time

# -------------------------
# Load env vars
# -------------------------
load_dotenv()


# -------------------------
# Init Flask + Mongo
# -------------------------
app = Flask(__name__)

# -------------------------
# Load face detection + embedding model
# -------------------------
face_proto = "deploy.prototxt"
face_model = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

embedder = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx", ""  # download from OpenCV model zoo
)

def get_face_encoding(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                continue

            aligned_face = frame[y1:y2, x1:x2]

            if aligned_face.size == 0:
                continue

            face_feature = embedder.feature(aligned_face)

            # 🔑 Normalize embedding
            norm_feat = cv2.normalize(face_feature, None, alpha=1.0, norm_type=cv2.NORM_L2)
            return norm_feat.flatten().tolist()
    return None

def capture_face_encoding(duration=3):
    """Capture face encodings for `duration` seconds and return average embedding."""
    cap = cv2.VideoCapture(0)
    encodings = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw guide mask (center oval)
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        axes = (w//4, h//3)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 255, 0), 2)

        cv2.putText(frame, "Align your face in the oval", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        encoding = get_face_encoding(frame)
        if encoding:
            encodings.append(encoding)

        cv2.imshow("Face Capture", frame)

        # Stop after duration seconds or Q press
        if time.time() - start_time > duration or cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not encodings:
        return None

    # ✅ Average embeddings for robustness
    return np.mean(np.array(encodings), axis=0).tolist()


# -------------------------
# API: Capture and update user
# -------------------------
@app.route("/capture-face", methods=["POST"])
def capture_face():
    try:
        data = request.json
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"success": False, "message": "User ID required"}), 400

        encoding = capture_face_encoding(duration=3)  # 3s guided capture

        if not encoding:
            return jsonify({"success": False, "message": "No face detected"}), 400

        return jsonify({
            "success": True,
            "message": "Face encoding extracted successfully",
            "userId": user_id,
            "encodingLength": len(encoding),
            "encoding": encoding
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    

# -------------------------
# API: Verify face
# -------------------------
@app.route("/verify-face", methods=["POST"])
def verify_face():
    try:
        data = request.json
        stored_encoding = data.get("encoding")

        if not stored_encoding:
            return jsonify({"success": False, "message": "Stored encoding required"}), 400

        encoding = capture_face_encoding(duration=3)

        if encoding is None:
            return jsonify({"success": False, "message": "No face detected"}), 400

        similarity = embedder.match(
            np.array(stored_encoding, dtype=np.float32),
            np.array(encoding, dtype=np.float32),
            cv2.FaceRecognizerSF_FR_COSINE
        )
        match_percent = round(similarity * 100, 2)

        return jsonify({
            "success": True,
            "match": similarity >= 0.6,
            "similarity": float(similarity),
            "matchPercent": match_percent
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
