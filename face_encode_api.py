import os
import cv2
import numpy as np
from flask import Flask, jsonify, request
from dotenv import load_dotenv
import time

# -------------------------
# Load env vars
# -------------------------
load_dotenv()

# -------------------------
# Init Flask
# -------------------------
app = Flask(__name__)

# -------------------------
# Load face detection + embedding model
# -------------------------
face_proto = "deploy.prototxt"
face_model = "res10_300x300_ssd_iter_140000.caffemodel"
face_net = cv2.dnn.readNetFromCaffe(face_proto, face_model)

# OpenCV SF embedder (from model zoo)
embedder = cv2.FaceRecognizerSF.create(
    "face_recognition_sface_2021dec.onnx", ""
)

def draw_face_mask(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)

    # Face oval
    axes = (w // 3, h // 3)
    mask = np.zeros_like(frame, dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, (0, 200, 0), -1)

    # Ears (two circles on each side of oval)
    ear_radius = 30
    left_ear = (center[0] - axes[0] - ear_radius // 2, center[1] - 40)
    right_ear = (center[0] + axes[0] + ear_radius // 2, center[1] - 40)
    cv2.circle(mask, left_ear, ear_radius, (0, 200, 0), -1)
    cv2.circle(mask, right_ear, ear_radius, (0, 200, 0), -1)

    # Apply opaque mask (blend 40% transparent)
    alpha = 0.4
    frame[:] = cv2.addWeighted(frame, 1, mask, alpha, 0)

def get_face_encoding(frame):
    """Return 1D normalized embedding (list) for the first detected face in frame, else None."""
    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                # clamp
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                if x2 - x1 < 20 or y2 - y1 < 20:
                    continue

                aligned_face = frame[y1:y2, x1:x2]

                if aligned_face.size == 0:
                    continue

                # Some OpenCV versions require resize; embedder often accepts various sizes, but safe to resize:
                aligned_face_resized = cv2.resize(aligned_face, (112, 112))

                face_feature = embedder.feature(aligned_face_resized)

                # Normalize embedding
                norm_feat = cv2.normalize(
                    face_feature, None, alpha=1.0, norm_type=cv2.NORM_L2
                )
                return norm_feat.flatten()
    except Exception as e:
        print("get_face_encoding error:", e)
    return None


def _create_capture_window(name="Face Capture", x=50, y=20):
    """Create window and attempt to move it to top of screen (best-effort)."""
    try:
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, 640, 480)
        cv2.moveWindow(name, x, y)
    except Exception:
        # headless server or not supported
        pass


def capture_pose_sequence(poses=("center", "left", "right"), duration_per_pose=3, show_window=True):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    encodings_by_pose = {}
    all_embeddings = []

    window_name = "Face Capture"
    if show_window:
        _create_capture_window(window_name, x=50, y=20)

    try:
        for pose in poses:
            start_time = time.time()
            embeddings = []

            direction_text = {
                "center": "Look straight",
                "left": "Turn LEFT slowly",
                "right": "Turn RIGHT slowly",
            }.get(pose, "Look straight")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                elapsed = time.time() - start_time
                if elapsed > duration_per_pose:
                    break

                # Draw face "shape mask" (see part 2 below 👇)
                draw_face_mask(frame)

                cv2.putText(frame, f"Pose: {pose.upper()} - {direction_text}", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                encoding = get_face_encoding(frame)
                if encoding is not None:
                    embeddings.append(encoding)
                    all_embeddings.append(encoding)

                if show_window:
                    cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            encodings_by_pose[pose] = [e.tolist() for e in embeddings] if embeddings else []

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    # ✅ Compute overall embedding from all embeddings
    overall = None
    if all_embeddings:
        overall = np.mean(np.array(all_embeddings, dtype=np.float32), axis=0).tolist()

    return encodings_by_pose, overall




def capture_live_embeddings(duration=4, show_window=True):
    """
    Capture embeddings during continuous capture for `duration` seconds and return list of embeddings found.
    Useful for verification (returns multiple embeddings per frame).
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    window_name = "Face Verify"
    if show_window:
        _create_capture_window(window_name, x=50, y=20)

    embeddings = []
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            elapsed = time.time() - start_time
            if elapsed > duration:
                break

            # draw static mask
            h, w = frame.shape[:2]
            center = (w // 2, h // 2)
            axes = (w // 5, h // 3)
            cv2.ellipse(frame, center, axes, 0, 0, 360, (0, 200, 0), 2)
            cv2.putText(frame, "Verifying... keep your face in the oval", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            encoding = get_face_encoding(frame)
            if encoding is not None:
                embeddings.append(encoding)
                cv2.circle(frame, (w-30, 30), 8, (0, 255, 0), -1)

            if show_window:
                cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    return embeddings  # list of 1D np arrays


# -------------------------
# API: Capture and update user (multi-pose enrollment)
# -------------------------
@app.route("/capture-face", methods=["POST"])
def capture_face():
    """
    Request body: { userId: "<id>", poses: ["center","left","right"] (optional), durationPerPose: 3 (optional) }
    Response includes:
      { success: true, encoding: [...], encodingsByPose: { center: [...], left: [...], right: [...] } }
    """
    try:
        data = request.json or {}
        user_id = data.get("userId")
        if not user_id:
            return jsonify({"success": False, "message": "User ID required"}), 400

        poses = data.get("poses", ["center", "left", "right"])
        duration_per_pose = float(data.get("durationPerPose", 3.0))
        show_window = bool(data.get("showWindow", True))

        enc_by_pose, overall = capture_pose_sequence(poses=poses, duration_per_pose=duration_per_pose, show_window=show_window)
        
        # Pick an "overall" encoding (for backward compatibility)
        overall = None
        all_embeddings = []
        for plist in enc_by_pose.values():
            all_embeddings.extend(plist)
        if all_embeddings:
            # take average embedding across all poses
            overall = np.mean(np.array(all_embeddings, dtype=np.float32), axis=0).tolist()

        if overall is None:
            return jsonify({"success": False, "message": "No face detected during enrollment"}), 400

        return jsonify({
            "success": True,
            "message": "Face encodings captured (multi-pose).",
            "userId": user_id,
            "encoding": overall,
            "encodingsByPose": enc_by_pose
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------------
# API: Verify face
# -------------------------
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def sim(a, b):
    a_norm = normalize_vector(np.array(a, dtype=np.float32))
    b_norm = normalize_vector(np.array(b, dtype=np.float32))
    try:
        return float(embedder.match(a_norm, b_norm, cv2.FaceRecognizerSF_FR_COSINE))
    except Exception as e:
        print("sim error:", e)
        return 0.0

@app.route("/verify-face", methods=["POST"])
def verify_face():
    try:
        data = request.json or {}
        threshold = float(data.get("threshold", 0.55))  # lower threshold for more tolerance

        stored_enc = data.get("encoding") or data.get("faceEncodedData")
        stored_by_pose = data.get("encodingsByPose") or data.get("encodings_by_pose")
        if not stored_by_pose and isinstance(stored_enc, dict):
            stored_by_pose = stored_enc
            stored_enc = None
        if stored_by_pose is None and stored_enc is None:
            return jsonify({"success": False, "message": "Stored encoding required"}), 400

        captured_list = capture_live_embeddings(duration=float(data.get("verifyDuration", 4.0)), show_window=True)
        if not captured_list:
            return jsonify({"success": False, "message": "No face detected"}), 400

        captured_np = [np.array(e, dtype=np.float32) for e in captured_list]

        if stored_by_pose:
            results = {}
            overall_match = True  # Require all poses to match

            for pose_name, pose_embeddings in stored_by_pose.items():
                if not pose_embeddings:
                    results[pose_name] = {"match": False, "matchPercent": 0, "maxSimilarity": 0}
                    overall_match = False
                    continue

                max_sim = 0
                for stored_arr in pose_embeddings:
                    stored_np = np.array(stored_arr, dtype=np.float32)
                    sims = [sim(stored_np, c) for c in captured_np]
                    if sims:
                        max_sim = max(max_sim, max(sims))

                results[pose_name] = {
                    "match": max_sim >= threshold,
                    "matchPercent": round(max_sim * 100, 2),
                    "maxSimilarity": round(max_sim, 4)
                }
                if max_sim < threshold:
                    overall_match = False

            return jsonify({
                "success": True,
                "overallMatch": overall_match,
                "perPose": results,
                "capturedFrames": len(captured_np)
            })
        else:
            stored_np = np.array(stored_enc, dtype=np.float32)
            avg_captured = np.mean(np.stack(captured_np, axis=0), axis=0)
            similarity = sim(stored_np, avg_captured)
            match = similarity >= threshold
            return jsonify({
                "success": True,
                "match": bool(match),
                "similarity": round(similarity, 4),
                "matchPercent": round(similarity*100,2),
                "capturedFrames": len(captured_np)
            })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500



# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)

