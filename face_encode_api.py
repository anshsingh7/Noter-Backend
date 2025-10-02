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
    """
    Capture embeddings per pose.
    Returns: (encodings_by_pose: dict, average_encoding: 1D np.array or None)
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    window_name = "Face Capture"
    if show_window:
        _create_capture_window(window_name, x=50, y=20)

    encodings_by_pose = {}

    # mapping for target rotation angle (visual only)
    pose_to_angle = {"center": 0, "left": -90, "right": 90}

    try:
        for pose in poses:
            start_time = time.time()
            embeddings = []

            direction_text = {
                "center": "Look straight",
                "left": "Turn your head LEFT slowly",
                "right": "Turn your head RIGHT slowly",
            }.get(pose, "Look straight")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed = time.time() - start_time
                if elapsed > duration_per_pose:
                    break

                # animate angle from 0 -> target
                start_angle = 0
                target_angle = pose_to_angle.get(pose, 0)
                angle = float(np.interp(elapsed, [0, duration_per_pose], [start_angle, target_angle]))

                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                axes = (w // 5, h // 3)

                # Draw rotating ellipse mask
                cv2.ellipse(frame, center, axes, angle, 0, 360, (0, 255, 0), 2)

                # Instruction text
                cv2.putText(frame, f"Pose: {pose.upper()} ({direction_text})", (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Small guidance for user
                cv2.putText(frame, "Align your face inside the oval", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 200), 1)

                # Attempt to extract encoding for this frame
                encoding = get_face_encoding(frame)
                if encoding is not None:
                    embeddings.append(encoding)

                    # show small indicator dot when detected
                    cv2.circle(frame, (w-30, 30), 8, (0, 255, 0), -1)

                if show_window:
                    cv2.imshow(window_name, frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    # user aborted
                    break

            # End of pose capture
            if embeddings:
                encodings_by_pose[pose] = np.mean(np.stack(embeddings, axis=0), axis=0)
            else:
                encodings_by_pose[pose] = None

            # short pause between poses
            time.sleep(0.3)

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    # compute overall average from all available pose embeddings
    available = [v for v in encodings_by_pose.values() if v is not None]
    overall = None
    if available:
        overall = np.mean(np.stack(available, axis=0), axis=0)

    # Convert numpy arrays to python lists for JSON serialization at return
    encodings_by_pose_lists = {}
    for k, v in encodings_by_pose.items():
        encodings_by_pose_lists[k] = v.tolist() if v is not None else None

    return encodings_by_pose_lists, (overall.tolist() if overall is not None else None)


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
@app.route("/verify-face", methods=["POST"])
def verify_face():
    """
    Accepts:
      - either { encoding: [...], threshold: 0.8 }  (legacy / single encoding)
      - or { encodingsByPose: { center: [...], left: [...], right: [...] }, threshold: 0.8 }
    Behavior:
      - Capture live embeddings for a few seconds (default 4s)
      - If encodingsByPose provided: for each stored pose compute the max similarity against any captured frame
        then require that ALL stored poses have max similarity >= threshold (i.e. matched all aspects).
      - If a single encoding provided: compare the average captured embedding to stored encoding.
    Returns detailed per-pose similarity and overall pass/fail.
    """
    try:
        data = request.json or {}
        threshold = float(data.get("threshold", 0.8))

        # Accept multiple field names for compatibility
        stored_enc = data.get("encoding") or data.get("faceEncodedData")
        stored_by_pose = data.get("encodingsByPose") or data.get("encodings_by_pose")

        # If stored_by_pose is not present but stored_enc looks like an object with pose keys, accept it
        if not stored_by_pose and isinstance(stored_enc, dict):
            stored_by_pose = stored_enc
            stored_enc = None

        # Ensure we have something
        if stored_by_pose is None and stored_enc is None:
            return jsonify({"success": False, "message": "Stored encoding or encodingsByPose required"}), 400

        # Capture live embeddings for verification
        captured_list = capture_live_embeddings(duration=float(data.get("verifyDuration", 4.0)), show_window=bool(data.get("showWindow", True)))
        if not captured_list:
            return jsonify({"success": False, "message": "No face detected during verification"}), 400

        # Convert captured embeddings to numpy array list
        captured_np = [np.array(e, dtype=np.float32) for e in captured_list]

        # Helper to compute similarity between two embeddings
        def sim(a, b):
            try:
                return float(embedder.match(np.array(a, dtype=np.float32), np.array(b, dtype=np.float32), cv2.FaceRecognizerSF_FR_COSINE))
            except Exception as e:
                print("sim error:", e)
                return 0.0

        if stored_by_pose:
            results = {}
            max_sims_per_pose = []
            for pose_name, stored_arr in stored_by_pose.items():
                if not stored_arr:
                    results[pose_name] = {"present": False, "maxSimilarity": 0.0, "match": False, "matchPercent": 0.0}
                    max_sims_per_pose.append(0.0)
                    continue

                stored_np = np.array(stored_arr, dtype=np.float32)
                # compute max similarity between stored pose embedding and any captured embedding
                sims = [sim(stored_np, c) for c in captured_np]
                max_sim = max(sims) if sims else 0.0
                match = max_sim >= threshold
                results[pose_name] = {
                    "present": True,
                    "maxSimilarity": float(max_sim),
                    "match": bool(match),
                    "matchPercent": round(max_sim * 100.0, 2)
                }
                max_sims_per_pose.append(max_sim)

            # Overall: require all pose matches (the strict "match every aspect" policy)
            overall_match = bool(len(max_sims_per_pose) > 0 and (min(max_sims_per_pose) >= threshold))
            overall_percent = round(min(max_sims_per_pose) * 100.0, 2) if max_sims_per_pose else 0.0

            return jsonify({
                "success": True,
                "overallMatch": overall_match,
                "overallMatchPercent": overall_percent,
                "perPose": results,
                "capturedFrames": len(captured_np)
            })

        else:
            # legacy single encoding compare to average captured embedding
            stored_np = np.array(stored_enc, dtype=np.float32)
            avg_captured = np.mean(np.stack(captured_np, axis=0), axis=0)
            similarity = sim(stored_np, avg_captured)
            match = similarity >= threshold
            return jsonify({
                "success": True,
                "match": bool(match),
                "similarity": float(similarity),
                "matchPercent": round(similarity * 100.0, 2),
                "capturedFrames": len(captured_np)
            })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# -------------------------
# Run Flask
# -------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)

