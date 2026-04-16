import os
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


class FaceEngine:
    """
    High-accuracy face detection + recognition engine (YuNet + SFace).

    - Detection: YuNet ONNX via `cv2.FaceDetectorYN`.
    - Recognition: SFace ONNX via `cv2.FaceRecognizerSF`.
    - Enrollment: loads known faces from `known_faces/<name>/*.(jpg|png|jpeg)` and builds an embedding gallery.
    - Overlays: draws boxes and (if matched) the person name + similarity score.
    - Labels: emits `FACE_DETECTED`, `FACE_PRIMARY`, `FACE_LOST`, `FACE_KNOWN`, `FACE_KNOWN_<name>`, `FACE_UNKNOWN`.
    """

    def __init__(
        self,
        model_dir: str,
        known_faces_dir: Optional[str] = None,
        grace_window_sec: float = 0.5,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        cosine_threshold: float = 0.363,
        recognize_every_n_frames: int = 10,
        primary_reco_iou_gate: float = 0.7,
    ):
        self._model_dir = model_dir
        self._known_faces_dir = known_faces_dir

        det_model = os.path.join(model_dir, "face_detection_yunet_2023mar.onnx")
        rec_model = os.path.join(model_dir, "face_recognition_sface_2021dec.onnx")
        if not os.path.exists(det_model):
            raise FileNotFoundError(f"Missing YuNet model: {det_model}")
        if not os.path.exists(rec_model):
            raise FileNotFoundError(f"Missing SFace model: {rec_model}")

        # Input size must be set to the current frame size; initialize with a safe default.
        self._detector = cv2.FaceDetectorYN_create(
            det_model,
            "",
            (320, 320),
            float(score_threshold),
            float(nms_threshold),
            int(top_k),
        )
        self._recognizer = cv2.FaceRecognizerSF_create(rec_model, "")

        self._cosine_threshold = float(cosine_threshold)
        self._grace_window_sec = float(grace_window_sec)
        self._recognize_every_n_frames = max(1, int(recognize_every_n_frames))
        self._primary_reco_iou_gate = float(primary_reco_iou_gate)

        self._last_primary_bbox: Optional[Tuple[int, int, int, int]] = None  # x,y,w,h
        self._last_seen_ts: Optional[float] = None
        self._frame_idx = 0

        # Cached recognition result for the primary face.
        self._primary_name: Optional[str] = None
        self._primary_sim: Optional[float] = None
        self._primary_last_reco_bbox: Optional[Tuple[int, int, int, int]] = None

        self._gallery: Dict[str, np.ndarray] = {}
        self._load_gallery()

    def _iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        ax2, ay2 = ax + aw, ay + ah
        bx2, by2 = bx + bw, by + bh

        ix1, iy1 = max(ax, bx), max(ay, by)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        union = (aw * ah) + (bw * bh) - inter
        return float(inter) / float(union + 1e-12)

    def _load_gallery(self) -> None:
        if not self._known_faces_dir:
            return
        if not os.path.isdir(self._known_faces_dir):
            return

        exts = {".jpg", ".jpeg", ".png", ".webp"}
        for name in os.listdir(self._known_faces_dir):
            person_dir = os.path.join(self._known_faces_dir, name)
            if not os.path.isdir(person_dir):
                continue

            feats: List[np.ndarray] = []
            for fname in os.listdir(person_dir):
                _, ext = os.path.splitext(fname.lower())
                if ext not in exts:
                    continue
                img_path = os.path.join(person_dir, fname)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                feat = self._extract_feature_from_image(img)
                if feat is not None:
                    feats.append(feat)

            if feats:
                # Normalize mean embedding
                mean_feat = np.mean(np.vstack(feats), axis=0, keepdims=True).astype(np.float32)
                mean_feat = mean_feat / (np.linalg.norm(mean_feat) + 1e-12)
                self._gallery[name] = mean_feat

    def set_recognize_every_n_frames(self, n: int) -> None:
        self._recognize_every_n_frames = max(1, int(n))

    def reload_face_models(self) -> None:
        """
        Reload known-face gallery from disk and clear cached recognition state.

        (The ONNX model files are loaded once at init; this is intended for
        reloading `known_faces/` content without restarting Jarvis.)
        """
        self._gallery.clear()
        self._load_gallery()
        self._primary_name = None
        self._primary_sim = None
        self._primary_last_reco_bbox = None

    def _extract_feature_from_image(self, img_bgr) -> Optional[np.ndarray]:
        # Limit image size to avoid detector issues with very large images
        max_dim = 1000
        h, w, _ = img_bgr.shape
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ih, iw, _ = img_bgr.shape
        self._detector.setInputSize((iw, ih))
        _, faces = self._detector.detect(img_bgr)
        if faces is None or len(faces) == 0:
            return None

        # Pick largest face
        best = max(faces, key=lambda f: float(f[2] * f[3]))
        aligned = self._recognizer.alignCrop(img_bgr, best)
        feat = self._recognizer.feature(aligned)
        feat = feat.astype(np.float32)
        feat = feat / (np.linalg.norm(feat) + 1e-12)
        return feat

    def _match_name(self, feat: np.ndarray) -> Tuple[Optional[str], float]:
        best_name = None
        best_sim = -1.0
        for name, gal in self._gallery.items():
            sim = float(self._recognizer.match(feat, gal, cv2.FaceRecognizerSF_FR_COSINE))
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name is None:
            return None, -1.0
        if best_sim >= self._cosine_threshold:
            return best_name, best_sim
        return None, best_sim

    def process(self, frame_bgr, mp_image=None) -> Set[str]:
        h, w, _ = frame_bgr.shape
        now = time.time()
        self._frame_idx += 1
        self._detector.setInputSize((w, h))

        labels: Set[str] = set()
        _, faces = self._detector.detect(frame_bgr)
        faces_arr = faces if faces is not None else np.zeros((0, 15), dtype=np.float32)
        face_count = int(faces_arr.shape[0])

        if face_count > 0:
            labels.add("FACE_DETECTED")
            labels.add("FACE_PRIMARY")

            # Select primary face by area * confidence
            def _key(f) -> float:
                x, y, bw, bh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4])
                return (bw * bh) + (score * 1000.0)

            primary = max(faces_arr, key=_key)
            x, y, bw, bh = map(int, primary[:4])
            self._last_primary_bbox = (x, y, bw, bh)
            self._last_seen_ts = now

            cx = x + bw / 2.0
            if cx < w * 0.33:
                labels.add("FACE_LEFT")
            elif cx > w * 0.66:
                labels.add("FACE_RIGHT")
            else:
                labels.add("FACE_CENTER")

            # Draw all face boxes (cheap).
            for face in faces_arr:
                fx, fy, fbw, fbh = map(int, face[:4])
                score = float(face[4])
                x1, y1 = max(fx, 0), max(fy, 0)
                x2, y2 = min(fx + fbw, w - 1), min(fy + fbh, h - 1)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(
                    frame_bgr,
                    f"{score:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

            # Recognition (expensive): only primary face, only every N frames,
            # and re-run sooner if the primary bbox moved significantly.
            did_reco = False
            if self._gallery and self._last_primary_bbox is not None:
                if self._frame_idx % self._recognize_every_n_frames == 0:
                    did_reco = True
                elif self._primary_last_reco_bbox is None:
                    did_reco = True
                else:
                    if self._iou(self._last_primary_bbox, self._primary_last_reco_bbox) < self._primary_reco_iou_gate:
                        did_reco = True

            if did_reco and self._gallery:
                aligned = self._recognizer.alignCrop(frame_bgr, primary)
                feat = self._recognizer.feature(aligned).astype(np.float32)
                feat = feat / (np.linalg.norm(feat) + 1e-12)
                name, sim = self._match_name(feat)
                self._primary_last_reco_bbox = self._last_primary_bbox
                self._primary_name = name
                self._primary_sim = float(sim)

            # Apply cached recognition result to labels + overlay (primary face only).
            if self._gallery and self._last_primary_bbox is not None:
                px, py, pbw, pbh = self._last_primary_bbox
                px1, py1 = max(px, 0), max(py, 0)
                px2, py2 = min(px + pbw, w - 1), min(py + pbh, h - 1)

                if self._primary_name is not None:
                    labels.add("FACE_KNOWN")
                    labels.add(f"FACE_KNOWN_{self._primary_name}")
                    sim = self._primary_sim if self._primary_sim is not None else 0.0
                    text = f"{self._primary_name} ({sim:.2f})"
                    color = (0, 255, 0)
                else:
                    labels.add("FACE_UNKNOWN")
                    sim = self._primary_sim if self._primary_sim is not None else -1.0
                    text = f"Unknown ({sim:.2f})" if sim >= 0 else "Unknown"
                    color = (0, 0, 255)

                cv2.putText(
                    frame_bgr,
                    text,
                    (px1, min(py2 + 25, h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
        else:
            # Lightweight tracking: keep last bbox for a small grace window.
            if self._last_seen_ts is not None and (now - self._last_seen_ts) <= self._grace_window_sec:
                labels.add("FACE_PRIMARY")
            elif self._last_seen_ts is not None:
                labels.add("FACE_LOST")
                self._last_primary_bbox = None
                self._last_seen_ts = None
                self._primary_name = None
                self._primary_sim = None
                self._primary_last_reco_bbox = None

        cv2.putText(
            frame_bgr,
            f"Faces: {face_count} Known: {len(self._gallery)}",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        if self._last_primary_bbox is not None:
            x, y, bw, bh = self._last_primary_bbox
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w - 1), min(y + bh, h - 1)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(
                frame_bgr,
                (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                4,
                (0, 255, 255),
                -1,
            )

        return labels

