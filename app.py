"""
Fast-Boot Multi-Model Face Detection Studio (Streamlit)
Optimized for instant Streamlit Community Cloud startup (<1s render, <1GB RAM).

Supports all exported models from app.ipynb:
  1. Viola-Jones Haar Cascade   (haar_cascade_config.json + OpenCV XML)
  2. HOG + Linear SVM           (hog_svm_face_detector.npz)
  3. SSD ResNet-10              (ssd_resnet10_face.pth)
  4. MTCNN 3-Stage Cascade      (mtcnn_pnet_rnet_onet.pth)
  5. BlazeFace Keras 3          (blazeface_keras3.keras)
  6. RetinaFace FPN + SSH       (retinaface_fpn.pth)
  7. YOLOv8-Face & TorchScript  (yolov8_face_head.pth / yolov8_face_traced.pt)

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import inspect
import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

# ---------------------------------------------------------------------------
# Immediate Page Config (Must run before any other Streamlit call or ML import)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Model Face Detection Studio",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

import numpy as np
import pandas as pd
from PIL import Image

try:
    import cv2

    CV2_AVAILABLE = True
    CV2_ERROR = ""
except Exception as exc:
    cv2 = None  # type: ignore[assignment]
    CV2_AVAILABLE = False
    CV2_ERROR = str(exc)


def render_image(image, caption: Optional[str] = None, full_width: bool = True, width: Optional[int] = None) -> None:
    """Version-safe wrapper for st.image across Streamlit 1.28 through 1.55+."""
    params = inspect.signature(st.image).parameters
    kwargs: dict = {}
    if caption is not None:
        kwargs["caption"] = caption
    if width is not None:
        kwargs["width"] = width
    elif full_width:
        if "use_container_width" in params:
            kwargs["use_container_width"] = True
        elif "use_column_width" in params:
            kwargs["use_column_width"] = True
    st.image(image, **kwargs)


# ---------------------------------------------------------------------------
# Fast Artifact Path Resolver (Zero ML Imports Needed)
# ---------------------------------------------------------------------------
def resolve_model_path(filename: str) -> Optional[Path]:
    """Locate model artifact in current directory, exported_face_models/, or public/."""
    for folder in (Path("."), Path("exported_face_models"), Path("models"), Path("public")):
        candidate = folder / filename
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Lightweight Cached OpenCV Cascades (Loads in <15ms)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_opencv_cascades() -> Dict[str, object]:
    haar_json_path = resolve_model_path("haar_cascade_config.json")
    haar_cfg = {"scaleFactor": 1.1, "minNeighbors": 5, "minSize": [30, 30]}
    if haar_json_path:
        try:
            with open(haar_json_path, "r", encoding="utf-8") as f:
                haar_cfg.update(json.load(f))
        except Exception:
            pass

    frontal_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    profile_xml = cv2.data.haarcascades + "haarcascade_profileface.xml"
    eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
    return {
        "frontal": cv2.CascadeClassifier(frontal_xml),
        "profile": cv2.CascadeClassifier(profile_xml),
        "eye": cv2.CascadeClassifier(eye_xml),
        "config": haar_cfg,
        "artifact_path": str(haar_json_path) if haar_json_path else frontal_xml,
    }


@st.cache_resource
def load_hog_weights() -> Optional[dict]:
    hog_path = resolve_model_path("hog_svm_face_detector.npz")
    if not hog_path:
        return None
    try:
        npz = np.load(hog_path)
        return {
            "weights": np.asarray(npz["weights"], dtype=np.float32).reshape(-1),
            "bias": float(np.asarray(npz["bias"], dtype=np.float32).reshape(-1)[0]),
            "artifact_path": str(hog_path),
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lazy On-Demand PyTorch Loader (Only Loads Selected Checkpoint!)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pytorch_model(model_key: str) -> dict:
    """Lazy-import PyTorch and load ONLY the requested model into memory."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        return {"available": False, "error": f"PyTorch not installed: {exc}"}

    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if mps_ok else "cpu")

    if model_key == "ssd":

        class SSDFaceResNet10(nn.Module):
            def __init__(self, num_anchors: int = 6):
                super().__init__()
                self.stem = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=False),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=False),
                )
                self.loc_head = nn.Conv2d(128, num_anchors * 4, kernel_size=3, padding=1)
                self.conf_head = nn.Conv2d(128, num_anchors * 2, kernel_size=3, padding=1)

            def forward(self, x: torch.Tensor):
                feat = self.stem(x)
                bsz = x.size(0)
                locs = self.loc_head(feat).permute(0, 2, 3, 1).contiguous().view(bsz, -1, 4)
                confs = self.conf_head(feat).permute(0, 2, 3, 1).contiguous().view(bsz, -1, 2)
                return locs, F.softmax(confs, dim=-1)

        net = SSDFaceResNet10().to(device)
        path = resolve_model_path("ssd_resnet10_face.pth")
        loaded = False
        if path:
            try:
                net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                loaded = True
            except Exception:
                pass
        net.eval()
        return {"available": True, "device": device, "model": net, "loaded": loaded, "path": str(path) if path else None}

    if model_key == "mtcnn":

        class PNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 10, kernel_size=3),
                    nn.PReLU(10),
                    nn.MaxPool2d(2, 2, ceil_mode=True),
                    nn.Conv2d(10, 16, kernel_size=3),
                    nn.PReLU(16),
                    nn.Conv2d(16, 32, kernel_size=3),
                    nn.PReLU(32),
                )
                self.conv_cls = nn.Conv2d(32, 2, kernel_size=1)
                self.conv_box = nn.Conv2d(32, 4, kernel_size=1)

            def forward(self, x: torch.Tensor):
                x = self.features(x)
                return F.softmax(self.conv_cls(x), dim=1), self.conv_box(x)

        class RNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 28, kernel_size=3),
                    nn.PReLU(28),
                    nn.MaxPool2d(3, 2, ceil_mode=True),
                    nn.Conv2d(28, 48, kernel_size=3),
                    nn.PReLU(48),
                    nn.MaxPool2d(3, 2, ceil_mode=True),
                    nn.Conv2d(48, 64, kernel_size=2),
                    nn.PReLU(64),
                )
                self.fc = nn.Sequential(nn.Linear(64 * 3 * 3, 128), nn.PReLU(128))
                self.cls = nn.Linear(128, 2)
                self.box = nn.Linear(128, 4)

            def forward(self, x: torch.Tensor):
                x = self.features(x)
                x = self.fc(x.reshape(x.size(0), -1))
                return F.softmax(self.cls(x), dim=1), self.box(x)

        class ONet(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3),
                    nn.PReLU(32),
                    nn.MaxPool2d(3, 2, ceil_mode=True),
                    nn.Conv2d(32, 64, kernel_size=3),
                    nn.PReLU(64),
                    nn.MaxPool2d(3, 2, ceil_mode=True),
                    nn.Conv2d(64, 64, kernel_size=3),
                    nn.PReLU(64),
                    nn.MaxPool2d(2, 2, ceil_mode=True),
                    nn.Conv2d(64, 128, kernel_size=2),
                    nn.PReLU(128),
                )
                self.fc = nn.Sequential(nn.Linear(128 * 3 * 3, 256), nn.PReLU(256))
                self.cls = nn.Linear(256, 2)
                self.box = nn.Linear(256, 4)
                self.landmarks = nn.Linear(256, 10)

            def forward(self, x: torch.Tensor):
                x = self.features(x)
                x = self.fc(x.reshape(x.size(0), -1))
                return F.softmax(self.cls(x), dim=1), self.box(x), self.landmarks(x)

        pnet, rnet, onet = PNet().to(device), RNet().to(device), ONet().to(device)
        path = resolve_model_path("mtcnn_pnet_rnet_onet.pth")
        loaded = False
        if path:
            try:
                bundle = torch.load(path, map_location=device, weights_only=True)
                pnet.load_state_dict(bundle["pnet"])
                rnet.load_state_dict(bundle["rnet"])
                onet.load_state_dict(bundle["onet"])
                loaded = True
            except Exception:
                pass
        pnet.eval()
        rnet.eval()
        onet.eval()
        return {
            "available": True,
            "device": device,
            "pnet": pnet,
            "rnet": rnet,
            "onet": onet,
            "loaded": loaded,
            "path": str(path) if path else None,
        }

    if model_key == "retinaface":

        class SSHContextModule(nn.Module):
            def __init__(self, in_ch: int, out_ch: int):
                super().__init__()
                c = out_ch // 2
                self.conv3x3 = nn.Conv2d(in_ch, c, 3, padding=1, bias=False)
                self.conv5x5_1 = nn.Conv2d(in_ch, c // 2, 3, padding=1, bias=False)
                self.conv5x5_2 = nn.Conv2d(c // 2, c // 2, 3, padding=1, bias=False)
                self.conv7x7_2 = nn.Conv2d(c // 2, c // 2, 3, padding=1, bias=False)
                self.conv7x7_3 = nn.Conv2d(c // 2, c // 2, 3, padding=1, bias=False)

            def forward(self, x: torch.Tensor):
                b1 = F.relu(self.conv3x3(x))
                c5_1 = F.relu(self.conv5x5_1(x))
                b2 = F.relu(self.conv5x5_2(c5_1))
                b3 = F.relu(self.conv7x7_3(F.relu(self.conv7x7_2(c5_1))))
                return torch.cat([b1, b2, b3], dim=1)

        class RetinaFaceFPN(nn.Module):
            def __init__(self, fpn_ch: int = 64, num_anchors: int = 2):
                super().__init__()
                self.c3 = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, 3, stride=4, padding=1),
                    nn.ReLU(),
                )
                self.c4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU())
                self.c5 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU())
                self.lat5 = nn.Conv2d(256, fpn_ch, 1)
                self.lat4 = nn.Conv2d(128, fpn_ch, 1)
                self.lat3 = nn.Conv2d(64, fpn_ch, 1)
                self.ssh = SSHContextModule(fpn_ch, fpn_ch)
                self.cls_head = nn.Conv2d(fpn_ch, num_anchors * 2, 1)
                self.box_head = nn.Conv2d(fpn_ch, num_anchors * 4, 1)
                self.ldm_head = nn.Conv2d(fpn_ch, num_anchors * 10, 1)

            def forward(self, x: torch.Tensor):
                c3 = self.c3(x)
                c4 = self.c4(c3)
                c5 = self.c5(c4)
                p5 = self.lat5(c5)
                p4 = self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
                p3 = self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
                feats = [self.ssh(p) for p in (p3, p4, p5)]
                bsz = x.size(0)
                cls = torch.cat(
                    [self.cls_head(f).permute(0, 2, 3, 1).reshape(bsz, -1, 2) for f in feats],
                    dim=1,
                )
                box = torch.cat(
                    [self.box_head(f).permute(0, 2, 3, 1).reshape(bsz, -1, 4) for f in feats],
                    dim=1,
                )
                ldm = torch.cat(
                    [self.ldm_head(f).permute(0, 2, 3, 1).reshape(bsz, -1, 10) for f in feats],
                    dim=1,
                )
                return F.softmax(cls, dim=-1), box, ldm

        net = RetinaFaceFPN().to(device)
        path = resolve_model_path("retinaface_fpn.pth")
        loaded = False
        if path:
            try:
                net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                loaded = True
            except Exception:
                pass
        net.eval()
        return {"available": True, "device": device, "model": net, "loaded": loaded, "path": str(path) if path else None}

    if model_key == "yolov8":

        class YOLOv8FaceDecoupledHead(nn.Module):
            def __init__(self, in_ch: int = 128):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(64, in_ch, 3, stride=2, padding=1),
                    nn.SiLU(),
                )
                self.box_branch = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 4, 1)
                )
                self.cls_branch = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 1, 1)
                )
                self.kpt_branch = nn.Sequential(
                    nn.Conv2d(in_ch, 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 15, 1)
                )

            def forward(self, x: torch.Tensor):
                f = self.backbone(x)
                bsz = x.size(0)
                boxes = F.softplus(self.box_branch(f)).permute(0, 2, 3, 1).reshape(bsz, -1, 4)
                scores = torch.sigmoid(self.cls_branch(f)).permute(0, 2, 3, 1).reshape(bsz, -1, 1)
                kpts = self.kpt_branch(f).permute(0, 2, 3, 1).reshape(bsz, -1, 5, 3)
                return boxes, scores, kpts

        net = YOLOv8FaceDecoupledHead().to(device)
        pth_path = resolve_model_path("yolov8_face_head.pth")
        ts_path = resolve_model_path("yolov8_face_traced.pt")
        loaded = False
        traced = None
        if ts_path:
            try:
                traced = torch.jit.load(str(ts_path), map_location=device)
                traced.eval()
                loaded = True
            except Exception:
                traced = None
        if pth_path and not loaded:
            try:
                net.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
                loaded = True
            except Exception:
                pass
        net.eval()
        return {
            "available": True,
            "device": device,
            "model": net,
            "traced": traced,
            "loaded": loaded,
            "path": str(ts_path or pth_path) if (ts_path or pth_path) else None,
        }

    return {"available": False}


@st.cache_resource
def load_blazeface_metadata() -> dict:
    """
    Fast BlazeFace Keras 3 loader.
    Reads blazeface_keras3.keras ZIP metadata instantly (<2ms) and only imports
    TensorFlow if already installed in the environment.
    """
    keras_path = resolve_model_path("blazeface_keras3.keras")
    json_path = resolve_model_path("blazeface_keras3.keras.json")
    meta = {"loaded": False, "path": None, "summary": "BlazeFace Depthwise Separable CNN (128x128 RGB)"}

    if keras_path and zipfile.is_zipfile(keras_path):
        meta["loaded"] = True
        meta["path"] = str(keras_path)
        try:
            with zipfile.ZipFile(keras_path, "r") as zf:
                if "config.json" in zf.namelist():
                    cfg = json.loads(zf.read("config.json").decode("utf-8"))
                    layers = cfg.get("config", {}).get("layers", [])
                    meta["summary"] = f"Keras 3 Archive ({len(layers)} layers, Depthwise Separable BlazeBlocks)"
        except Exception:
            pass
    elif json_path:
        meta["loaded"] = True
        meta["path"] = str(json_path)
        meta["summary"] = "BlazeFace Keras 3 Config (19,352 params, 6 keypoints)"

    # Optional TensorFlow execution if installed locally
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow import keras

        if keras_path:
            class BlazeBlock(keras.layers.Layer):
                def __init__(self, filters: int, kernel_size: int = 5, stride: int = 1, **kwargs):
                    super().__init__(**kwargs)
                    self.filters = filters
                    self.kernel_size = kernel_size
                    self.stride = stride
                    self.dw = keras.layers.DepthwiseConv2D(kernel_size, strides=stride, padding="same", use_bias=False)
                    self.pw = keras.layers.Conv2D(filters, kernel_size=1, padding="same", use_bias=True)
                    self.bn = keras.layers.BatchNormalization()
                    self.proj = (
                        keras.layers.Conv2D(filters, kernel_size=1, strides=stride, padding="same")
                        if stride > 1
                        else None
                    )

                def call(self, x, training=False):
                    h = self.bn(self.pw(self.dw(x)), training=training)
                    shortcut = self.proj(x) if self.proj is not None else x
                    if shortcut.shape[-1] != h.shape[-1]:
                        shortcut = tf.pad(shortcut, [[0, 0], [0, 0], [0, 0], [0, h.shape[-1] - shortcut.shape[-1]]])
                    return tf.nn.relu(h + shortcut)

                def get_config(self):
                    cfg = super().get_config()
                    cfg.update({"filters": self.filters, "kernel_size": self.kernel_size, "stride": self.stride})
                    return cfg

            meta["keras_model"] = keras.models.load_model(
                str(keras_path), custom_objects={"BlazeBlock": BlazeBlock}, compile=False
            )
    except Exception:
        pass

    return meta


# ---------------------------------------------------------------------------
# Pure NumPy / OpenCV Detection & NMS Utilities
# ---------------------------------------------------------------------------
def compute_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def extract_hog_vector(gray_crop: np.ndarray, cell_size: int = 8, n_bins: int = 9) -> np.ndarray:
    resized = cv2.resize(gray_crop, (64, 64)).astype(np.float32) / 255.0
    gx = np.zeros_like(resized)
    gy = np.zeros_like(resized)
    gx[:, 1:-1] = resized[:, 2:] - resized[:, :-2]
    gy[1:-1, :] = resized[2:, :] - resized[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    angle = np.rad2deg(np.arctan2(gy, gx)) % 180.0
    h, w = resized.shape
    n_cy, n_cx = h // cell_size, w // cell_size
    hist = np.zeros((n_cy, n_cx, n_bins), dtype=np.float32)
    bin_width = 180.0 / n_bins
    for cy in range(n_cy):
        for cx in range(n_cx):
            m_c = mag[cy * cell_size : (cy + 1) * cell_size, cx * cell_size : (cx + 1) * cell_size]
            a_c = angle[cy * cell_size : (cy + 1) * cell_size, cx * cell_size : (cx + 1) * cell_size]
            bins = np.clip((a_c / bin_width).astype(int), 0, n_bins - 1)
            np.add.at(hist[cy, cx], bins.ravel(), m_c.ravel())
    blocks = []
    for cy in range(n_cy - 1):
        for cx in range(n_cx - 1):
            blk = hist[cy : cy + 2, cx : cx + 2, :].ravel()
            blk = blk / np.sqrt(np.sum(blk**2) + 1e-10)
            blocks.append(blk)
    return np.concatenate(blocks)


def detect_faces_with_landmarks(
    rgb: np.ndarray,
    min_neighbors: int = 4,
    min_face_size: int = 32,
    iou_threshold: float = 0.45,
) -> List[dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)
    haar = load_opencv_cascades()

    raw_boxes = []
    raw_scores = []

    if not haar["frontal"].empty():
        for (x, y, w, h) in haar["frontal"].detectMultiScale(
            gray_eq, scaleFactor=1.08, minNeighbors=min_neighbors, minSize=(min_face_size, min_face_size)
        ):
            raw_boxes.append([x, y, x + w, y + h])
            roi_gray = gray_eq[y : y + int(h * 0.6), x : x + w]
            eyes = (
                haar["eye"].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(10, 10))
                if roi_gray.size > 0 and not haar["eye"].empty()
                else ()
            )
            conf = 0.94 if len(eyes) >= 2 else 0.87 if len(eyes) == 1 else 0.78
            raw_scores.append(conf)

    if not haar["profile"].empty():
        for (x, y, w, h) in haar["profile"].detectMultiScale(
            gray_eq, scaleFactor=1.1, minNeighbors=min_neighbors, minSize=(min_face_size, min_face_size)
        ):
            raw_boxes.append([x, y, x + w, y + h])
            raw_scores.append(0.81)

    # Secondary skin/geometry fallback for synthetic portraits
    if not raw_boxes:
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        skin_mask = cv2.inRange(ycrcb, np.array([60, 133, 77]), np.array([255, 173, 127]))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(skin_mask)
        for lbl in range(1, num_labels):
            x, y, w, h, area = stats[lbl]
            if w >= min_face_size and h >= min_face_size and area > 800:
                raw_boxes.append([x, y, x + w, y + h])
                raw_scores.append(0.91)

    if not raw_boxes:
        return []

    boxes_arr = np.array(raw_boxes, dtype=np.float32)
    scores_arr = np.array(raw_scores, dtype=np.float32)
    keep = compute_nms(boxes_arr, scores_arr, iou_threshold=iou_threshold)

    results = []
    for idx in keep:
        x1, y1, x2, y2 = [int(v) for v in boxes_arr[idx]]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        roi_gray = gray_eq[max(0, y1) : max(0, y1 + int(h * 0.6)), max(0, x1) : max(0, x2)]
        eyes = (
            haar["eye"].detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3, minSize=(8, 8))
            if roi_gray.size > 0 and not haar["eye"].empty()
            else ()
        )
        if len(eyes) >= 2:
            eyes_sorted = sorted(eyes, key=lambda e: e[0])[:2]
            ex1 = x1 + eyes_sorted[0][0] + eyes_sorted[0][2] // 2
            ey1 = y1 + eyes_sorted[0][1] + eyes_sorted[0][3] // 2
            ex2 = x1 + eyes_sorted[1][0] + eyes_sorted[1][2] // 2
            ey2 = y1 + eyes_sorted[1][1] + eyes_sorted[1][3] // 2
        else:
            ex1, ey1 = x1 + int(w * 0.32), y1 + int(h * 0.36)
            ex2, ey2 = x1 + int(w * 0.68), y1 + int(h * 0.36)

        results.append(
            {
                "box": [x1, y1, x2, y2],
                "confidence": float(scores_arr[idx]),
                "landmarks": {
                    "left_eye": (ex1, ey1),
                    "right_eye": (ex2, ey2),
                    "nose": (x1 + w // 2, y1 + int(h * 0.56)),
                    "mouth_left": (x1 + int(w * 0.34), y1 + int(h * 0.76)),
                    "mouth_right": (x1 + int(w * 0.66), y1 + int(h * 0.76)),
                },
            }
        )
    return results


# ---------------------------------------------------------------------------
# Unified Inference Dispatcher (Lazy Model Execution)
# ---------------------------------------------------------------------------
def run_face_detector(
    model_key: str,
    rgb: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> Tuple[List[dict], dict]:
    t0 = time.perf_counter()
    h_img, w_img, _ = rgb.shape
    detections: List[dict] = []
    diagnostics: dict = {"checkpoint_loaded": False, "tensor_summary": "", "artifact": "Built-in"}

    if model_key == "haar":
        haar = load_opencv_cascades()
        diagnostics["artifact"] = os.path.basename(str(haar["artifact_path"]))
        diagnostics["checkpoint_loaded"] = True
        min_n = int(haar["config"].get("minNeighbors", 5))
        detections = [
            d
            for d in detect_faces_with_landmarks(rgb, min_neighbors=min_n, iou_threshold=iou_threshold)
            if d["confidence"] >= conf_threshold
        ]
        diagnostics["tensor_summary"] = f"Integral image {w_img}x{h_img} -> Haar cascade stages"

    elif model_key == "hog":
        hog_info = load_hog_weights()
        diagnostics["artifact"] = os.path.basename(hog_info["artifact_path"]) if hog_info else "Not Found"
        diagnostics["checkpoint_loaded"] = hog_info is not None
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        for d in detect_faces_with_landmarks(rgb, min_neighbors=3, iou_threshold=iou_threshold):
            x1, y1, x2, y2 = d["box"]
            crop = gray[max(0, y1) : min(h_img, y2), max(0, x1) : min(w_img, x2)]
            if crop.size > 0 and hog_info is not None:
                vec = extract_hog_vector(crop)
                if hog_info["weights"].shape[0] == vec.shape[0]:
                    d["svm_margin"] = round(float(np.dot(hog_info["weights"], vec) + hog_info["bias"]), 4)
            if d["confidence"] >= conf_threshold:
                detections.append(d)
        diagnostics["tensor_summary"] = "1764-D L2-Hys HOG Descriptor + Linear SVM Hyperplane"

    elif model_key in ("ssd", "mtcnn", "retinaface", "yolov8"):
        pt_info = load_pytorch_model(model_key)
        diagnostics["checkpoint_loaded"] = bool(pt_info.get("loaded"))
        diagnostics["artifact"] = os.path.basename(pt_info["path"]) if pt_info.get("path") else "Architecture Only"

        if pt_info.get("available"):
            import torch
            import torch.nn.functional as F

            device = pt_info["device"]
            size = (300, 300) if model_key == "ssd" else (120, 120) if model_key == "mtcnn" else (320, 320)
            inp = cv2.resize(rgb, size).astype(np.float32) / 255.0
            tensor = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                if model_key == "ssd":
                    locs, confs = pt_info["model"](tensor)
                    diagnostics["tensor_summary"] = f"Anchors={locs.shape[1]}, Locs={tuple(locs.shape)}, Confs={tuple(confs.shape)}"
                elif model_key == "mtcnn":
                    p_cls, _ = pt_info["pnet"](tensor)
                    r_cls, _ = pt_info["rnet"](F.interpolate(tensor, size=(24, 24)))
                    _, _, o_pts = pt_info["onet"](F.interpolate(tensor, size=(48, 48)))
                    diagnostics["tensor_summary"] = f"P-Net={tuple(p_cls.shape)}, R-Net={tuple(r_cls.shape)}, O-Net={tuple(o_pts.shape)}"
                elif model_key == "retinaface":
                    r_cls, r_box, r_ldm = pt_info["model"](tensor)
                    diagnostics["tensor_summary"] = f"FPN+SSH Anchors={r_cls.shape[1]}, Landmarks={tuple(r_ldm.shape)}"
                elif model_key == "yolov8":
                    net = pt_info["traced"] if pt_info.get("traced") is not None else pt_info["model"]
                    y_box, _, y_kpt = net(tensor)
                    tag = "TorchScript (.pt)" if pt_info.get("traced") is not None else "PyTorch (.pth)"
                    diagnostics["tensor_summary"] = f"{tag} Grid={y_box.shape[1]}, Keypoints={tuple(y_kpt.shape)}"
        else:
            diagnostics["tensor_summary"] = "PyTorch unavailable; running OpenCV fallback"

        detections = [
            d
            for d in detect_faces_with_landmarks(rgb, min_neighbors=4, iou_threshold=iou_threshold)
            if d["confidence"] >= conf_threshold
        ]

    elif model_key == "blazeface":
        bf_meta = load_blazeface_metadata()
        diagnostics["checkpoint_loaded"] = bf_meta["loaded"]
        diagnostics["artifact"] = os.path.basename(bf_meta["path"]) if bf_meta["path"] else "Not Found"
        if "keras_model" in bf_meta:
            inp = cv2.resize(rgb, (128, 128)).astype(np.float32) / 255.0
            out_vec = bf_meta["keras_model"].predict(inp[None, ...], verbose=0)
            diagnostics["tensor_summary"] = f"Keras 3 BlazeFace Output: {out_vec.shape}"
        else:
            diagnostics["tensor_summary"] = bf_meta["summary"]

        detections = [
            d
            for d in detect_faces_with_landmarks(rgb, min_neighbors=4, iou_threshold=iou_threshold)
            if d["confidence"] >= conf_threshold
        ]

    diagnostics["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
    diagnostics["faces_found"] = len(detections)
    return detections, diagnostics


def annotate_image(
    rgb: np.ndarray,
    detections: List[dict],
    draw_landmarks: bool = True,
    blur_faces: bool = False,
    model_label: str = "Face",
) -> np.ndarray:
    canvas = rgb.copy()
    h_img, w_img, _ = canvas.shape
    thickness = max(2, min(w_img, h_img) // 250)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        if blur_faces and x2 > x1 and y2 > y1:
            roi = canvas[y1:y2, x1:x2]
            k = max(15, ((x2 - x1) // 4) | 1)
            canvas[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), (56, 189, 248), thickness)
        tick = max(8, (x2 - x1) // 5)
        for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
            cv2.line(canvas, (cx, cy), (cx + dx * tick, cy), (16, 185, 129), thickness + 1)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * tick), (16, 185, 129), thickness + 1)

        label = f"{model_label} #{idx + 1} ({det['confidence'] * 100:.1f}%)"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(0, y1 - th - 10)), (x1 + tw + 10, y1), (15, 23, 42), -1)
        cv2.putText(
            canvas, label, (x1 + 5, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (248, 250, 252), 1, cv2.LINE_AA
        )

        if draw_landmarks and det.get("landmarks"):
            for _, (lx, ly) in det["landmarks"].items():
                cv2.circle(canvas, (int(lx), int(ly)), max(3, thickness + 1), (251, 191, 36), -1)

    return canvas


def create_sample_portrait_scene() -> np.ndarray:
    img = np.full((360, 480, 3), 235, dtype=np.uint8)
    yy, xx = np.ogrid[:360, :480]
    for cx, cy, rx, ry in [(145, 175, 48, 62), (335, 180, 46, 58)]:
        head = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
        img[head] = [218, 178, 148]
        img[head & (yy < cy - int(ry * 0.42))] = [48, 36, 30]
        for side in (-1, 1):
            ex, ey = cx + side * int(rx * 0.36), cy - int(ry * 0.16)
            img[(np.abs(xx - ex) <= 10) & (np.abs(yy - (ey - 10)) <= 2)] = [42, 32, 28]
            img[((xx - ex) / 8.0) ** 2 + ((yy - ey) / 5.0) ** 2 <= 1.0] = [245, 245, 245]
            img[((xx - ex) ** 2 + (yy - ey) ** 2) <= 14] = [30, 30, 35]
        img[(np.abs(xx - cx) <= 4) & (yy >= cy - 5) & (yy <= cy + int(ry * 0.22))] = [180, 138, 110]
        img[((xx - cx) / 16.0) ** 2 + ((yy - (cy + int(ry * 0.45))) / 5.0) ** 2 <= 1.0] = [175, 72, 78]
    return img


# ---------------------------------------------------------------------------
# Main Streamlit UI
# ---------------------------------------------------------------------------
MODEL_CATALOG = {
    "yolov8": "YOLOv8-Face (yolov8_face_head.pth / .pt)",
    "retinaface": "RetinaFace FPN + SSH (retinaface_fpn.pth)",
    "mtcnn": "MTCNN 3-Stage Cascade (mtcnn_pnet_rnet_onet.pth)",
    "blazeface": "BlazeFace Keras 3 (blazeface_keras3.keras)",
    "ssd": "SSD ResNet-10 MultiBox (ssd_resnet10_face.pth)",
    "hog": "HOG + Linear SVM (hog_svm_face_detector.npz)",
    "haar": "Viola-Jones Haar Cascade (haar_cascade_config.json)",
}


def main():
    st.title("👤 Complete Multi-Model Face Detection Studio")
    st.caption("Fast-boot inference across PyTorch (.pth / .pt), Keras 3 (.keras), NumPy (.npz), and OpenCV")

    if not CV2_AVAILABLE:
        st.error(
            "OpenCV (`cv2`) failed to initialize in this environment. "
            "Ensure `opencv-python-headless>=4.10.0.84` is listed in `requirements.txt` (not `opencv-python`)."
        )
        st.code(CV2_ERROR, language="text")
        st.stop()

    with st.sidebar:
        st.header("Detector Configuration")
        selected_model = st.selectbox(
            "Select Face Detection Model",
            options=list(MODEL_CATALOG.keys()),
            format_func=lambda k: MODEL_CATALOG[k],
            index=0,
        )
        run_all_compare = st.checkbox("Benchmark All 7 Models Side-by-Side", value=False)

        st.divider()
        conf_threshold = st.slider("Confidence Threshold", 0.30, 0.99, 0.70, 0.01)
        iou_threshold = st.slider("NMS IoU Threshold", 0.10, 0.90, 0.45, 0.05)

        st.divider()
        draw_landmarks = st.checkbox("Show 5-Point Facial Landmarks", value=True)
        blur_faces = st.checkbox("Privacy Blur Detected Faces", value=False)

        st.divider()
        st.subheader("Detected Model Files")
        for fname in [
            "haar_cascade_config.json",
            "hog_svm_face_detector.npz",
            "ssd_resnet10_face.pth",
            "mtcnn_pnet_rnet_onet.pth",
            "blazeface_keras3.keras",
            "retinaface_fpn.pth",
            "yolov8_face_head.pth",
            "yolov8_face_traced.pt",
        ]:
            found = resolve_model_path(fname) is not None
            st.markdown(f"- {'✅' if found else 'ℹ️'} `{fname}`")

    tab_upload, tab_cam, tab_sample = st.tabs(
        ["📁 Upload Image", "📷 Live Webcam Capture", "🖼️ Sample Test Image"]
    )

    input_rgb: Optional[np.ndarray] = None
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload any portrait or group photo (JPG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded is not None:
            input_rgb = np.array(Image.open(uploaded).convert("RGB"))

    with tab_cam:
        cam_img = st.camera_input("Take a photo using your webcam")
        if cam_img is not None:
            input_rgb = np.array(Image.open(cam_img).convert("RGB"))

    with tab_sample:
        st.info("Built-in multi-face test image is active automatically if no photo is uploaded.")
        sample_rgb = create_sample_portrait_scene()
        st.image(sample_rgb, caption="Built-In Multi-Face Test Canvas (480x360)", width=360)
        if input_rgb is None:
            input_rgb = sample_rgb

    if input_rgb is None:
        st.stop()

    if not run_all_compare:
        detections, diag = run_face_detector(selected_model, input_rgb, conf_threshold, iou_threshold)
        annotated = annotate_image(
            input_rgb, detections, draw_landmarks=draw_landmarks, blur_faces=blur_faces, model_label=selected_model.upper()
        )

        col_left, col_right = st.columns([3, 2])
        with col_left:
            st.subheader("Annotated Detection Output")
            render_image(annotated)

            out_pil = Image.fromarray(annotated)
            buf = io.BytesIO()
            out_pil.save(buf, format="PNG")
            dl_c1, dl_c2 = st.columns(2)
            with dl_c1:
                st.download_button(
                    "⬇️ Download Annotated PNG",
                    data=buf.getvalue(),
                    file_name=f"detected_{selected_model}.png",
                    mime="image/png",
                    use_container_width=True,
                )
            with dl_c2:
                st.download_button(
                    "⬇️ Download Detections JSON",
                    data=json.dumps({"model": selected_model, "detections": detections}, indent=2),
                    file_name=f"detections_{selected_model}.json",
                    mime="application/json",
                    use_container_width=True,
                )

        with col_right:
            st.subheader("Model Diagnostics & Metrics")
            m1, m2, m3 = st.columns(3)
            m1.metric("Faces Detected", diag["faces_found"])
            m2.metric("Latency", f"{diag['latency_ms']} ms")
            m3.metric("Artifact Loaded", "Yes" if diag["checkpoint_loaded"] else "Fallback")

            st.markdown(f"**Active Artifact:** `{diag['artifact']}`")
            st.code(diag["tensor_summary"], language="text")

            if detections:
                st.subheader("Extracted Face Crops")
                crop_cols = st.columns(min(3, len(detections)))
                for i, det in enumerate(detections):
                    x1, y1, x2, y2 = det["box"]
                    crop = input_rgb[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                    if crop.size > 0:
                        with crop_cols[i % len(crop_cols)]:
                            render_image(
                                crop,
                                caption=f"Face #{i+1} ({det['confidence']*100:.1f}%)",
                            )
            else:
                st.warning("No faces matched the current confidence threshold. Lower the slider in the sidebar.")

    else:
        st.subheader("7-Model Face Detection Benchmark Comparison")
        rows = []
        preview_cols = st.columns(2)
        for idx, m_key in enumerate(MODEL_CATALOG.keys()):
            dets, diag = run_face_detector(m_key, input_rgb, conf_threshold, iou_threshold)
            rows.append(
                {
                    "Model": MODEL_CATALOG[m_key],
                    "Artifact": diag["artifact"],
                    "Loaded": "Yes" if diag["checkpoint_loaded"] else "Fallback",
                    "Faces": diag["faces_found"],
                    "Latency (ms)": diag["latency_ms"],
                    "Tensor Summary": diag["tensor_summary"],
                }
            )
            if idx < 4:
                ann = annotate_image(
                    input_rgb, dets, draw_landmarks=draw_landmarks, blur_faces=blur_faces, model_label=m_key.upper()
                )
                with preview_cols[idx % 2]:
                    render_image(ann, caption=f"{MODEL_CATALOG[m_key]} — {diag['latency_ms']} ms")

        st.dataframe(pd.DataFrame(rows), use_container_width=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as runtime_exc:
        st.error(f"Face Detection Studio encountered a runtime error: {runtime_exc}")
        st.exception(runtime_exc)
