import cv2
import mediapipe as mp
import warnings
import torch
import torchvision.transforms as T
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import os
import shutil
import time

from train_age import AgeEstimator
from train_gender import GenderClassifier

SMOOTHING_AGE = 10
SMOOTHING_GENDER = 15

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model_age = AgeEstimator().to(device)
model_gender = GenderClassifier().to(device)

state = torch.load("model_age_checkpoints/age_estimator_best.pth", map_location=device)
model_age.load_state_dict(state)
state = torch.load(
    "model_gender_checkpoints/gender_classifier_best.pth", map_location=device
)
model_gender.load_state_dict(state)

model_age.eval()
model_gender.eval()


transform = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def euclidean(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


warnings.filterwarnings("ignore", category=UserWarning)
CAMERA_INDEX = 1

# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.7)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print(f"❌ Impossible d'accéder à la caméra d'index {CAMERA_INDEX}")
    exit()


faces = {}
next_face_id = 0

MAX_DISTANCE = 80  # pixels
FACE_TIMEOUT = 1.0  # secondes

while True:
    ret, frame = cap.read()
    if not ret:
        print(f"❌ La caméra d'index {CAMERA_INDEX} n'a pas fourni d'image")
        break

    current_time = time.time()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(frame_rgb)

    h, w, _ = frame.shape

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = bbox.xmin * w
            y = bbox.ymin * h
            bw = bbox.width * w
            bh = bbox.height * h

            # Ajustements visage
            bw = bh  # garantuee square box

            x = int(max(0, x))
            y = int(max(0, y))
            bw = int(min(bw, w - x))
            bh = int(min(bh, h - y))

            # Face update
            cx = x + bw // 2
            cy = y + bh // 2
            matched_id = None
            min_dist = float("inf")

            for face_id, data in faces.items():
                dist = euclidean((cx, cy), data["center"])
                if dist < min_dist and dist < MAX_DISTANCE:
                    min_dist = dist
                    matched_id = face_id

            if matched_id is None:
                matched_id = next_face_id
                next_face_id += 1

                faces[matched_id] = {
                    "center": (cx, cy),
                    "ages": [],
                    "genders": [],
                    "last_seen": current_time,
                }

            face_data = faces[matched_id]
            face_data["center"] = (cx, cy)
            face_data["last_seen"] = current_time

            # Crop visage
            face_crop = frame[y : y + bh, x : x + bw]

            if face_crop.size == 0:
                continue

            # Prédiction âge
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)

            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred_age = model_age(face_tensor).item()
                pred_gender = model_gender(face_tensor).item()

            # AGE
            face_data["ages"].append(pred_age)
            if len(face_data["ages"]) >= SMOOTHING_AGE:
                face_data["ages"].pop(0)

            ages_tensor = torch.tensor(face_data["ages"])
            pred_age = ages_tensor.mean().item()
            std = ages_tensor.std(unbiased=False).item()

            # GENDER
            face_data["genders"].append(pred_gender)
            if len(face_data["genders"]) >= SMOOTHING_GENDER:
                face_data["genders"].pop(0)

            logits_average = sum(face_data["genders"]) / len(face_data["genders"])
            pred_gender = "Man" if logits_average > 0 else "Woman"

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

            # Text age
            text = f"Age : {round(pred_age - std)} - {round(pred_age + std)}"
            text_y = y + bh + 45

            if text_y < h:
                cv2.putText(
                    frame,
                    text,
                    (x + bw // 6, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            # Tect gender
            gender_text = f"Gender : {pred_gender} ({logits_average:.2f})"
            gender_text_y = text_y + 45

            if gender_text_y < h:
                cv2.putText(
                    frame,
                    gender_text,
                    (x + bw // 6, gender_text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

    faces = {
        fid: data
        for fid, data in faces.items()
        if current_time - data["last_seen"] < FACE_TIMEOUT
    }

    cv2.imshow("Face Age Gender Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
