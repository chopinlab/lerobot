# act_inference.py — 최종본 (장치 경로 고정 방식)
import time
import httpx
import numpy as np
import cv2

PHOSPHOBOT_API_URL = "http://localhost:8020"

# ★ 캡처 이미지 확인 후 배정 — 학습 main 시점 카메라를 MAIN에
CAMERA_MAIN = "/dev/video0"        # 학습 데이터 main과 같은 시점으로
CAMERA_SECONDARY = "/dev/video4"   # secondary_0과 같은 시점으로
# 반대면 두 값을 서로 바꾸세요

from phosphobot.am import ACT

# 카메라를 OpenCV로 직접 열기 (AllCameras 대신 — 인덱스 혼동 원천 차단)
cap_main = cv2.VideoCapture(CAMERA_MAIN)
cap_sec = cv2.VideoCapture(CAMERA_SECONDARY)
time.sleep(1)

def grab(cap):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("카메라 프레임 실패")
    frame = cv2.resize(frame, (320, 240))            # (폭, 높이)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # BGR→RGB

model = ACT(
    model_id="/home/eric/.cache/huggingface/lerobot/outputs/train/act_pick_pig/checkpoints/100000/pretrained_model"
)

try:
    while True:
        images = [grab(cap_main), grab(cap_sec)]     # [main, secondary_0] 순서
        state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()
        actions = model(
            {"state": np.array(state["angles"]), "images": np.array(images)}
        )
        for action in actions:
            httpx.post(
                f"{PHOSPHOBOT_API_URL}/joints/write",
                json={"angles": action[0].tolist()},
            )
            time.sleep(1 / 30)
except KeyboardInterrupt:
    print("중지")
finally:
    cap_main.release()
    cap_sec.release()
