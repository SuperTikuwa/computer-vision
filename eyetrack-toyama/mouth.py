
import cv2
import dlib
import pyautogui
import numpy as np

# 目のアスペクト比を計算する関数


def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# 口のアスペクト比を計算する関数


def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[3] - mouth[9])  # 上唇の幅
    B = np.linalg.norm(mouth[2] - mouth[10])  # 下唇の幅
    C = np.linalg.norm(mouth[0] - mouth[6])  # 口の高さ
    mar = (A + B) / (2.0 * C)
    return mar


# EARのしきい値と瞬きのフレーム数
EAR_THRESHOLD = 0.3
EAR_CONSEC_FRAMES = 3

# MAR（口のアスペクト比）のしきい値と口を開けるフレーム数
MAR_THRESHOLD = 0.7
MOUTH_CONSEC_FRAMES = 5

# 瞬きカウンタ、口を開けるカウンタ、総瞬き数、総口開け数
blink_counter = 0
mouth_open_counter = 0
total_blinks = 0
total_mouth_opens = 0

window_width = 1280
window_height = 720

# システムのフルスクリーン解像度を取得
screen_width, screen_height = pyautogui.size()

# Dlibの顔検出器と目検出器を初期化
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)  # カメラ画像の横幅を1280に設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)  # カメラ画像の縦幅を720に設定

# ウィンドウを中央に配置するための座標を計算
position_x = int(screen_width / 2 - window_width / 2)
position_y = int(screen_height / 2 - window_height / 2)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)  # フレームを左右反転

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 目のランドマークを取得
        leftEye = np.array(
            [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)])
        rightEye = np.array(
            [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)])
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 両目の平均EARを計算
        ear = (leftEAR + rightEAR) / 2.0

        # EARがしきい値以下であれば瞬きカウンタを増やす
        if ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            # 瞬きカウンタが連続フレーム数以上であれば瞬きと見なし、クリックイベントを実行
            if blink_counter >= EAR_CONSEC_FRAMES:
                total_blinks += 1
                pyautogui.click()
            blink_counter = 0

        # 口のランドマークを取得
        mouth = np.array(
            [(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])
        mar = mouth_aspect_ratio(mouth)

        # MARがしきい値以上であれば口開けカウンタを増やす
        if mar > MAR_THRESHOLD:
            mouth_open_counter += 1
        else:
            # 口開けカウンタが連続フレーム数以上であれば口開けと見なし、クリックイベントを実行
            if mouth_open_counter >= MOUTH_CONSEC_FRAMES:
                total_mouth_opens += 1
                pyautogui.click()
            mouth_open_counter = 0

        # 顔の中心位置を取得してマウスを動かす
        face_center_x = face.center().x
        face_center_y = face.center().y

        if face_center_y < 200:
            pyautogui.moveRel(0, -10, duration=0.1)

        pyautogui.moveTo(
            face_center_x * screen_width / frame.shape[1],
            face_center_y * screen_height / frame.shape[0],
            duration=0.1
        )

    cv2.moveWindow("Frame", position_x, position_y)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
