import cv2
import dlib
from imutils import face_utils
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("predictor/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)


def eye_trim(frame, landmark) -> map:
    r_x1, r_y1 = landmark[36][0], landmark[36][1]
    r_x2, r_y2 = landmark[37][0], landmark[37][1]
    r_x3, r_y3 = landmark[39][0], landmark[39][1]
    r_x4, r_y4 = landmark[40][0], landmark[40][1]

    l_x1, l_y1 = landmark[42][0], landmark[42][1]
    l_x2, l_y2 = landmark[43][0], landmark[43][1]
    l_x3, l_y3 = landmark[45][0], landmark[45][1]
    l_x4, l_y4 = landmark[46][0], landmark[46][1]

    trim_val = 2
    r_frame_trim = frame[r_y2-trim_val:r_y4+trim_val, r_x1:r_x3]
    l_frame_trim = frame[l_y2-trim_val:l_y4+trim_val, l_x1:l_x3]

    r_height,r_width = r_frame_trim.shape[0],r_frame_trim.shape[1]
    l_height,l_width = l_frame_trim.shape[0],l_frame_trim.shape[1]

    r_frame_trim_resize = cv2.resize(r_frame_trim , (int(r_width*7.0), int(r_height*7.0)))
    l_frame_trim_resize = cv2.resize(l_frame_trim , (int(l_width*7.0), int(l_height*7.0)))
    r_frame_gray = cv2.cvtColor(r_frame_trim_resize, cv2.COLOR_BGR2GRAY)
    l_frame_gray = cv2.cvtColor(l_frame_trim_resize, cv2.COLOR_BGR2GRAY)

    r_frame_gray = cv2.GaussianBlur(r_frame_gray,(7,7),0)
    l_frame_gray = cv2.GaussianBlur(l_frame_gray,(7,7),0)

    thresh = 20
    maxval = 255
    e_th,r_frame_bin = cv2.threshold(r_frame_gray,thresh,maxval,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)
    l_th,l_frame_bin = cv2.threshold(l_frame_gray,thresh,maxval,cv2.THRESH_OTSU|cv2.THRESH_BINARY_INV)

    return {"r_frame_trim_resize":r_frame_trim_resize,"l_frame_trim_resize":l_frame_trim_resize,"r_frame_bin":r_frame_bin,"l_frame_bin":l_frame_bin,"r_x1":r_x1,"r_y2":r_y2,"l_x1":l_x1,"l_y2":l_y2}




while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray,1)

    if len(faces) == 0:
        print("No face detected")
        continue

    for face in faces:
        landmark = predictor(gray, face)
        landmark = face_utils.shape_to_np(landmark)

        eye = eye_trim(frame, landmark)

        r_eye_contours, _ = cv2.findContours(eye["r_frame_bin"], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        r_eye_contours = sorted(r_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True) #輪郭が一番大きい順に並べる
        if(len(r_eye_contours)==0):
            print("Right Blink")
        # else:
        #     for cnt in r_eye_contours:
        #         (x, y, w, h) = cv2.boundingRect(cnt)
        #         cv2.circle(eye["r_frame_trim_resize"], (int(x+w/2), int(y+h/2)), int((w+h)/4), (255, 0, 0), 2)
        #         cv2.circle(frame, (int(eye["r_x1"]+(x+w)/10), int(eye["r_y2"]-3+(y+h)/10)), int((w+h)/20), (0, 255, 0), 1)
        #         break

        l_eye_contours, _ = cv2.findContours(eye["l_frame_bin"], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        l_eye_contours = sorted(l_eye_contours, key=lambda x: cv2.contourArea(x), reverse=True)
        if(len(l_eye_contours)==0):
            print("Left Blink")
        # else:
        #     for cnt in l_eye_contours:
        #         (x, y, w, h) = cv2.boundingRect(cnt)
        #         cv2.circle(eye["l_frame_trim_resize"], (int(x+w/2), int(y+h/2)), int((w+h)/4), (255, 0, 0), 2)
        #         cv2.circle(frame, (int(eye["l_x1"]+(x+w)/10), int(eye["l_y2"]-3+(y+h)/10)), int((w+h)/20), (0, 255, 0), 1)
        #         break


        for (i,(x,y)) in enumerate(landmark):
            if i < 36 or i > 47:
                continue
            cv2.circle(frame, (x,y), 1, (0,0,255), -1)

    cv2.imshow('frame', frame)
    cv2.imshow("right eye trim",eye["r_frame_trim_resize"])
    cv2.imshow("left eye trim",eye["l_frame_trim_resize"])

    cv2.imshow("right eye black white",eye["r_frame_bin"])
    cv2.imshow("left eye black white",eye["l_frame_bin"])

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
