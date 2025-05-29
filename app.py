import cv2
import dlib
import numpy as np
import streamlit as st
import tempfile
import pandas as pd
import time
from scipy.spatial import distance as dist
from threading import Thread
import platform

if platform.system() == 'Windows':
    import winsound

def play_alert_sound():
    if platform.system() == 'Windows':
        winsound.Beep(1000, 500)
    else:
        pass

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[13], mouth[19])
    B = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[15], mouth[17])
    D = dist.euclidean(mouth[12], mouth[16])
    mar = (A + B + C) / (3.0 * D)
    return mar

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.6
FATIGUE_FRAME_LIMIT = 15

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

st.title("🚨 Fatigue Detection Web App")

mode = st.radio("Choose input mode:", ("Upload Video", "Use Webcam"))

fatigue_log = []
fatigue_start_time = None
fatigue_frames_count = 0
total_fatigue_time = 0

def process_frame(frame):
    global fatigue_frames_count, fatigue_start_time, total_fatigue_time
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    fatigue_detected = False

    for rect in rects:
        shape = predictor(gray, rect)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        leftEye = shape_np[36:42]
        rightEye = shape_np[42:48]
        mouth = shape_np[48:68]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        mar = mouth_aspect_ratio(mouth)

        for (x, y) in np.concatenate((leftEye, rightEye, mouth)):
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        if ear < EYE_AR_THRESH or mar > MOUTH_AR_THRESH:
            fatigue_frames_count += 1
            if fatigue_frames_count == 1:
                fatigue_start_time = time.time()
            if fatigue_frames_count >= FATIGUE_FRAME_LIMIT:
                fatigue_detected = True
                if fatigue_start_time is not None:
                    elapsed = time.time() - fatigue_start_time
                    total_fatigue_time += elapsed
                    fatigue_start_time = time.time()
                fatigue_log.append({
                    "timestamp": time.strftime('%H:%M:%S'),
                    "ear": round(ear, 3),
                    "mar": round(mar, 3),
                    "fatigue": True
                })
        else:
            if fatigue_frames_count >= FATIGUE_FRAME_LIMIT:
                fatigue_frames_count = 0
                fatigue_start_time = None
            else:
                fatigue_frames_count = 0
                fatigue_start_time = None

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if fatigue_detected:
            cv2.putText(frame, "FATIGUE ALERT!", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            Thread(target=play_alert_sound).start()

    return frame, fatigue_detected, ear, mar

if mode == "Upload Video":
    uploaded_file = st.file_uploader("Upload a video file (mp4, avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_num += 1
            processed_frame, fatigue_detected, ear, mar = process_frame(frame)
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            progress_bar.progress(min(frame_num / frame_count, 1.0))

        cap.release()

        st.success(f"Total fatigue events detected: {len(fatigue_log)}")
        st.write(f"Approximate total fatigue time: {int(total_fatigue_time)} seconds")

        if fatigue_log:
            df = pd.DataFrame(fatigue_log)
            st.subheader("Fatigue Log")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download fatigue log as CSV", data=csv, file_name="fatigue_log.csv")

elif mode == "Use Webcam":
    run = st.checkbox('Start Webcam')
    stframe = st.empty()

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to grab frame")
                break

            processed_frame, fatigue_detected, ear, mar = process_frame(frame)
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cap.release()
