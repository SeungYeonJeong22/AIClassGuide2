from flask import Flask, render_template, Response
import cv2
import asyncio
import websockets
import pickle
import os
from threading import Thread
from dotenv import load_dotenv
from emotion_analysis import analyze_emotion
import logging
import time

load_dotenv('environments.env')

app = Flask(__name__)

# 서버 설정
WEBSOCKET_HOST = os.getenv('WEBSOCKET_HOST', 'localhost')
WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_PORT', 8000))

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 학생 페이지 라우팅
@app.route('/')
def home():
    return render_template('student.html')

# 학생 비디오 스트리밍 처리
def generate_student_video():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open video capture device")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame from video capture device")
            time.sleep(1)  # 잠시 대기 후 다시 시도
            continue  # 프레임을 잡을 수 없으면 계속 반복

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        # HTML에서 비디오를 표시하기 위한 MIME 타입 설정
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

    cap.release()

# 학생 비디오 피드 라우트
@app.route('/student_video_feed')
def student_video_feed():
    return Response(generate_student_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket을 통한 비디오 및 감정 분석 결과 전송
async def send_student_video():
    uri = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"
    logging.info(f"Attempting to connect to WebSocket at {uri}")
    cap = cv2.VideoCapture(0)

    try:
        async with websockets.connect(uri) as websocket:
            log_flag = False
            logging.info("Connected to WebSocket")
            while True:
                if not cap.isOpened():
                    logging.error("Video capture device is not opened. Retrying...")
                    cap = cv2.VideoCapture(0)  # 장치 다시 열기 시도
                    time.sleep(1)
                    continue  # 비디오 장치가 열리지 않으면 재시도

                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to grab frame. Retrying...")
                    time.sleep(1)  # 실패한 경우 1초 대기 후 다시 시도
                    continue

                # 감정 분석 수행
                result, texts, border_color = analyze_emotion(frame)

                # 프레임과 감정 데이터를 WebSocket으로 전송
                _, buffer = cv2.imencode('.jpg', frame)
                frame_data = buffer.tobytes()

                data = {
                    "frame": frame_data,
                    "emotion": result['emotion'],
                    "dominant_emotion": result['dominant_emotion']
                }
                try:
                    await websocket.send(pickle.dumps(data))
                except Exception as e:
                    if not log_flag:
                        logging.info("Waiting for student connection")
                        log_flag = True

                await asyncio.sleep(0.1)

    except Exception as e:
        logging.error(f"WebSocket connection error: {e}")

    finally:
        cap.release()

# WebSocket 비디오 전송을 비동기로 실행
def start_sending_video():
    asyncio.run(send_student_video())

# WebSocket 비디오 전송을 위한 스레드 시작
if __name__ == "__main__":
    video_thread = Thread(target=start_sending_video)
    video_thread.start()
    app.run(host='0.0.0.0', port=5001)