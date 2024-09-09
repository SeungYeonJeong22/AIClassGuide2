import cv2
import asyncio
import websockets
import pickle
import numpy as np
from flask import Flask, render_template, Response
from threading import Thread
from dotenv import load_dotenv
from plotting import plot_emotion_history
import logging
import os

load_dotenv('environments.env')

app = Flask(__name__)

# 서버 설정
WEBSOCKET_HOST = os.getenv('WEBSOCKET_HOST', 'localhost')
WEBSOCKET_PORT = int(os.getenv('WEBSOCKET_PORT', 8000))

# 전역 변수 정의
connected_clients = set()
emotion_history = []
time_history = []
emotion_values_history = []
start_time_offset = 0
student_frame = None
student_emotion = {}  # Initialize as an empty dictionary

# 로깅 설정
logging.basicConfig(level=logging.INFO)

# 웹 페이지 라우팅
@app.route('/')
def index():
    return render_template('index.html')

# 선생님 비디오 스트리밍 처리
def generate_teacher_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_data = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
    cap.release()

# 선생님 비디오 피드 라우트
@app.route('/teacher_video_feed')
def teacher_video_feed():
    return Response(generate_teacher_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 학생 비디오 및 감정 분석 데이터를 수신
async def receive_student_video(websocket):
    global student_frame, student_emotion, emotion_history, time_history, emotion_values_history, start_time_offset

    while True:
        try:
            data = await websocket.recv()
            data = pickle.loads(data)

            frame_data = data["frame"]
            emotion_data = data["emotion"]
            dominant_emotion = data["dominant_emotion"]

            time_history.append(len(time_history))  # 단순 시간
            emotion_history.append(dominant_emotion)
            emotion_values_history.append(emotion_data)

            student_emotion = emotion_data
            nparr = np.frombuffer(frame_data, np.uint8)
            student_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        except Exception as e:
            logging.error(f"Error receiving student video: {e}")
            break

# 감정 분석 결과를 프레임에 오버레이하는 함수
def overlay_text_on_frame(frame, texts, border_color):
    overlay = frame.copy()
    alpha = 0.6  # 투명도 조절
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # 흰색 사각형
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 50  # 첫 번째 텍스트의 위치
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)
        text_position += 50

    # 프레임에 테두리 추가
    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), border_color, 10)

    return frame

# 학생 비디오 스트리밍 처리
def generate_student_video():
    global student_frame, student_emotion
    log_flag = False
    while True:
        if student_frame is not None:
            # 감정에 따라 테두리 색상 결정
            dominant_emotion = emotion_history[-1] if emotion_history else "neutral"
            neutral_value = student_emotion.get('neutral', 100)
            if (dominant_emotion == "neutral" and neutral_value < 30) or dominant_emotion in {'angry', 'disgust', 'fear', 'sad'}:
                border_color = (0, 0, 255)  # 부정적 감정 또는 낮은 중립 감정에 대해 빨간 테두리
            else:
                border_color = (0, 255, 0)  # 긍정적 감정 또는 높은 중립 감정에 대해 초록 테두리

            texts = [
                f"Dominant Emotion: {dominant_emotion} {round(student_emotion.get(dominant_emotion, 0), 1)}%",
            ]

            # 학생 비디오 프레임에 감정 분석 텍스트와 테두리 추가
            student_frame_with_overlay = overlay_text_on_frame(student_frame, texts, border_color)

            # 플롯 생성 (학생 비디오의 높이에 맞게 조정)
            plot_img = plot_emotion_history(student_frame_with_overlay.shape[0], start_time_offset, emotion_history, time_history, emotion_values_history)

            # 플롯의 가로 세로 비율 유지하면서 학생 비디오의 높이에 맞게 조정
            plot_aspect_ratio = plot_img.shape[1] / plot_img.shape[0]  # 플롯의 가로/세로 비율
            target_height = student_frame_with_overlay.shape[0]  # 학생 비디오 높이에 맞춤
            target_width = int(target_height * plot_aspect_ratio)  # 비율에 맞춘 너비 계산
            plot_img_resized = cv2.resize(plot_img, (target_width, target_height))

            # 학생 프레임과 플롯을 가로로 결합 (하나의 플롯만)
            combined_frame = np.hstack((student_frame_with_overlay, plot_img_resized))

            ret, buffer = cv2.imencode('.jpg', combined_frame)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            if not log_flag:
                logging.info("Waiting for student connection")
                log_flag = True
            waiting_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(waiting_frame, "Waiting for student...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', waiting_frame)
            frame_data = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

# 학생 비디오 피드 라우트
@app.route('/student_video_feed')
def student_video_feed():
    return Response(generate_student_video(), mimetype='multipart/x-mixed-replace; boundary=frame')


# WebSocket 연결 처리
async def handle_connection(websocket, path):
    global connected_clients
    connected_clients.add(websocket)
    try:
        await receive_student_video(websocket)
        logging.info("Connection open with student")
    except Exception as e:
        logging.error(f"Connection closed: {e}")
    finally:
        connected_clients.remove(websocket)

# WebSocket 서버 시작
async def start_server():
    async with websockets.serve(handle_connection, WEBSOCKET_HOST, WEBSOCKET_PORT):
        await asyncio.Future()  # run forever

# WebSocket 비디오 수신을 위한 스레드 시작
def start_receiving_student_video():
    asyncio.run(start_server())

if __name__ == "__main__":
    video_thread = Thread(target=start_receiving_student_video)
    video_thread.start()
    app.run(host='0.0.0.0', port=5000)