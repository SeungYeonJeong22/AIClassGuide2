import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import cv2

# 감정의 순서를 정의 (부정적 -> 긍정적)
emotion_order = ['angry', 'disgust', 'fear', 'sad', 'neutral', 'happy', 'surprise']

def calculate_y_position(emotion, value):
    idx = emotion_order.index(emotion)
    if idx == len(emotion_order) - 1:
        return idx  # 마지막 감정인 경우, 바로 위치

    # 감정 간의 위치를 점수에 따라 계산
    return idx + (value / 100.0)

def plot_emotion_history(height, x_start, emotion_history, time_history, emotion_values_history):
    # 플롯의 크기를 크게 설정 (가로: 24, 세로: 12)
    plt.figure(figsize=(24, 12))  # 가로와 세로 크기를 크게 설정

    # X, Y 좌표 배열 초기화
    x_values = []
    y_values = []

    # 3분(180초) 단위로 플롯을 초기화하는 로직 추가
    plot_duration = 180  # 3분 동안 데이터를 그린 후 초기화
    if len(time_history) > 0 and time_history[-1] >= plot_duration:
        # 3분이 지나면 x_start를 플롯의 현재 마지막 시간으로 설정하여 플롯을 초기화
        x_start += plot_duration

        # 3분마다 이전 데이터를 비움
        emotion_history.clear()
        time_history.clear()
        emotion_values_history.clear()

    # 각 시간에 따른 감정과 해당 위치 결정
    for i in range(len(emotion_history)):
        emotion = emotion_history[i]
        y_pos = calculate_y_position(emotion, emotion_values_history[i][emotion])

        x_values.append(time_history[i] + x_start)
        y_values.append(y_pos)

    # 선 그래프 그리기 (항상 파란색, 선 굵기 4.0)
    plt.plot(x_values, y_values, color='blue', linewidth=4.0)

    # 글씨 크기 및 스타일 조정
    plt.yticks(ticks=range(len(emotion_order)), labels=emotion_order, fontsize=24)
    plt.xticks(fontsize=24)  # x축 눈금의 글씨 크기 설정
    plt.ylim(-0.5, len(emotion_order) - 0.5)
    plt.xlim(x_start, x_start + plot_duration)  # 3분 (180초) 동안의 범위
    plt.xlabel('Time (seconds)', fontsize=32)  # X축 라벨 크기 설정
    plt.ylabel('Emotion', fontsize=28)  # Y축 라벨 크기 설정
    plt.title('Emotion Over Time', fontsize=32)  # 타이틀 크기 설정
    plt.grid(True)

    # 플롯 이미지를 바이트 배열로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # 이미지를 OpenCV 형식으로 변환
    plot_img = Image.open(buf)
    plot_img = np.array(plot_img)
    plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV

    return plot_img