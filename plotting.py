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
    # 플롯의 가로 길이를 적절하게 조정 (가로: 12, 세로: height / 100 * 2.5)
    plt.figure(figsize=(12, height / 100 * 2.5))  # 플롯의 가로를 키움 (이전보다 비율 조정)

    # X, Y 좌표 배열 초기화
    x_values = []
    y_values = []

    # 각 시간에 따른 감정과 해당 위치 결정
    for i in range(len(emotion_history)):
        # X축이 구간을 넘어서면 새로운 선을 그리도록 함
        if i > 0 and time_history[i] < time_history[i-1]:
            plt.plot(x_values, y_values, color='blue', linewidth=3.0)  # 선의 굵기를 3.0으로 설정
            x_values = []  # 새 구간 시작
            y_values = []

        emotion = emotion_history[i]
        y_pos = calculate_y_position(emotion, emotion_values_history[i][emotion])

        x_values.append(time_history[i] + x_start)
        y_values.append(y_pos)

    # 남은 점들을 연결하여 마지막 선을 그림
    plt.plot(x_values, y_values, color='blue', linewidth=3.0)  # 선의 굵기를 3.0으로 설정

    # 글씨 크기를 더 키움
    plt.yticks(ticks=range(len(emotion_order)), labels=emotion_order, fontsize=18)
    plt.ylim(-0.5, len(emotion_order) - 0.5)
    plt.xlim(x_start, x_start + 60)  # 1분 (60초) 동안의 범위
    plt.xlabel('Time (seconds)', fontsize=20)  # X축 라벨 크기 설정
    plt.ylabel('Emotion', fontsize=20)  # Y축 라벨 크기 설정
    plt.title('Emotion Over Time', fontsize=24)  # 타이틀 크기 설정
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