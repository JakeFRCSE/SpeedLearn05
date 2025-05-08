import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

X = np.random.randn(1000, 100) # 1000x100 행렬 생성 -> 1000개의 데이터, 100개의 특성
node_num = 100 # 노드 개수
hidden_layer_size = 5 # 은닉층 개수
activation = {} # 활성화 함수 저장할 딕셔너리

for i in range(hidden_layer_size):
    if i != 0: # i=0은 첫번째 hidden layer이므로, 이전 층의 활성화 함수를 사용하지 않음 -> 입력층 값을 그대로 받음
        X = activation[i-1] # 이전 층의 활성화 함수를 현재 층의 입력으로 사용

    # w = np.random.randn(node_num, node_num) * 1 # *1은 표준편차를 1로 설정
    w = np.random.randn(node_num, node_num) * 0.01 # *0.01은 표준편차를 0.01로 설정
    # w = np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num) 
    # w = np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

    a = np.dot(X, w) # 행렬 곱
    z = sigmoid(a) # 활성화 함수 적용
    activation[i] = z # 현재 층의 활성화 함수를 딕셔너리에 저장

# 히스토그램 그리기
for i, a in activation.items():
    plt.subplot(1, len(activation), i+1) # 1행 len(activation)열의 서브플롯 생성
    plt.title(str(i+1) + "-layer") # 서브플롯 제목 설정
    if i != 0: plt.yticks([], []) # y축 눈금 제거
    plt.hist(a.flatten(), 30, range=(0,1)) # 히스토그램 그리기
plt.show() # 그래프 출력

