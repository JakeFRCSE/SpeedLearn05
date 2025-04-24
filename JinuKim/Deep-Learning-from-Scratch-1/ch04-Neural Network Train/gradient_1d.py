import numpy as np
import matplotlib.pyplot as plt

def numerical_gradient(f, x): # f: 함수, x: 입력값
    h = 1e-4 # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h) # 중앙 차분법을 사용하여 수치 미분을 계산

def function_1(x):
    return x ** 2

x = np.arange(0.0, 20.0, 0.1) # 0.0부터 20.0까지 0.1 간격으로 생성(이때 20.0은 포함되지 않음)
y = function_1(x) # y = x^2
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y)
plt.show()