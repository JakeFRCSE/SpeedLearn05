import numpy as np

# 원 핫 인코딩을 사용
def cross_entropy_error(y, t): # y: 예측값, t: 정답 레이블
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size) 
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# 숫자 레이블을 사용
def cross_entropy_error(y, t): # y: 예측값, t: 정답 레이블
    if y.ndim == 1:
        t = t.reshape(1, t.size) 
        y = y.reshape(1, y.size) 
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size # y[np.arange(batch_size), t] : 정답 레이블에 해당하는 예측값

