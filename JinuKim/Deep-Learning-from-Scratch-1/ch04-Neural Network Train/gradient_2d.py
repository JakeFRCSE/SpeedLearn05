#편미분
def numerical_gradient(f, x): # f: 함수, x: 입력값
    h = 1e-4 # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h) # 중앙 차분법을 사용하여 수치 미분을 계산

def functon_2(x): # function_2(x) = x[0]^2 + x[1]^2를 구현한 것 
    return x[0]**2 + x[1]**2 # or np.sum(x**2)

# x[0]=3, x[1]=4일 때 x[0]에 대한 편미분을 구하는 함수 편미분을 구하는 함수
def function_tmp1(x0): 
    return x0*x0 + 4.0**2.0 

# x[0]=3, x[1]=4일 때 x[1]에 대한 편미분을 구하는 함수 편미분을 구하는 함수
def function_tmp2(x1): 
    return 3.0**2.0 + x1*x1