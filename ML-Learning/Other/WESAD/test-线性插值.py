import numpy as np

# 使用线性插值对齐数据
def interpolate_data(data, target_length):
    current_length = len(data)
    x_old = np.linspace(0, 1, current_length)
    print("x_old:", x_old)
    x_new = np.linspace(0, 1, target_length)
    print("x_new:", x_new)
    interpolated_data = np.interp(x_new, x_old, data)
    return interpolated_data


arr = [1,2,3]
target_length = 10
interpolated_data = interpolate_data(arr, target_length)
print("arr:",arr)
print("interpolated_data:",interpolated_data)
