# 1. 使用字典记录姓名与成绩
score_dict= {'张三': 95, '李四': 75, '王五': 85}

# 2. 编写函数：计算平均分，找出最高分
def calc_scores(score_dict):
    # 取出所有成绩
    scores = list(score_dict.values())
    # 计算平均分
    average = sum(scores) / len(scores)
    # 找最高分
    max_score = max(scores)
    return average, max_score

# 调用函数
avg, high = calc_scores(score_dict)

# 打印输出
print("成绩字典：", score_dict)
print(f"平均分：{avg:.2f}")
print(f"最高分：{high}")

import numpy as np
# 3. NumPy 创建两个矩阵，做矩阵乘法
matrix_A = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
matrix_B = np.array([
    [7, 8],
    [9, 10],
    [11, 12]
])

# 矩阵乘法
result_matrix = np.matmul(matrix_A, matrix_B)

# ========== 4. 输出矩阵 + 形状 ==========
print("\n矩阵A：")
print(matrix_A)
print(f"矩阵A shape = {matrix_A.shape}")

print("\n矩阵B：")
print(matrix_B)
print(f"矩阵B shape = {matrix_B.shape}")

print("\n矩阵乘法结果：")
print(result_matrix)
print(f"结果矩阵 shape = {result_matrix.shape}")