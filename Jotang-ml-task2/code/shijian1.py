from PIL import Image
import numpy as np
import torch

# 读取图片（就在当前task2目录，文件名outer1.png）
img = Image.open("outer1.png")

# 转numpy数组 (HWC格式，Pillow默认)
img_np = np.array(img)
print("===== NumPy HWC =====")
print("shape:", img_np.shape)  # (高度H,宽度W,通道C)
print("dtype:", img_np.dtype)
print("最大值:", img_np.max())
print("最小值:", img_np.min())

# 取一个像素点，y是高度方向，x宽度方向
y, x = 10, 20
pixel_rgb = img_np[y, x]
print(f"像素({y},{x}) RGB值：", pixel_rgb)

# 转torch tensor，HWC -> CHW
img_tensor = torch.from_numpy(img_np).permute(2,0,1)
print("\n===== Torch Tensor CHW =====")
print("tensor shape:", img_tensor.shape)
print("dtype:", img_tensor.dtype)
print("最大值:", img_tensor.max().item())
print("最小值:", img_tensor.min().item())

def conv2d_manual(img_np, kernel, padding=0, stride=1):
    """
    手写二维卷积(互相关运算),支持HWC格式灰度图(H,W)或者RGB图(H,W,3)
    :param img_np: 输入图像数组,HWC 或者 HW
    :param kernel: 卷积核，二维矩阵 (kh, kw)
    :param padding: 边缘填充0的圈数
    :param stride: 滑动步长
    :return: 输出图像数组
    """
    # 获取卷积核高、宽
    kh, kw = kernel.shape
    # 判断是否是彩色图
    if len(img_np.shape) == 3:
        H, W, C = img_np.shape
    else:
        H, W = img_np.shape
        C = 1
        img_np = img_np[..., np.newaxis]  # 增加通道维度，统一处理逻辑

    # 1. 边缘填充 padding，补0
    img_pad = np.pad(img_np, ((padding, padding), (padding, padding), (0,0)), 
    mode="constant", constant_values=0).astype(np.float32)


    # 2. 计算输出特征图高、宽
    out_h = int((H + 2 * padding - kh) / stride + 1)
    out_w = int((W + 2 * padding - kw) / stride + 1)
    out = np.zeros((out_h, out_w, C), dtype=np.float32)

    # 3. 双重循环滑动卷积核【核心手写卷积部分，没有调用现成卷积API！】
    for y in range(out_h):
        for x in range(out_w):
            # 取出当前卷积窗口
            y_start = y * stride
            y_end = y_start + kh
            x_start = x * stride
            x_end = x_start + kw
            window = img_pad[y_start:y_end, x_start:x_end, :]
            # 窗口 和 卷积核 相乘求和
            for c in range(C):
                out[y, x, c] = np.sum(window[:,:,c] * kernel)

    # 把数值限制在0~255，转回uint8图片格式
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    # 如果原来不是彩色图，去掉通道维度
    if C == 1:
        out = np.squeeze(out, axis=-1)
    return out

# ================= 调用手写卷积 + 保存图片 =================
# 定义锐化卷积核
kernel_sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

# 调用手写卷积函数
result_img = conv2d_manual(img_np, kernel_sharpen, padding=0, stride=1)

# 保存输出图片
im = Image.fromarray(result_img)
im.save("sharpen_out.png")
print(" 卷积运行完成，已生成 sharpen_out.png")

# ================= 任务：三种卷积核依次处理 =================
# 1. 3×3 均值模糊
kernel_mean = np.ones((3, 3), dtype=np.float32) / 9.0
mean_out = conv2d_manual(img_np, kernel_mean, padding=1, stride=1)
Image.fromarray(mean_out).save("mean3x3_out.png")
print(" 均值模糊完成，已生成 mean3x3_out.png")

# 2. 5×5 高斯模糊（近似高斯核，权重和为1）
kernel_gaussian = np.array([
    [1,  4,  6,  4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1,  4,  6,  4, 1],
], dtype=np.float32) / 256.0
gaussian_out = conv2d_manual(img_np, kernel_gaussian, padding=2, stride=1)
Image.fromarray(gaussian_out).save("gaussian5x5_out.png")
print(" 高斯模糊完成，已生成 gaussian5x5_out.png")

# 3. Sobel 竖直边缘检测（Sobel-X，检测竖向边缘）
kernel_sobel_v = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
], dtype=np.float32)
sobel_v_out = conv2d_manual(img_np, kernel_sobel_v, padding=1, stride=1)
Image.fromarray(sobel_v_out).save("sobel_v_out.png")
print(" Sobel竖直边缘检测完成，已生成 sobel_v_out.png")

print("\n===== 三种卷积处理完成 =====")
print("生成文件：mean3x3_out.png, gaussian5x5_out.png, sobel_v_out.png")

# ========= Task7 valid / same 对比实验 =========
# 使用3×3卷积核，stride=1
kernel_3x3 = np.ones((3,3))/9.0

# 1. Valid模式 padding=0
print("\n==== Valid (padding=0) ====")
res_valid = conv2d_manual(img_np, kernel_3x3, padding=0, stride=1)
print(f"输入尺寸 H,W = {img_np.shape[0], img_np.shape[1]}")
print(f"输出尺寸 H,W = {res_valid.shape[0], res_valid.shape[1]}")
im_valid = Image.fromarray(res_valid)
im_valid.save("conv_valid.png")

# 2. Same模式，padding=1（3×3核，p=1，stride=1）
print("\n==== Same (padding=1) ====")
res_same = conv2d_manual(img_np, kernel_3x3, padding=1, stride=1)
print(f"输入尺寸 H,W = {img_np.shape[0], img_np.shape[1]}")
print(f"输出尺寸 H,W = {res_same.shape[0], res_same.shape[1]}")
im_same = Image.fromarray(res_same)
im_same.save("conv_same.png")

# 输出尺寸公式
print("\n卷积输出尺寸公式：")
print("H_out = floor((H_in + 2*padding - kernel_size)/stride) + 1")
print("W_out = floor((W_in + 2*padding - kernel_size)/stride) + 1")
