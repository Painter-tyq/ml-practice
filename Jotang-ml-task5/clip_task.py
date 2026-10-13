import os
import torch
import clip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw
import torch.nn.functional as F

# ==========================================
# 任务2-步骤1：准备4张内容和对应的英文描述
# ==========================================
# 为了让你能直接跑通，我写了自动生成4张测试图片的代码。
# 如果你有自己的图片，把 USE_DUMMY 改成 False，并修改 image_paths 即可。
USE_DUMMY = True 

if USE_DUMMY:
    print("正在生成4张测试图片...")
    os.makedirs("test_images", exist_ok=True)
    colors = ["red", "blue", "green", "yellow"]
    image_paths = []
    for i, color in enumerate(colors):
        img = Image.new("RGB", (224, 224), color=color)
        path = f"test_images/{color}_square.jpg"
        img.save(path)
        image_paths.append(path)

# 4句正确的英文描述 + 任务2-步骤4要求：故意加入一句很相似但关键内容错误的描述（比如颜色错误）
# 假设前4句对应上面的红、蓝、绿、黄方块，第5句是故意写错的紫色
text_descriptions = [
    "a red square",       # 对应图1
    "a blue square",      # 对应图2
    "a green square",     # 对应图3
    "a yellow square",    # 对应图4
    "a purple square"     # 任务2-步骤4：故意加入的错误描述（错误颜色）
]

# ==========================================
# 环境与模型加载 (推荐RN50)
# ==========================================
# 自动检测是否有 GPU，没有就用 CPU (轻薄本走这里)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"当前使用的设备: {device}")

# 加载较小的 RN50 模型 (首次运行会自动下载，约100MB，请耐心等待)
print("正在加载 CLIP 模型 (RN50)，首次运行需下载权重...")
model, preprocess = clip.load("RN50", device=device)
model.eval()

# ==========================================
# 任务2-步骤2：提取图片特征和文本特征，并L2归一化
# ==========================================
# 1. 处理并提取图片特征
image_tensors = torch.stack([preprocess(Image.open(p)).to(device) for p in image_paths])
with torch.no_grad():
    image_features = model.encode_image(image_tensors)

# 2. 处理并提取文本特征
text_tokens = clip.tokenize(text_descriptions).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)

# 3. 打印形状
print(f"图片特征形状: {image_features.shape}") # 应该是 [4, 1024] (RN50的维度是1024)
print(f"文本特征形状: {text_features.shape}") # 应该是 [5, 1024]

# 4. L2 归一化 (计算余弦相似度前的必须步骤)
image_features = F.normalize(image_features, p=2, dim=-1)
text_features = F.normalize(text_features, p=2, dim=-1)

# ==========================================
# 任务2-步骤3：计算余弦相似度矩阵并画热力图
# ==========================================
# 因为已经L2归一化，直接矩阵乘法就是余弦相似度
similarity_matrix = (image_features @ text_features.T).cpu().numpy()

# 绘制热力图
plt.figure(figsize=(10, 6))
# 坐标轴标签
x_labels = [f"T{i+1}: {t}" for i, t in enumerate(text_descriptions)]
y_labels = [f"I{i+1}: {os.path.basename(p)}" for i, p in enumerate(image_paths)]

sns.heatmap(similarity_matrix, annot=True, cmap="Blues", xticklabels=x_labels, yticklabels=y_labels, fmt=".3f")
plt.title("CLIP Image-Text Similarity Matrix")
plt.tight_layout()
plt.savefig("similarity_heatmap.png") # 保存图片到本地
print("热力图已保存为 similarity_heatmap.png")
# plt.show() # 如果你是在 Jupyter Notebook 里跑，把这行取消注释可以弹窗看图

# ==========================================
# 任务2-步骤3 & 4：配对结果分析
# ==========================================
print("\n--- 配对结果分析 ---")
for i in range(len(image_paths)):
    # 找到相似度最高的文本索引
    best_match_idx = np.argmax(similarity_matrix[i])
    score = similarity_matrix[i][best_match_idx]
    is_correct = (best_match_idx == i) # 因为我们前面的正确描述刚好和图片顺序一致
    
    print(f"图片 {i+1} ({os.path.basename(image_paths[i])}) 最匹配的文本是: '{text_descriptions[best_match_idx]}' (相似度: {score:.4f})")
    if is_correct:
        print("   -> ✅ 配对成功")
    else:
        print("   -> ❌ 配对失败")

print("\n--- 任务2-步骤4：错误描述分析 ---")
# 检查错句(第5句，索引4)被分配给了哪张图
wrong_text_idx = 4
wrong_text_scores = similarity_matrix[:, wrong_text_idx]
for i in range(len(image_paths)):
    print(f"错句 '{text_descriptions[wrong_text_idx]}' 与 图片 {i+1} 的相似度: {wrong_text_scores[i]:.4f}")

print("\n【分析指南】")
print("1. 观察热力图对角线是否呈现深蓝色（高相似度）。")
print("2. 错句（紫色方块）与所有正确图片的相似度应该都低于正确句子的相似度。")
print("3. CLIP 能够区分颜色、数量等细微差别，如果错句相似度很高，说明CLIP没分辨出来。")