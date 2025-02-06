import os
from PIL import Image
import numpy as np

# 文件夹路径
sourceA_folder = 'D:\code\python\MSI-DTrans\Datasets\Eval\Lytro\sourceA'
sourceB_folder = 'D:\code\python\MSI-DTrans\Datasets\Eval\Lytro\sourceB'
outputA_folder = 'D:\\code\\python\\MSI-DTrans\\Datasets\\Eval\\Lytro_blur\\sourceA'
outputB_folder = 'D:\\code\\python\\MSI-DTrans\\Datasets\\Eval\\Lytro_blur\\sourceB'

# 确保输出文件夹存在
os.makedirs(outputA_folder, exist_ok=True)
os.makedirs(outputB_folder, exist_ok=True)

# 获取sourceA和sourceB的文件名（假设文件名一一对应）
sourceA_images = sorted(os.listdir(sourceA_folder))
sourceB_images = sorted(os.listdir(sourceB_folder))

# 检查文件夹下的图像是否一致
assert len(sourceA_images) == len(sourceB_images), "sourceA和sourceB文件数量不一致！"


# 图像混合函数
def blend_images(imgA, imgB, ratioA, ratioB):
    # 确保两张图片大小一致，若不同则调整大小
    if imgA.size != imgB.size:
        imgB = imgB.resize(imgA.size)

    # 转换为numpy数组进行加权求和
    imgA_array = np.array(imgA)
    imgB_array = np.array(imgB)

    blended_array = (imgA_array * ratioA + imgB_array * ratioB).astype(np.uint8)

    # 转换回Image对象
    blended_img = Image.fromarray(blended_array)
    return blended_img


# 遍历图像进行合成
for a_img_name, b_img_name in zip(sourceA_images, sourceB_images):
    # 加载图像
    imgA = Image.open(os.path.join(sourceA_folder, a_img_name))
    imgB = Image.open(os.path.join(sourceB_folder, b_img_name))

    # 生成新的图像
    new_imgA = blend_images(imgA, imgB, 0.9, 0.1)  # 0.55A + 0.45B
    new_imgB = blend_images(imgA, imgB, 0.1, 0.9)  # 0.45A + 0.55B

    # 保存合成后的图像
    new_imgA.save(os.path.join(outputA_folder, a_img_name))
    new_imgB.save(os.path.join(outputB_folder, b_img_name))

    # 输出保存路径
    print(
        f"保存合成图像: {a_img_name} -> {os.path.join(outputA_folder, a_img_name)}, {b_img_name} -> {os.path.join(outputB_folder, b_img_name)}")

print("所有图像处理完成！")

