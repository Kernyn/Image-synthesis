# 开发人员:  caiyifeng
# 开发时间:  2025/5/16
# 文件名:    ImageSynthesis.py

import cv2
import numpy as np
import glob
import os

def load_images_and_masks(folder, target_size=None):
    images = []
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    file_list = []
    for ext in exts:
        file_list.extend(glob.glob(os.path.join(folder, ext)))
    file_list = sorted(file_list)

    print(f"找到图片数量：{len(file_list)}")
    for file in file_list:
        print(f"加载图片: {file}")
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"加载失败: {file}")
            continue

        # 生成BGR和mask
        if len(img.shape) == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            mask = np.ones(bgr.shape[:2], dtype=np.uint8) * 255  # 全不透明
        elif img.shape[2] == 4:
            bgr = img[:, :, :3]
            alpha = img[:, :, 3]
            _, mask = cv2.threshold(alpha, 128, 255, cv2.THRESH_BINARY)
            mask = mask.astype(np.uint8)
        else:
            bgr = img
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            mask = mask.astype(np.uint8)

        if target_size:
            bgr = cv2.resize(bgr, target_size, interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        images.append((bgr, mask))

    return images

def stack_images_left_top(images, invert=0):
    if not images:
        print("无图片，无法堆叠")
        return None

    h, w = images[0][0].shape[:2]  # 统一大小，取第一个尺寸

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    alpha = np.zeros((h, w), dtype=np.uint8)  # 记录已合成区域掩码

    for idx, (img, mask) in enumerate(images):
        if invert == 1:
            img_proc = img.copy()
            img_proc[mask > 0] = 255 - img[mask > 0]
        else:
            img_proc = img

        mask_8u = mask.astype(np.uint8)
        inv_mask = cv2.bitwise_not(mask_8u)

        # 背景保留画布当前像素未被遮挡区域
        bg = cv2.bitwise_and(canvas, canvas, mask=inv_mask)
        # 前景为当前图像有效区域
        fg = cv2.bitwise_and(img_proc, img_proc, mask=mask_8u)
        # 合成
        merged = cv2.add(bg, fg)

        canvas = merged
        alpha = cv2.bitwise_or(alpha, mask_8u)  # 更新掩码

    return canvas

def get_and_update_counter(counter_file="counter.txt"):
    if not os.path.exists(counter_file):
        count = 1
    else:
        try:
            with open(counter_file, "r") as f:
                count = int(f.read().strip())
        except Exception:
            count = 1
    with open(counter_file, "w") as f:
        f.write(str(count + 1))
    return count

def main():
    folder = "fragments"
    invert = 1  # 0关闭反转，1开启反转

    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
    file_list = []
    for ext in exts:
        file_list.extend(glob.glob(os.path.join(folder, ext)))
    file_list = sorted(file_list)

    if not file_list:
        print("没有找到任何图像文件！")
        return

    first_img = cv2.imread(file_list[0], cv2.IMREAD_UNCHANGED)
    if first_img is None:
        print("无法读取第一张图片！")
        return

    if len(first_img.shape) == 2:
        first_img = cv2.cvtColor(first_img, cv2.COLOR_GRAY2BGR)
    elif first_img.shape[2] == 4:
        first_img = first_img[:, :, :3]

    target_size = (first_img.shape[1], first_img.shape[0])  # (宽, 高)

    images = load_images_and_masks(folder, target_size=target_size)
    if not images:
        print("没有加载到任何图片！请检查文件夹和图片格式。")
        return

    canvas = stack_images_left_top(images, invert=invert)
    if canvas is None:
        print("合成失败")
        return

    count = get_and_update_counter()
    save_path = f"result_stack_left_top_{count}.png"
    cv2.imwrite(save_path, canvas)
    print(f"合成图已保存为：{save_path}")

if __name__ == "__main__":
    main()
mosaic