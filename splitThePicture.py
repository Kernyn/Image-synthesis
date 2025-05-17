import cv2
import numpy as np
import os
import random


def split_image_with_blocks(image_path, output_dir, num_fragments=4, blocks_per_fragment=50, block_size=5, invert=0,
                            bg_color=(0, 0, 0, 0)):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("无法读取图片！")
        return

    # 转3通道BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    h, w = img.shape[:2]

    # 颜色反转
    if invert == 1:
        img = 255 - img

    # 计算块格数
    blocks_x = w // block_size
    blocks_y = h // block_size
    total_blocks = blocks_x * blocks_y

    # 创建所有块的索引列表 (block_y, block_x)
    all_blocks = [(by, bx) for by in range(blocks_y) for bx in range(blocks_x)]

    # 先打乱
    random.shuffle(all_blocks)

    # 给每个碎片分配块
    fragments_blocks = [[] for _ in range(num_fragments)]
    idx = 0
    for block in all_blocks:
        fragments_blocks[idx].append(block)
        idx = (idx + 1) % num_fragments

    # 如果每片想要的块数少，则切片一下
    for i in range(num_fragments):
        if blocks_per_fragment < len(fragments_blocks[i]):
            fragments_blocks[i] = random.sample(fragments_blocks[i], blocks_per_fragment)

    # 生成碎片图
    for i, blocks in enumerate(fragments_blocks):
        layer = np.zeros((h, w, 4), dtype=np.uint8)
        layer[:, :, 0] = bg_color[0]
        layer[:, :, 1] = bg_color[1]
        layer[:, :, 2] = bg_color[2]
        layer[:, :, 3] = bg_color[3]  # 透明度

        for (by, bx) in blocks:
            y_start = by * block_size
            x_start = bx * block_size
            y_end = y_start + block_size
            x_end = x_start + block_size

            # 防止边界越界
            y_end = min(y_end, h)
            x_end = min(x_end, w)

            layer[y_start:y_end, x_start:x_end, 0:3] = img[y_start:y_end, x_start:x_end]
            layer[y_start:y_end, x_start:x_end, 3] = 255

        out_path = os.path.join(output_dir, f"fragment_{i + 1}.png")
        cv2.imwrite(out_path, layer)
        print(f"保存子图: {out_path}")

    print(f"完成拆分为 {num_fragments} 张块状子图。每片大约 {blocks_per_fragment} 块。")


def main():
    image_path = "result_left_aligned_invert_1_16.png"  # 原图路径
    output_dir = "layers"  # 输出文件夹
    num_fragments = 8  # 拆分张数
    blocks_per_fragment = 5000  # 每张碎片包含块数（注意总块数必须够分）
    block_size = 10  # 块大小（像素）
    invert = 1  # 颜色反转开关
    bg_color = (0, 0, 0, 0)  # 背景颜色(B,G,R,A),默认透明

    split_image_with_blocks(image_path, output_dir, num_fragments, blocks_per_fragment, block_size, invert, bg_color)


if __name__ == "__main__":
    main()
