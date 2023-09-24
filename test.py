import cv2
import numpy as np

def convert_to_grayscale(disparity_map):
    grayscale_map = cv2.cvtColor(disparity_map, cv2.COLOR_BGR2GRAY)
    return grayscale_map

def save_disparity_results(disparity_map, output_file):
    with open(output_file, 'w') as f:
        height, width = disparity_map.shape[:2]
        for y in range(height):
            for x in range(width):
                disparity = disparity_map[y, x] / 16.0
                f.write(f'{disparity} ')
            f.write('\n')

# 加载多通道视差图像
disparity_map = cv2.imread('./data/disparity/0000020_10.png')

# 将多通道视差图像转换为单通道灰度图像
grayscale_map = convert_to_grayscale(disparity_map)

# 保存单通道灰度图像为视差结果文件
save_disparity_results(grayscale_map, './data/disparity/0000020_10.txt')