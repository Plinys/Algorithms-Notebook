import numpy as np
import cv2

image = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('source image', image)
cv2.waitKey(0)
# 利用astype() 将uint8 类型数组转换为 float 方便运算
image_array = image.astype("float")

kernal = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])
# 卷积内核权重值应该为1 否则卷积会造成亮度过大
kernal_sum = (np.abs(kernal)).sum()

i_height, i_width = image_array.shape
k_height, k_width = kernal.shape
# padding 的大小依据内核大小来决定
h_padding_size = int(k_height/2)
w_padding_size = int(k_width/2)

# 图像padding 以0 填充边界
image_padding = cv2.copyMakeBorder(image_array, h_padding_size, h_padding_size,
                                   w_padding_size, w_padding_size,
                                   cv2.BORDER_CONSTANT, value=0)

dst_image = np.zeros(image.shape, dtype=float)
for y in range(h_padding_size, i_height + h_padding_size):
    for x in range(w_padding_size, i_width + w_padding_size):
        roi = image_padding[(y - h_padding_size):(y + h_padding_size + 1),
              (x - w_padding_size):(x + w_padding_size + 1)]
        new_pixel = ((roi * kernal).sum())/kernal_sum
        dst_image[y - h_padding_size, x - w_padding_size] = new_pixel

pixel_max = dst_image.max()
print(pixel_max)
dst_image = dst_image.astype("uint8")
cv2.imshow('fliter image', dst_image)
cv2.waitKey(0)
cv2.destroyAllWindows()