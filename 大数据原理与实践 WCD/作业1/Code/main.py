import cv2

graph = cv2.imread('WaTaShi.jpg', cv2.IMREAD_GRAYSCALE)
graph1 = cv2.imread('WaTaShi.jpg')
# 求梯度
grad_x = cv2.Sobel(graph, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(graph, cv2.CV_64F, 0, 1, ksize=3)
# 求边缘强度
combination = cv2.magnitude(grad_x, grad_y)
# 转化为unit8
grad_x = cv2.convertScaleAbs(grad_x)
grad_y = cv2.convertScaleAbs(grad_y)
combination = cv2.convertScaleAbs(combination)
# 图像分割
value, binary_edge = cv2.threshold(combination, 50, 255, cv2.THRESH_BINARY)


# 设置目标大小（宽，高）
target_size = (700, 400)
# 调整图像大小
grad_x_resize = cv2.resize(grad_x, target_size)
grad_y_resize = cv2.resize(grad_y, target_size)
binary_edge_resize = cv2.resize(binary_edge, target_size)
graph1 = cv2.resize(graph1, target_size)
# 显示调整大小后的图像
cv2.imshow('Origin graph', graph1)
cv2.imshow('Sobel X', grad_x_resize)
cv2.imshow('Sobel Y', grad_y_resize)
cv2.imshow('Edge Detection', binary_edge_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
