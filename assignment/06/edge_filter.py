import cv2
import numpy as np

image_file = "./sudoku.png"

img = cv2.imread(image_file)
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# Edge 1: Edge differential
gx_kernel = np.array([[-1, 1]])
gy_kernel = np.array([[-1], [1]])

edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

edge_1 = edge_gx + edge_gy
edge_1 = cv2.putText(edge_1, "Edge differential", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_1 : ", edge_1.shape)

# Edge 2: Roberts
gx_kernel = np.array([[1, 0], [0, -1]])
gy_kernel = np.array([[0, 1], [-1, 1]])

edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gy_kernel)

edge_2 = edge_gx + edge_gy
edge_2 = cv2.putText(edge_2, "Roberts", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_2 : ", edge_2.shape)

# Edge 3: Prewitt
gx_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
gy_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

edge_gx = cv2.filter2D(img, -1, gx_kernel)
edge_gy = cv2.filter2D(img, -1, gx_kernel)

edge_3 = edge_gx + edge_gy
edge_3 = cv2.putText(edge_3, "Prewitt", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_3 : ", edge_3.shape)

# Edge 4: Sobel
edge_gx = cv2.Sobel(img, -1, 1, 0, ksize=3)
edge_gy = cv2.Sobel(img, -1, 0, 1, ksize=3)

edge_4 = edge_gx + edge_gy
edge_4 = cv2.putText(edge_4, "Sobel", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_4 : ", edge_4.shape)

# Edge 5: Scharr
edge_gx = cv2.Scharr(img, -1, 1, 0)
edge_gy = cv2.Scharr(img, -1, 0, 1)

edge_5 = edge_gx + edge_gy
edge_5 = cv2.putText(edge_5, "Scharr", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_5 : ", edge_5.shape)

# Edge 6: Laplacian
edge_6 = cv2.Laplacian(img, -1)
edge_6 = cv2.putText(edge_6, "Laplacian", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_6 : ", edge_6.shape)

# Edge 7: Canny
edge_7 = cv2.Canny(img, 100, 200)
edge_7 = cv2.cvtColor(edge_7, cv2.COLOR_GRAY2BGR)
edge_7 = cv2.putText(edge_7, "Canny", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))
print("edge_7 : ", edge_7.shape)

img = cv2.putText(img, "Input", (10, img.shape[0]-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.9, (0, 255, 0))

merged_1 = np.hstack((img, edge_1, edge_2, edge_3))
merged_2 = np.hstack((edge_4, edge_5, edge_6, edge_7))
merged   = np.vstack((merged_1, merged_2))
cv2.imshow("Edge Images", merged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("edge_compare.png", merged)
