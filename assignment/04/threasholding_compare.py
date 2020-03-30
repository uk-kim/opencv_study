import numpy as np
import os
import cv2

# 1D Least Square
def fit_1d(src):
    # Y = a1*x1 + a2*x2 + a3
    # -> Y = X * A     : we have to find the fitting parameter A = [a1; a2; a3]
    # Use 1d least squares method with psuedo inverse to find the values of A.
    # 1) X^T * Y = X^T * X * A
    # 2) (X^T * X)^-1 * X^T * Y = (X^T * X)^-1 * X^T * X * A
    #    ->  A = (X^T * X)^-1 * X^T * Y       : That is psuedo inverse matrix of X
    print("Input : ", src)
    print("Input.shape : ", src.shape)

    w_range = np.arange(src.shape[1])
    h_range = np.arange(src.shape[0])

    xs, ys = np.meshgrid(w_range, h_range)
    xs = xs.ravel()
    ys = ys.ravel()
    X = np.stack([xs, ys, np.ones_like(xs)]).transpose([1, 0])  # [[x11, x12], [x21, x22], ... , [xn1, xn2]] : N * 2 matrix
    Y = src.ravel().astype(np.float)
    Y = np.expand_dims(Y, -1)   # [[y1], [y2], ... , [yn]] : N*1 matrix
    print("X shape : ", X.shape)
    print("Y shape : ", Y.shape)

    X_pinv = np.linalg.pinv(X)   # Same as (X^T * X)^-1
    print("X_pinv shape : ", X_pinv.shape)
    A = np.matmul(X_pinv, Y)
    print("A shape : ", A.shape)

    surface = np.matmul(X, A)
    surface = np.reshape(surface, src.shape)
    print("Fitted surface shape : ", surface.shape)
    return surface

# 2D Least Square
def fit_2d(src):
    # Y = a1*x1^2 + a2*x1*x2 + a3*x2^2 + a4
    # -> Y = X * A     : we have to find the fitting parameter A = [a1; a2; a3; a4]
    #                    X must be as [x1^2, x1*x2, x2^2, 1]
    # Use 1d least squares method with psuedo inverse to find the values of A.
    # 1) X^T * Y = X^T * X * A
    # 2) (X^T * X)^-1 * X^T * Y = (X^T * X)^-1 * X^T * X * A
    #    ->  A = (X^T * X)^-1 * X^T * Y       : That is psuedo inverse matrix of X
    print("Input : ", src)
    print("Input.shape : ", src.shape)

    w_range = np.arange(src.shape[1])
    h_range = np.arange(src.shape[0])

    xs, ys = np.meshgrid(w_range, h_range)
    xs = xs.ravel()
    ys = ys.ravel()

    X = np.stack([xs * xs, xs*ys, ys*ys, np.ones_like(xs)]).transpose([1, 0])  # [[x11, x12], [x21, x22], ... , [xn1, xn2]] : N * 2 matrix
    Y = src.ravel().astype(np.float)
    Y = np.expand_dims(Y, -1)   # [[y1], [y2], ... , [yn]] : N*1 matrix
    print("X shape : ", X.shape)
    print("Y shape : ", Y.shape)

    X_pinv = np.linalg.pinv(X)   # Same as (X^T * X)^-1
    print("X_pinv shape : ", X_pinv.shape)
    A = np.matmul(X_pinv, Y)
    print("A shape : ", A.shape)

    surface = np.matmul(X, A)
    surface = np.reshape(surface, src.shape)
    print("Fitted surface shape : ", surface.shape)
    return surface

def main():
    src_img_path = "./sudoku.png"
    src_img = cv2.imread(src_img_path)
    src_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

    print("1D Fitting")
    surface_1d = fit_1d(src_gray)
    print()
    print("2D Fitting")
    surface_2d = fit_2d(src_gray)

    diff_1 = src_gray.astype(np.float) - surface_1d
    diff_2 = src_gray.astype(np.float) - surface_2d

    th_img_1 = np.zeros_like(src_gray)
    th_img_2 = np.zeros_like(src_gray)
    th_img_3 = np.zeros_like(src_gray)
    th_img_4 = np.zeros_like(src_gray)
    th_img_5 = np.zeros_like(src_gray)
    th_img_6 = np.zeros_like(src_gray)

    tau1 = 0
    tau2 = 0
    print("Diff 1 --> max: {}, min: {}".format(diff_1.max(), diff_1.min()))
    print("Diff 2 --> max: {}, min: {}".format(diff_2.max(), diff_2.min()))
    th_img_1[diff_1 >= tau1] = 255
    th_img_2[diff_2 >= tau2] = 255
    th_img_3[src_gray >= src_gray.mean()] = 255
    _, th_img_4 = cv2.threshold(src_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    th_img_5 = cv2.adaptiveThreshold(src_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5)
    th_img_6 = cv2.adaptiveThreshold(src_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)

    surface_1d_int8 = surface_1d.copy()
    surface_1d_int8[surface_1d < 0] = 0
    surface_1d_int8[surface_1d > 255] = 255
    surface_1d_int8 = surface_1d_int8.astype(np.uint8)

    surface_2d_int8 = surface_2d.copy()
    surface_2d_int8[surface_2d < 0] = 0
    surface_2d_int8[surface_2d > 255] = 255
    surface_2d_int8 = surface_2d_int8.astype(np.uint8)

    cv2.imwrite("./result/sudoku_gray.jpg", src_gray)
    cv2.imwrite("./result/sudoku_th_1d_filter.jpg", th_img_1)
    cv2.imwrite("./result/sudoku_1d_surface.jpg", surface_1d_int8)
    cv2.imwrite("./result/sudoku_2d_surface.jpg", surface_2d_int8)
    cv2.imwrite("./result/sudoku_th_2d_filter.jpg", th_img_2)
    cv2.imwrite("./result/sudoku_th_mean.jpg", th_img_3)
    cv2.imwrite("./result/sudoku_th_otsu.jpg", th_img_4)
    cv2.imwrite("./result/sudoku_th_adaptive_mean.jpg", th_img_5)
    cv2.imwrite("./result/sudoku_th_adaptive_gaussian.jpg", th_img_6)

    cv2.imshow("Source Image : Gray", src_gray)
    cv2.imshow("Thresholding : 1D Fitting", th_img_1)
    cv2.imshow("Thresholding : 2D Fitting", th_img_2)
    cv2.imshow("Thresholding : Mean", th_img_3)
    cv2.imshow("Thresholding : otsu", th_img_4)
    cv2.imshow("Thresholding : Adaptive-Mean", th_img_5)
    cv2.imshow("Thresholding : Adaptive-Gaussian", th_img_6)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
