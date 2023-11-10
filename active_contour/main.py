import numpy as np
import cv2
from scipy import interpolate
from scipy.ndimage.filters import laplace
from skimage.filters import sobel, gaussian
import sys
import utils


def A(n, a, b):
    A = np.eye(n, k=1) * (-a - 4 * b) + np.eye(n, k=2) * b + np.eye(n, k=n-2) * b + np.eye(n, k=n-1) * (-a - 4 * b)
    A += A.T + np.eye(n, k=0) * (2 * a + 6 * b)
    return A

def P_ext(img, w_line, w_edge):
        P_line = gaussian(img, 2)
        P_edge = sobel(P_line)
        P_ext =  w_line * P_line + w_edge * P_edge
        return P_ext


def snake_deform(img, init_snake, alpha, beta, tau, w_line, w_edge, kappa1):
    kappa2 = -0.8
    x_normal = np.zeros(init_snake.shape[0])
    y_normal = np.zeros(init_snake.shape[0])
    
    inv_m = np.linalg.inv(np.eye(init_snake.shape[0]) + tau * A(init_snake.shape[0], alpha, beta))
#     потенциал внешних сил, задан на сетке
    ext = P_ext(img, w_line, w_edge)
    P_ext_interp = interpolate.RectBivariateSpline(np.arange(ext.shape[1]), np.arange(ext.shape[0]), ext.T, s=0)
    
    x = init_snake
    for _ in range(800):
        x_pred = x
        F_ext_x = P_ext_interp(x[:, 0], x[:, 1], dx=1, grid=False)
        F_ext_y = P_ext_interp(x[:, 0], x[:, 1], dy=1, grid=False)
        
        F_ext_x /= np.linalg.norm(F_ext_x, ord=1)
        F_ext_y /= np.linalg.norm(F_ext_y, ord=1)

        for i in range(x.shape[0]-1):
            x_normal[i] = (x[:, 1])[i+1] - (x[:, 1])[i]
            y_normal[i] = (x[:, 0])[i] - (x[:, 0])[i+1]
        
        x[:, 0] = np.matmul(inv_m, x[:, 0] + tau * (kappa1 * x_normal - kappa2 * F_ext_x))
        x[:, 1] = np.matmul(inv_m, x[:, 1] + tau * (kappa1 * y_normal - kappa2 * F_ext_y))

        x = reparametrization(x, img.shape)
    return x

def reparametrization(coord, shape):
    coord[-1, :] = coord[0, :]
    tck, u = interpolate.splprep([coord[:, 0], coord[:, 1]], s=0, per=True)
    xi, yi = interpolate.splev(np.linspace(0, 1, coord.shape[0]), tck)
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    xi[xi >= shape[0]] = shape[0] - 1
    yi[yi >= shape[1]] = shape[1] - 1
    return np.concatenate((xi.reshape(-1, 1), yi.reshape(-1, 1)), axis=1)

img = cv2.imread(sys.argv[1], 0).astype(float) / 255
init = np.loadtxt(sys.argv[2])
output_path = sys.argv[3]

alpha = float(sys.argv[4])
beta = float(sys.argv[5])
tau = float(sys.argv[6])
w_line = float(sys.argv[7])
w_edge = float(sys.argv[8])
kappa = float(sys.argv[9])

snake = snake_deform(img, init, alpha, beta, tau, w_line, w_edge, kappa)
utils.save_mask(output_path, snake, img)