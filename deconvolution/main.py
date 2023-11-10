import numpy as np
import cv2
import sys
from scipy.ndimage import convolve

def derivative_nev(z, H, u):
    res = 2 * (convolve(convolve(z, H), H[::-1, ::-1]) - convolve(u, H[::-1, ::-1]))
    return res

def derivative_btv(z, p=1):
    height, width = z.shape
    res = np.zeros(z.shape)
    z_pad = np.pad(z, pad_width=p)
    for x in range(-p, p + 1):
        for y in range(-p, p + 1):
            if x**2 + y**2 == 0:
                continue
            sign = np.sign(z_pad[p - x : p + height - x, p - y : p + width - y] - z)
            sign_pad = np.pad(sign, pad_width=p)
            res += (sign_pad[p + x : p + height + x, p + y : p + width + y] - sign) / (x**2 + y**2) ** 0.5
    return res

def derivative_btv2(z, p=1):
    height, width = z.shape
    res = np.zeros(z.shape)
    z_pad = np.pad(z, pad_width=p)
    for x in range(-p, p + 1):
        for y in range(-p, p + 1):
            if x**2 + y**2 == 0:
                continue
            sign = np.sign(z_pad[p - x : p + height - x, p - y : p + width - y] + z_pad[p + x : p + height + x, p + y : p + width + y] - 2 * z)
            sign_pad = np.pad(sign, pad_width=p)
            res += (sign_pad[p - x : p + height - x, p - y : p + width - y] + \
                    sign_pad[p + x : p + height + x, p + y : p + width + y] - 2 * sign) / (x**2 + y**2) ** 0.5
    return res

def derivative_func(z, H, u, alpha_1, alpha_2):
    return derivative_nev(z, H, u) + alpha_1 * derivative_btv(z) + alpha_2 * derivative_btv2(z)


def nesterov(H, u, alpha_1, alpha_2, mu=0.8):
    z = u.copy()
    v = np.zeros(u.shape)
    num_iterations = 100
    
    for t in range(1, num_iterations):
        g = derivative_func(z + mu * v, H, u, alpha_1, alpha_2)
        beta = 1 * 0.2 ** (float(t) / num_iterations)
        v = mu * v - beta * g
        z = z + v
    z = np.clip(z, a_min=0, a_max=255)
    return z

blurred = cv2.imread(sys.argv[1], 0).astype(float)
kernel = cv2.imread(sys.argv[2], 0).astype(float)
output_path = sys.argv[3]
noise_level = float(sys.argv[4])


kernel = kernel / kernel.sum()
alpha_1 = noise_level * 0.08 + 0.03
alpha_2 = 0.0005 * np.power(noise_level, 2.2)
output_img = nesterov(kernel, blurred, alpha_1, alpha_2)

cv2.imwrite(output_path, output_img)