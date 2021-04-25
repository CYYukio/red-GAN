import numpy as np


def get_random_patchs(LDCT_slice, NDCT_slice, patch_size, whole_size=512):
    whole_h = whole_w = whole_size
    h = w = patch_size

    # patch image range
    hd, hu = h // 2, int(whole_h - np.round(h / 2))
    wd, wu = w // 2, int(whole_w - np.round(w / 2))

    # patch image center(coordinate on whole image)
    h_pc, w_pc = np.random.choice(range(hd, hu + 1)), np.random.choice(range(wd, wu + 1))
    LDCT_patch = LDCT_slice[:, h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]
    NDCT_patch = NDCT_slice[:, h_pc - hd: int(h_pc + np.round(h / 2)), w_pc - wd: int(w_pc + np.round(h / 2))]

    return LDCT_patch, NDCT_patch


# 以左上角为index
def get_index_patch(LDCT_slice, X, Y, patch_size):
    LDCT_patch = LDCT_slice[X:X+patch_size, Y:Y+patch_size]

    return LDCT_patch


def cal_psnr(im1, im2):
    mse = (np.abs(im1 - im2) ** 2).mean()
    psnr = 10 * np.log10(65535 * 65535 / mse)
    return psnr
