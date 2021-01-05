import numpy as np
from feature_extraction.tsfeature import feature_core, feature_fft, feature_time


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# get data on each axis
def get_coor(container_A1, container_G1, reg=None):
    xa, ya, za = np.array(container_A1)[:, 1], np.array(container_A1)[:, 2], np.array(container_A1)[:, 3]
    xw, yw, zw = np.array(container_G1)[:, 1], np.array(container_G1)[:, 2], np.array(container_G1)[:, 3]
    if reg == 'stdr':
        xa, ya, za = standardization(xa), standardization(ya), standardization(za)
        xw, yw, zw = standardization(xw), standardization(yw), standardization(zw)
    elif reg == 'norm':
        xa, ya, za = normalization(xa), normalization(ya), normalization(za)
        xw, yw, zw = normalization(xw), normalization(yw), normalization(zw)
    return xa, ya, za, xw, yw, zw


# 根据时间窗口取得19个时域频域特征
def get_aw_feature(walk_start_stamp, walk_end_stamp, container_A1, container_G1):
    xa, ya, za, xw, yw, zw = get_coor(container_A1, container_G1)
    xa, ya, za = xa[walk_start_stamp:walk_end_stamp], ya[walk_start_stamp:walk_end_stamp], za[walk_start_stamp:walk_end_stamp]
    xw, yw, zw = xw[walk_start_stamp:walk_end_stamp], yw[walk_start_stamp:walk_end_stamp], zw[walk_start_stamp:walk_end_stamp]

    # 六个加速度的统计特征 （窗口为500，步长为500）
    feature_xa = feature_core.sequence_feature(xa, 0, 0)
    feature_ya = feature_core.sequence_feature(ya, 0, 0)
    feature_za = feature_core.sequence_feature(za, 0, 0)
    feature_xw = feature_core.sequence_feature(xw, 0, 0)
    feature_yw = feature_core.sequence_feature(yw, 0, 0)
    feature_zw = feature_core.sequence_feature(zw, 0, 0)

    # return [feature_xa]
    return [feature_xa, feature_ya, feature_za, feature_xw, feature_yw, feature_zw]