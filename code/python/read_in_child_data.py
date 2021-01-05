import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from feature_extraction.tsfeature import feature_core, feature_fft, feature_time

link1 = r'C:\Users\20804370\Desktop\activity-rec-feature-extr\activityrecognition\code\python\child_data\motion_data_2020-12-26__10-00-29.csv'
link2 = r'C:\Users\20804370\Desktop\activity-rec-feature-extr\activityrecognition\code\python\child_data\motion_data_2020-12-26__11-14-14.csv'
link3 = r'C:\Users\20804370\Desktop\activity-rec-feature-extr\activityrecognition\code\python\child_data\motion_data_2020-12-26__14-35-58.csv'

lines = pd.read_csv(link3)
# print(lines)

lines.columns = ['ord', 'time', 'ax', 'ay', 'az', 'wx', 'wy', 'wz', 'tile']
lines.drop(['ord', 'time', 'tile'], axis=1, inplace=True)

container_A1, container_G1 = [], []
a_axis = ['ax', 'ay', 'az']
g_axis = ['wx', 'wy', 'wz']
for axis in a_axis:
    container_A1.append(pd.Series(lines[axis], name=axis))
for axis in g_axis:
    container_G1.append(pd.Series(lines[axis], name=axis))


container_A1 = np.array(container_A1)
container_A1 = container_A1.T
container_G1 = np.array(container_G1)
container_G1 = container_G1.T


# label the slidding window, return tag, start by a boundary closer to target activity, include more 'others' sample
def label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend):
    # initialize overall start and end stamp with head_tail_extend
    start, end, taglist = [], [], []
    for extend in head_tail_extend:
        for i, (_, act) in enumerate(stamp.items()):
            start.append(0 if act[0] - extend < 0 else act[0] - extend)
            end.append(act[1] + extend)
    # tag corresponding activity for sliding window
    # that has an overlap with activity larger than overlap_tolerance = frequency * react_time
    # and other position 'others'
    for k in range(len(window_size)):
        tags = {}
        for j in range(len(start)):
            for i in range(start[j], end[j], window_size[k]//2):
                tags[i], s, e = 'others', i, i + window_size[k]
                for name, act in stamp.items():
                    if (e - overlap_tolerance[k] < act[0]) or (s + overlap_tolerance[k] > act[1]):
                        continue
                    else:
                        if name.startswith('r'):
                            tags[i] = 'rope_skipping'
                        else:
                            tags[i] = name
        taglist.append(tags)
    return taglist


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


# get data on each axis
def get_coor(container_A1, container_G1, reg=None):
    xa, ya, za = np.array(container_A1)[:, 0], np.array(container_A1)[:, 1], np.array(container_A1)[:, 2]
    xw, yw, zw = np.array(container_G1)[:, 0], np.array(container_G1)[:, 1], np.array(container_G1)[:, 2]
    if reg == 'stdr':
        xa, ya, za = standardization(xa), standardization(ya), standardization(za)
        xw, yw, zw = standardization(xw), standardization(yw), standardization(zw)
    elif reg == 'norm':
        xa, ya, za = normalization(xa), normalization(ya), normalization(za)
        xw, yw, zw = normalization(xw), normalization(yw), normalization(zw)
    return xa, ya, za, xw, yw, zw


# 根据时间窗口取得19个时域频域特征
def get_aw_feature(walk_start_stamp, walk_end_stamp, container_A1, container_G1):
    xa, ya, za, xw, yw, zw = get_coor(container_A1, container_G1, reg='norm')
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


def process():
    stamp = {}
    # s1, e1, s2, e2 = 56750, 65740, 119700, 123500
    # stamp['rope_skipping'] = [s2, e2]
    # stamp['sit_up1'] = [s1, e1]

    # s1, e1, s2, e2, s3, e3, s4, e4 = 17580, 18740, 38220, 40610, 41370, 42150, 42650, 43130
    # s5, e5 = 112100, 114300
    # stamp['rope_skipping1'] = [s1, e1]
    # stamp['rope_skipping2'] = [s2, e2]
    # stamp['rope_skipping3'] = [s3, e3]
    # stamp['rope_skipping4'] = [s4, e4]
    # stamp['sit_up1'] = [s5, e5]

    s1, e1, s2, e2 = 70750, 72960, 111800, 116500
    stamp['rope_skipping'] = [s2, e2]
    stamp['sit_up1'] = [s1, e1]

    # 调试绘图
    # sos = signal.butter(3, 1, 'lp', fs=52, output='sos')
    # accdata = signal.sosfilt(sos, [g[-1] for g in container_G1])
    # plt.plot([sumvector(a[1], a[2], a[3]) for a in container_A1], label='sum acceleration')
    # plt.plot([g[2] for g in container_G1], label='z angular velocity')
    # # plt.plot(accdata, label='high pass filter wave')
    # plt.legend()
    # plt.plot([a[1] for a in container_G1], label='wx')
    # plt.plot([a[2] for a in container_G1], label='wy')
    # plt.plot([a[3] for a in container_G1], label='wz')
    # plt.plot([d[0] for d in delta_angle], label='theta')
    # plt.plot([d[1] for d in delta_angle], label='fai')
    # plt.plot([d[2] for d in delta_angle], label='psi')
    # plt.show()

    # window_time = [0.5,1,3,5,8,10]  # s time of a window frame
    window_time = [10]
    print('window time is:', window_time)
    overlap_rate = 0.2
    react_time = np.multiply(overlap_rate, window_time)  # s time for algrm to get enough data (for window analysis)
    print('react time is:', react_time)
    frequency = 52  # Hz designated frequency of sampling
    window_size = np.rint(np.multiply(window_time, frequency)).astype(int)  # number of samples in a window frame
    overlap_tolerance = np.rint(np.multiply(react_time, frequency)).astype(
        int)  # minimum number of samples for overlap the act would be considered on
    extend_coefficient = 1.5
    head_tail_extend = np.multiply(window_size, extend_coefficient).astype(
        int)  # number of samples used to over samling before and after the series to acquire more generalized data
    # assert (window_time > react_time)  # window frame is always longer than the reaction period

    # tag dict
    taglist = label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend)

    # feature dict
    ftlist = []
    for i in range(len(taglist)):
        tags, size, ft = taglist[i], window_size[i], {}
        print('This is window time of {}s'.format(window_time[i]))
        for k, _ in tags.items():
            print(k)
            ft[k] = get_aw_feature(k, k + size, container_A1, container_G1)
        ftlist.append(ft)

    print(ftlist)
    print('feature num is:', len(next(iter(ftlist[0].values()))[0]))
    print('total dim is:', len(next(iter(ftlist[0].values()))))
    # save the feature and tags in pickel file
    pd.DataFrame(ftlist).to_pickle(r'.\multiclass_new_tag\feature_child3_w10_norm.pickle')
    pd.DataFrame(taglist).to_pickle(r'.\multiclass_new_tag\tag_child3_w10_norm.pickle')
    print('preprocess over')


if __name__ == '__main__':
    process()