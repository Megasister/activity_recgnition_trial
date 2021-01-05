import numpy as np
import pandas as pd
import parameter
import txt_preprocess
import quadranion
import get_feature


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


# label the slidding window, return tag
def label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend):
    # initialize overall start and end stamp with head_tail_extend
    start, end, taglist = [], [], []
    for extend in head_tail_extend:
        for i, (_, act) in enumerate(stamp.items()):
            if i == 0:
                start.append(0 if act[0] - extend < 0 else act[0] - extend)
            if i == len(stamp)-1:
                end.append(act[1] + extend)
    # tag corresponding activity for sliding window
    # that has an overlap with activity larger than overlap_tolerance = frequency * react_time
    # and other position 'others'
    for j in range(len(start)):
        tags = {}
        for i in range(start[j], end[j], window_size[j]//2):
            tags[i], s, e = 'others', i, i + window_size[j]
            for name, act in stamp.items():
                if (e - overlap_tolerance[j] < act[0]) or (s + overlap_tolerance[j] > act[1]):
                    continue
                else:
                    # tags[i] = 'rope_skipping'
                    tags[i] = name
        taglist.append(tags)
    return taglist


def load_init_data():
    container_A1, container_G1 = txt_preprocess.preprocess(parameter.link1)
    print('write over')

    # # 取得四元数
    delta_angle = quadranion.get_quadranion(container_G1)
    print(delta_angle)

    stamp = {}
    # # for link_sk
    # s1, e1, s2, e2, s3, e3, s4, e4 = 1100, 1600, 2000, 3300, 3880, 4050, 4500, 5440
    # stamp['rope_skipping'] = [s1, e1]
    # stamp['rope_skipping1'] = [s2, e2]
    # stamp['rope_skipping2'] = [s3, e3]
    # stamp['rope_skipping3'] = [s4, e4]

    # for link1, conservative policy which confines the range of each activity clearly
    # allow sliding window to swipe the entire range
    # if sliding window has an overlapping range with specified ranges over 2s, tag the window as positive
    s1, e1, s2, e2, s3, e3, s4, e4, s5, e5 = 1550, 3520, 6900, 9350, 11480, 17580, 20800, 25260, 28670, 34740
    stamp['sit_up1'] = [s1, e1]
    stamp['sit_up2'] = [s2, e2]
    stamp['jog'] = [s3, e3]
    stamp['rope_skipping'] = [s4, e4]
    stamp['walking'] = [s5, e5]

    print(container_A1)

    # # for link2
    # s1, e1, s2, e2, s3, e3, s4, e4 = 37900, 40150, 42950, 45050, 52400, 58450, 65040, 66350
    # stamp['sit_up1'] = [s1, e1]
    # stamp['sit_up2'] = [s2, e2]
    # stamp['jog'] = [s3, e3]
    # stamp['rope_skipping'] = [s4, e4]

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

    # tag dict
    taglist = label_target_activity(stamp, window_size, overlap_tolerance, head_tail_extend)
    # feature dict
    ftlist = []
    for i in range(len(taglist)):
        tags, size, ft = taglist[i], window_size[i], {}
        print('This is window time of {}s'.format(window_time[i]))
        for k, _ in tags.items():
            print(k)
            ft[k] = get_feature.get_aw_feature(k, k+size, container_A1, container_G1)
        ftlist.append(ft)

    # print(ftlist)
    print('feature num is:', len(next(iter(ftlist[0].values()))[0]))
    print('total dim is:', len(next(iter(ftlist[0].values()))))
    # save the feature and tags in pickel file
    # pd.DataFrame(ftlist).to_pickle(r'.\multiclass_new_tag\feature2_w5.pickle')
    # pd.DataFrame(taglist).to_pickle(r'.\multiclass_new_tag\tag2_w5.pickle')
    # print('preprocess over')


if __name__ == '__main__':
    load_init_data()