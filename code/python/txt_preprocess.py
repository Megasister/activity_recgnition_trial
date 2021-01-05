import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import parameter


# n_class = ['jog', 'others', 'rope_skipping', 'sit_up1', 'sit_up2', 'walking']
n_class = ['others', 'rope_skipping', 'sit_up1']


def get_data(link):
    file = open(
        # second data series
        link,
        encoding="UTF-8",
        errors='ignore'
    )
    line = file.readline()
    GYRO, ACCEL = [], []
    while line:
        # print(line)
        if 'QSensorTest:' in line.split():
            if "GYRO" in line.split():
                GYRO.append(line.split()[:2] + line.split()[6:])
                # print(line.split()[:2] + line.split()[6:])
            if 'ACCEL' in line.split():
                ACCEL.append(line.split()[:2] + line.split()[6:])
                # print(line.split())

        line = file.readline()
    file.close()
    return GYRO, ACCEL


# 获得不同运动时间段对应的开始和结束时间戳
def get_timestamp_step_before_container(container, start_time):
    start = 0
    for i in range(len(container)):
        if pd.to_datetime(container[i][1]) > start_time:
            start = i
            break
    return start


# align two container at beginning
def align_init(g_start_time, a_start_time, accel, gyro):
    ACC, GY = [], []
    if g_start_time > a_start_time:
        for i in range(len(accel)):
            if pd.to_datetime(accel[i][1]) > g_start_time:
                ACC = accel[i:]
                break
        return ACC, gyro
    else:
        for k in range(len(gyro)):
            if pd.to_datetime(gyro[k][1]) > a_start_time:
                GY = gyro[k:]
                break
        return accel, GY


# align GY or ACC when one is upfront too far
# arg1: the one upfront at designated time slot
# arg2: one lag back
# arg3: designated time slot
def align_by_time(accel, gyro, interval_time, frequency, delay_max_time):
    ACC, GY, cut_point = [], [], 0
    interval_start = get_timestamp_step_before_container(accel, interval_time)

    if pd.to_datetime(accel[interval_start][1]) < pd.to_datetime(gyro[interval_start][1]):
        print('wrong order')
    for i in range(interval_start, interval_start + frequency * delay_max_time):
        if pd.to_datetime(accel[interval_start][1]) < pd.to_datetime(gyro[i][1]):
            cut_point = i
            break
    # print(interval_start)
    return gyro[:interval_start] + gyro[cut_point:]


def time_validate(container_a, container_g):
    tolerance = pd.Timedelta('00:00:05')
    for i in range(len(container_a)):
        try:
            diff = pd.to_datetime(container_g[i][1]) - pd.to_datetime(container_a[i][1])
            if diff > tolerance:
                print(i)
                print(container_a[i])
                print(container_g[i])
                print(diff)
        except:
            continue


def get_timestamp_step(container, start_time, end_time):
    start, end = 0, 0
    for i in range(len(container)):
        if container[i][0] > start_time:
            start = i
            break
    for k in range(start, len(container)):
        if container[k][0] > end_time:
            end = k
            break
    return start, end


def preprocess(link):
    G1, A1 = get_data(link)
    # Align the timestamp as close as possible uniformly conforming to the latter one
    G1_start_time = pd.to_datetime(G1[0][1])
    A1_start_time = pd.to_datetime(A1[0][1])
    # print(GYRO_start_time)
    # print(ACCEL_start_time)
    ACC1, GY1 = align_init(G1_start_time, A1_start_time, A1, G1)
    GY1 = align_by_time(ACC1, GY1, pd.to_datetime('07:43:50'), frequency=50, delay_max_time=1000)
    ACC1 = align_by_time(GY1, ACC1, pd.to_datetime('07:59:22.108'), frequency=50, delay_max_time=1000)

    # time_validate(ACC1, GY1)
    # contains time and coord
    container_A1, container_G1 = [], []
    for category in [ACC1, GY1]:
        for data in category:
            nums = data[3].split(',')
            x = float(nums[0][13:])
            y = float(nums[1][4:])
            z = float(nums[2][4:])
            if category == ACC1:
                container_A1.append([pd.to_datetime(data[1]), x, y, z])
            if category == GY1:
                container_G1.append([pd.to_datetime(data[1]), x, y, z])

    return container_A1, container_G1


if __name__ == '__main__':
    # preprocess(parameter.link1)
    pass