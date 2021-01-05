#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
from datetime import datetime
import csv


keyword_list = [
    'healthd: battery l=',
    'Hub log:skipping_count_process',
    'Hub log:fever_detection_process',
    'Hub log:motion_valid_process',
    'Hub log:calorie_process'
]


def read_file(file_path):
    file_handle = open(file_path, 'r', encoding='utf-8')
    lines = file_handle.readlines()
    file_handle.close()
    ret = []
    for line in lines:
        for keyword in keyword_list:
            if line.find(keyword) >= 0:
                ret.append(line)
                break
    return ret


def transfer_to_time_info(dt):
    # 转换如下格式的时间为timestamp格式
    # 2019-09-27 08:00:00
    # 2019-09-04__00-00-27
    dt = dt.replace('-', ' ')
    dt = dt.replace("_", " ")
    dt = dt.replace(":", " ")

    # 过滤掉字符串里面除了'0123456789 '之外的字符
    dt = filter(lambda ch: ch in '0123456789 ', dt)

    my_time = ''

    for ch in dt:
        if len(my_time) == 0 and ch == ' ':
            continue

        my_time = my_time + ch

    try:
        ret = time.strptime(my_time, "%Y %m %d  %H %M %S")
    except:
        return None
    else:
        return ret


def time_to_stamp(dt):
    time_array = transfer_to_time_info(dt)
    if time_array is None:
        return 0
    timestamp = time.mktime(time_array)

    return int(timestamp)


def get_line_timestamp(line):
    if line.find(keyword_list[0]) < 0:
        return 0
    time_str_start = line.find('time=[')
    if time_str_start < 0:
        return 0
    time_str_start += len('time=[')

    line_time = '2020' + '-' + line[time_str_start:time_str_start + 14]
    line_time_stamp = time_to_stamp(line_time)

    return line_time_stamp


def get_machine_time(line):
    num_seg = line[5:16]
    num_list = num_seg.split('.')
    return int(num_list[0])


def export_motion_data(src_lines, keyword):
    ret = []
    machine_time_start = 0
    timestamp_start = 0

    for line in src_lines:
        if line.find(keyword_list[0]) >= 0:
            machine_time_start = get_machine_time(line)
            timestamp_start = get_line_timestamp(line)

        if line.find(keyword) < 0:
            continue

        cur_machine_time = get_machine_time(line)

        cur_timestamp = timestamp_start + (cur_machine_time - machine_time_start)
        cur_timestamp = abs(cur_timestamp)
        list_val = [str(cur_timestamp)]

        dateArray = datetime.fromtimestamp(cur_timestamp)
        dateStr = dateArray.strftime("%Y-%m-%d %H:%M:%S")
        list_val.append(dateStr)

        st_input = line.find('child_data:') + len('child_data:')
        end_input = line.find(',output:')
        st_output = end_input + len(',output:')
        if st_input < 0 or end_input < 0:
            continue

        input_val = line[st_input:end_input]
        list_val += input_val.split(' ')
        output_val = line[st_output:len(line)-1]
        list_val += output_val.split(' ')
        ret.append(list_val)
    return ret


def write_csv_file(file_name, data):
    f = open(file_name, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(data)
    f.close()


if __name__ == '__main__':
    time = '2020-12-26__11-04-56'
    lines = read_file('E:\\Folder_20201226\\kernel\\Log.kernel_{}.log'.format(time))

    motion_data = export_motion_data(lines, keyword_list[1])
    write_csv_file('E:\\Folder_20201226\\csv\\motion_data_{}.csv'.format(time), motion_data)

    # fever_data = export_motion_data(lines, keyword_list[2])
    # write_csv_file('E:\\Folder_20201226\\csv\\fever_data_{}.csv'.format(time), fever_data)
    #
    # motion_valid_data = export_motion_data(lines, keyword_list[3])
    # write_csv_file('E:\\Folder_20201226\\csv\\motion_valid_data_{}.csv'.format(time), motion_valid_data)
    #
    # calorie_data = export_motion_data(lines, keyword_list[4])
    # write_csv_file('E:\\Folder_20201226\\csv\\calorie_data_{}.csv'.format(time), calorie_data)
