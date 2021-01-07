# kid_link1
# s1, e1, s2, e2, s3, e3, s4, e4 = 0, 6127, 56750, 65740, 119700, 123500, 166500, 193400
# stamp['walking'] = [(s1, e1)]
# stamp['rope_skipping'] = [(s2, e2)]
# stamp['sit_up1'] = [(s3, e3)]
# stamp['jog'] = [(s4, e4)]
#
# ext = 1000

# # 输出mat文件
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_仰卧起坐.mat', {'GYRO': container_G1[s2:e2], 'ACCEL': container_A1[s2:e2]})
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_跑步.mat', {'GYRO': container_G1[s4:e4], 'ACCEL': container_A1[s4:e4]})
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_跳绳.mat', {'GYRO': container_G1[s3:e3], 'ACCEL': container_A1[s3:e3]})
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_走路.mat', {'GYRO': container_G1[s1:e1], 'ACCEL': container_A1[s1:e1]})
# 加长版
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_仰卧起坐_加长.mat', {'GYRO': container_G1[s2-ext:e2+ext], 'ACCEL': container_A1[s2-ext:e2+ext]})
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_跑步_加长.mat', {'GYRO': container_G1[s4-ext:e4+ext], 'ACCEL': container_A1[s4-ext:e4+ext]})
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_跳绳_加长.mat', {'GYRO': container_G1[s3-ext:e3+ext], 'ACCEL': container_A1[s3-ext:e3+ext]})
# scipy.io.savemat('kid_segmentation_mat_data/谭思远_10点_走路_加长.mat', {'GYRO': container_G1[s1:e1+ext], 'ACCEL': container_A1[s1:e1+ext]})

# kid_link2
# so1, eo1, so2, eo2, so3, eo3, so4, eo4 = 21420, 22240, 22380, 22500, 22720, 22880, 22990, 23450
# s1, e1, s2, e2, s3, e3, s4, e4 = 17580, 18740, 38220, 40610, 41370, 42150, 42650, 43130
# s5, e5 = 112100, 114300
# s6, e6, s7, e7 = 79980, 81790, 82710, 103500
# stamp['rope_skipping1'] = [(so1, eo1), (so2, eo2), (so3, eo3), (so4, eo4)]
# stamp['rope_skipping2'] = [(s1, e1), (s2, e2), (s3, e3), (s4, e4)]
# stamp['sit_up1'] = [(s5, e5)]
# stamp['walking'] = [(s6, e6), (s7, e7)]

# scipy.io.savemat('胡欣怡_11-12点_仰卧起坐.mat', {'GYRO': container_G1[s5:e5], 'ACCEL': container_A1[s5:e5]})
# scipy.io.savemat('胡欣怡_11-12点_跳绳1.mat', {'GYRO': container_G1[so1:eo4], 'ACCEL': container_A1[so1:eo4]})
# scipy.io.savemat('胡欣怡_11-12点_跳绳2.mat', {'GYRO': container_G1[s1:e4], 'ACCEL': container_A1[s1:e4]})
# scipy.io.savemat('胡欣怡_11-12点_走路.mat', {'GYRO': container_G1[s6:e7], 'ACCEL': container_A1[s6:e7]})
# scipy.io.savemat('kid_segmentation_mat_data/胡欣怡_11-12点_仰卧起坐_加长.mat', {'GYRO': container_G1[s5 - ext:e5 + ext], 'ACCEL': container_A1[s5 - ext:e5 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/胡欣怡_11-12点_跳绳1_加长.mat', {'GYRO': container_G1[so1 - ext:eo4 + ext], 'ACCEL': container_A1[so1 - ext:eo4 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/胡欣怡_11-12点_跳绳2_加长.mat', {'GYRO': container_G1[s1 - ext:e4 + ext], 'ACCEL': container_A1[s1 - ext:e4 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/胡欣怡_11-12点_走路_加长.mat', {'GYRO': container_G1[s6 - ext:e7 + ext], 'ACCEL': container_A1[s6 - ext:e7 + ext]})


# # kid_link3
# s1, e1 = 14800, 49120
# s2, e2 = 70750, 73130
# s3, e3, s4, e4, s5, e5 = 111800, 114000, 114800, 117200, 117800, 118600
# s6, e6 = 136800, 157100
# stamp['walking'] = [(s1, e1)]
# stamp['sit_up1'] = [(s2, e2)]
# stamp['rope_skipping'] = [(s3, e3), (s4, e4), (s5, e5)]
# stamp['jog'] = [(s6, e6)]
# scipy.io.savemat('kid_segmentation_mat_data/黄兆宇_2-3点_仰卧起坐_加长.mat', {'GYRO': container_G1[s2 - ext:e2 + ext], 'ACCEL': container_A1[s2 - ext:e2 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/黄兆宇_2-3点_跳绳1_加长.mat', {'GYRO': container_G1[s3 - ext:e5 + ext], 'ACCEL': container_A1[s3 - ext:e5 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/黄兆宇_2-3点_跑步_加长.mat', {'GYRO': container_G1[s6 - ext:e6 + ext], 'ACCEL': container_A1[s6 - ext:e6 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/黄兆宇_2-3点_走路_加长.mat', {'GYRO': container_G1[s1 - ext:e1 + ext], 'ACCEL': container_A1[s1 - ext:e1 + ext]})

# # kid 4 is problematic in terms of differentiating jogging and walking
# # kid_link4
# s1, e1, s2, e2, s3, e3 = 26290, 46740, 48230, 51600, 52070, 54990
# s4, e4, s5, e5 = 86140, 88190, 88800, 91410
# s6, e6 = 67670, 71670
# s7, e7 = 129900, 153800
# stamp['walking'] = [(s1, e1), (s2, e2), (s3, e3)]
# stamp['rope_skipping'] = [(s4, e4), (s5, e5)]
# stamp['sit_up1'] = [(s6, e6)]
# stamp['jog'] = [(s7, e7)]

# scipy.io.savemat('陈博思_13点_仰卧起坐.mat', {'GYRO': container_G1[s6:e6], 'ACCEL': container_A1[s6:e6]})
# scipy.io.savemat('陈博思_13点_跳绳.mat', {'GYRO': container_G1[s4:e5], 'ACCEL': container_A1[s4:e5]})
# scipy.io.savemat('陈博思_13点_走路.mat', {'GYRO': container_G1[s1:e3], 'ACCEL': container_A1[s1:e3]})
# scipy.io.savemat('陈博思_13点_跑步.mat', {'GYRO': container_G1[s7:e7], 'ACCEL': container_A1[s7:e7]})
# scipy.io.savemat('kid_segmentation_mat_data/陈博思_13点_仰卧起坐_加长.mat', {'GYRO': container_G1[s6 - ext:e6 + ext], 'ACCEL': container_A1[s6 - ext:e6 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/陈博思_13点_跳绳_加长.mat', {'GYRO': container_G1[s4 - ext:e5 + ext], 'ACCEL': container_A1[s4 - ext:e5 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/陈博思_13点_走路_加长.mat', {'GYRO': container_G1[s1 - ext:e3 + ext], 'ACCEL': container_A1[s1 - ext:e3 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/陈博思_13点_跑步_加长.mat', {'GYRO': container_G1[s7 - ext:e7 + ext], 'ACCEL': container_A1[s7 - ext:e7 + ext]})
# print('ok')


# # kid_link5
# s1, e1, s2, e2= 609, 37730, 52600, 78370
# stamp['walking'] = [(s1, e1)]
# stamp['jog'] = [(s2, e2)]
# scipy.io.savemat('kid_segmentation_mat_data/汪泽成_3-4点_走路_加长.mat', {'GYRO': container_G1[s1 - ext:e1 + ext], 'ACCEL': container_A1[s1 - ext:e1 + ext]})
# scipy.io.savemat('kid_segmentation_mat_data/汪泽成_3-4点_跑步_加长.mat', {'GYRO': container_G1[s2 - ext:e2 + ext], 'ACCEL': container_A1[s2 - ext:e2 + ext]})
