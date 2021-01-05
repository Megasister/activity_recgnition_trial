import math


# 获得合成矢量
def sumvector(*args):
    return math.sqrt(sum([pow(arg, 2) for arg in args]))


# 取得四元数
def q_iteration(q0, q1, q2, q3, wx, wy, wz, delta):
    q0 = q0 + (-wx * q1 - wy * q2 - wz * q3) * delta / 2
    q1 = q1 + ( wx * q0 - wy * q3 + wz * q2) * delta / 2
    q2 = q2 + ( wx * q3 + wy * q0 - wz * q1) * delta / 2
    q3 = q3 + (-wx * q2 + wy * q1 - wz * q0) * delta / 2
    # 归一化
    s = sumvector(q0, q1, q2, q3)
    return q0 / s, q1 / s, q2 / s, q3 / s


# 根据四元数算出姿态角每步的变化量
def get_quadranion(container_G):
    q0, q1, q2, q3 = 1, 0, 0, 0
    Q = []
    for g in container_G:
        gx, gy, gz = g[1], g[2], g[3]
        delta = 0.02
        q0, q1, q2, q3 = q_iteration(q0, q1, q2, q3, gx, gy, gz, delta)
        Q.append([q0, q1, q2, q3])

    delta_angle = []
    for q in Q:
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        del_theta = -math.asin(2 * (q1 * q2 - q0 * q3))
        del_fai = math.atan(2 * (q1 * q3 - q0 * q1) / (q3 ** 2 + q2 ** 2 - q1 ** 2 + q0 ** 2))
        del_psi = math.atan(2 * (q1 * q2 - q0 * q3) / (q1 ** 2 + q0 ** 2 - q3 ** 2 - q2 ** 2))
        delta_angle.append([del_theta, del_fai, del_psi])

    return delta_angle
