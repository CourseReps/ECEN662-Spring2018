from quatematics import Quat


def viconToTargetMap(q_vicon_I, q_vicon_t, p_vicon_I, p_vicon_t):
    # Change vicon quaternions from wxyz to xyzw
    q_vicon_I = Quat(q_vicon_I)
    q_vicon_t = Quat(q_vicon_t)

    # Returns both attitude of IMU and its position with respect to the target
    q_t_I = q_vicon_I * q_vicon_t.inverse()
    q_t_I.normalize()
    p_t_I = q_vicon_t.asDCM().dot(p_vicon_I - p_vicon_t)

    return q_t_I, p_t_I
