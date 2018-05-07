from quatematics import Quat


def propagate_imu(w, s):
    DT = imu_time - last_update_time
    # DT=0.005

    # Propagate orientation

    # w_hat = AngularRate(w - bias_g_est)
    w_hat = w - bias_g_est

    q_Ik_to_I_increment = Quat(np.dot(0.5 * Quat.eye.Xi(), w_hat) * DT + Quat.eye.asColVector())
    q_Ik_to_I_increment.normalize()
    q_W_to_I = q_Ik_to_I_increment * q_W_to_Ik
    q_W_to_I_history = np.append(q_W_to_I_history, q_W_to_I.asColVector(), axis=1)
    sim_time = np.append(sim_time, np.array([clock.now()]))

    # Integrate acceleration
    s_hat = AngularRate(s - bias_a_est)

    v_W_hat = v_W_hat + (np.dot(q_W_to_I.asRM(), s_hat) + g_W) * DT
    v_W_history = np.append(v_W_history, v_W_hat, axis=1)

    # Integrate velocity
    p_IW_hat = p_IW_hat + v_W_hat * DT
    p_W_history = np.append(p_W_history, p_IW_hat, axis=1)
