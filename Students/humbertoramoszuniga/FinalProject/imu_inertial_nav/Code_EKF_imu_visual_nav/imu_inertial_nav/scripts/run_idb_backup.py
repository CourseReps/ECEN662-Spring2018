from Hest import Hest, Correction
import numpy as np
import os
from quatematics import AngularRate, Quat, skew_symmetric
from rpmsat import database_from_file, Clock

import matplotlib.pylab as plt
from multiplot2d import MultiPlotter
from scipy.interpolate import interp1d

import cv2

np.set_printoptions(linewidth=999999)
#######
# INPUT
#######

# where is the input file?
# database_path = "/data/all_noise.idb"
# database_path = "/data/feature_noise_only.idb"
# database_path = "/data/imu_noise_only.idb"
database_path = "/data/no_noise.idb"
# database_path = "/data/data_2.idb"
# database_path = "/data/data_3.idb"

##The data for the report was generated with trajectory 3. This is data_3.idb.
# Comment lines with comparisons "if camera" deactivate the video generation.


# database_path = "/data/all_noise.idb"

# Trajectory to generate video. This is just demostrative. No good results since bias was not initialized properly, but
# just need the video. This file was very big (1GB), so it was not included in submitted zip.
# database_path = "/data/data_3.idb"

# database_path = "/Dropbox/aims_db/humberto/all_noise.idb"
# database_path = "/Dropbox/aims_db/humberto/test.idb"

# Create a video writer to save the resultant video
# fourcc = cv2.cv.CV_FOURCC(*'XVID')
# video_out = cv2.VideoWriter('Result.mpg',-1, 10, (640, 512))
# video_out = cv2.VideoWriter('/home/humberto/catkin_ws/src/klt/src/outpy.avi',fourcc, 40, (640,512))
# the simulated clock with a finite time step
dt = 0.00001
DT = 0.001
last_update_time = 0.
time_old = 0.

# when does the clock start?
t0 = 0.
# when does the clock end?
tf = 35.
###############
# PROCESS INPUT
###############

# Camera parameters and intersensor parameters
fx = 432.  # pixels
fy = 432.  # pixels
cx = 317.  # pixels
cy = 252.  # pixels

# Intrinsics
cameraMatrix = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1]).reshape(3, 3)
distorsion_coeffs = np.array([-0.213922, 0.088003, 0.000180, -0.000312, 0.000000])
# Number of features.
m = 16
# Testing
# p_c_i = np.array([0.0254 * 2, -0.045, -0.025]).reshape(3, 1)
# Working nicely
# p_c_i = np.array([0.0508, -0.045, -0.025]).reshape(3, 1)
# initial guess
p_c_i = np.array([0.0508, -0.0508, 0.]).reshape(3, 1)
q_ic = Quat([0.5, 0.5, 0.5, 0.5])

# load input file
idb = database_from_file(os.environ['HOME'] + database_path)

# create clock, which handles the simulation "time"
# each sensor gets a reference to the clock
# the sensors will use the clock to "know" when new measurements are
# available
clock = Clock(t0, tf, dt)

# create view of sensors
imu = idb.imu
feature_detector = idb.feature_detector
camera = idb.camera

# add the clock to the sensors
imu.add_clock(clock)
feature_detector.add_clock(clock)
if camera != None:
    camera.add_clock(clock)

# Initialize quaternions, no rotation if no arguments.
ref_time = idb.reference.kinematics.time
ref_position_history = idb.reference.kinematics.position
ref_attitude_history = idb.reference.kinematics.attitude

# g_vicon = np.array([0, 0, -9.81]).reshape(3, 1)
g_vicon = np.array([0, 0, -9.79343]).reshape(3, 1)
if not hasattr(idb, 'target_attitude'):
    g_W = g_vicon
else:
    g_W = idb.target_attitude.asDCM().dot(g_vicon)

# State initialization.
# print q_W_to_I_history
q_W_to_Ik = Quat(ref_attitude_history[:, 0:1])
q_W_to_I = Quat()
v_W_hat = np.array([0., 0., 0.]).reshape(3, 1)
p_IW_hat = ref_position_history[:, 0:1]

# Good value for bias for hardware testing. Estimate initial bias. Data must be still for 5 seconds.
w_data = np.zeros((3, len(idb.imu.data)))
s_data = np.zeros((3, len(idb.imu.data)))

for k, msg in enumerate(idb.imu.data):
    w_data[:, k] = msg.angular_rate.flatten()
    s_data[:, k] = msg.acceleration.flatten()

n_sec = 5
rate = 200
k_max = n_sec * rate
w_bias_est = np.mean(w_data[:, :k_max], axis=1)
# print w_bias_est

# bias_g_est = np.array(w_bias_est).reshape(3, 1)
# bias_g_est = np.array([-0.01317, -0.0018, -0.01]).reshape(3, 1)
# bias_true = np.array([-0.00344207,  0.0006554 ,  0.0036169 ]).reshape(3,1)
# bias_a_est = np.mean(s_data[:,:k_max], axis=1).reshape(3,1)
# bias_a_est = np.array([0.,0,0]).reshape(3,1)
bias_a_est = np.array([[2.04553032e-03],
                       [1.79890027e-03],
                       [-5.22963386e-05]])

bias_g_est = np.array([0, 0, 0]).reshape(3, 1)

q_W_to_I_history = ref_attitude_history[:, 0:1]  # Argument means returns as a column
p_W_history = ref_position_history[:, 0:1]  # Argument means returns as a column
v_W_history = ref_position_history[:, 0:1]  # Argument means returns as a column
w_history = np.array([0., 0, 0]).reshape(3, 1)
w_hat_history = np.array([0., 0, 0]).reshape(3, 1)
bias_g_est_history = np.array([0., 0, 0]).reshape(3, 1)
bias_true_history = np.array([0., 0, 0]).reshape(3, 1)

sim_time = np.array([0.])

# Initialize the covariance matrix
# Testing P
P = np.diag(np.concatenate([[0.0025] * 3, [0.001] * 3, [0.001] * 3, [0.1] * 3, [0.02] * 3]))

# Original P
# P = np.diag(np.concatenate([[0.0027] * 3, [0.001] * 3, [0.01] * 3, [0.02] * 3, [0.02] * 3]))
sigmas_position_history = np.array([0., 0., 0]).reshape(3, 1)
sigmas_biasg_history = np.array([0., 0., 0]).reshape(3, 1)

# Q testing
# Q = np.diag(np.concatenate([[0.001] * 3, [0.015] * 3, [0.00004] * 3, [0.0002] * 3])) ** 2
# Testing
Q = np.diag(np.concatenate([[0.001] * 3, [0.015] * 3, [0.0004] * 3, [0.0002] * 3])) ** 2
# Original Q
# Q = np.diag(np.concatenate([[0.001] * 3, [0.015] * 3, [0.00004] * 3, [0.0002] * 3])) ** 2
# Testing
# R = np.diag(([0.5] * m * 2)) ** 2
R = np.diag(([0.5] * m * 2)) ** 2
# Original
# R = np.diag(([0.05] * m * 2)) ** 2
# Ruco target positions
h_hat_i = Hest([0., 0, 0])
# w = np.array([0., 0, 0])
# s = np.array([0., 0, 0])
# w_hat = np.array([0., 0, 0])

num_imu_measure = 0
initial_pose_estimated = 0
y0 = np.zeros(16).reshape(16, 1)
imu_time_history = np.array([0])
while clock():
    uv_plot_list = []
    uv_measured_list = []

    if imu.poll():
        imu_time = imu.latest().true_time
        if num_imu_measure > 0:
            w = imu.latest().angular_rate
            s = imu.latest().acceleration
            DT = imu_time - last_update_time
            w_history = np.append(w_history, np.array(w), axis=1)
            imu_time_history = np.append(imu_time_history, imu_time)

            # Propagate orientation

            # w_hat = AngularRate(w - bias_g_est)
            w_hat = w - bias_g_est
            # w_hat = w
            w_hat_history = np.append(w_hat_history, np.array(w_hat), axis=1)

            # bias_true_history = np.append(bias_true_history,bias_true,axis=1)
            # bias_g_est_history = np.append(bias_g_est_history,bias_g_est,axis=1)

            q_Ik_to_I = Quat(np.dot((1. / 2) * DT * w_hat.Omega() + np.eye(4), Quat.eye.asColVector()))
            # q_Ik_to_I = Quat(np.dot(0.5 * Quat.eye.Xi(), w_hat) * DT + Quat.eye.asColVector())
            q_Ik_to_I.normalize()
            q_W_to_I = q_Ik_to_I * q_W_to_Ik
            q_W_to_I.normalize()

            # Integrate acceleration
            s_hat = AngularRate(s - bias_a_est)
            v_W_hat = v_W_hat + (np.dot(q_W_to_I.asDCM().T, s_hat) + g_W) * DT
            # print "Velocity",v_W_hat
            # print "\n"

            # Integrate velocity
            p_IW_hat = p_IW_hat + v_W_hat * DT

            # Form the covariance differential equation
            F = np.zeros(15 * 15).reshape(15, 15)

            # Populate F
            # First row
            F[0:3, 0:3] = -skew_symmetric(w_hat)
            F[0:3, 9:12] = -np.eye(3)

            # Second row
            F[3:6, 6:9] = np.eye(3)

            # Third row
            F[6:9, 0:3] = -q_W_to_I.asDCM().T.dot(s_hat.skew())
            F[6:9, 12:15] = -q_W_to_I.asDCM().T

            # Populate G
            G = np.zeros(15 * 12).reshape(15, 12)
            # First row
            G[0:3, 0:3] = -np.eye(3)
            # Third row
            G[6:9, 3:6] = -q_W_to_I.asDCM().T
            # Fourth row
            G[9:12, 6:9] = np.eye(3)
            # Last row
            G[12:15, 9:12] = np.eye(3)

            # Reset the problem to estimate next state
            q_W_to_Ik = q_W_to_I
            # print "\n"
            P += (F.dot(P) + P.dot(F.T) + G.dot(Q).dot(G.T)) * DT

            # Compute bound errors from the covariance matrix
            P_position_diag = np.diagonal(P[3:6, 3:6]).reshape(3, 1)
            P_biasg_diag = np.diagonal(P[9:12, 9:12]).reshape(3, 1)

            # Save the values
            q_W_to_I_history = np.append(q_W_to_I_history, q_W_to_I.asColVector(), axis=1)
            sim_time = np.append(sim_time, np.array([clock.now()]))
            v_W_history = np.append(v_W_history, v_W_hat, axis=1)
            p_W_history = np.append(p_W_history, p_IW_hat, axis=1)
            sigmas_position_history = np.append(sigmas_position_history, np.sqrt(P_position_diag) * 3, axis=1)
            sigmas_biasg_history = np.append(sigmas_biasg_history, np.sqrt(P_biasg_diag) * 3, axis=1)

        num_imu_measure += 1
        last_update_time = imu_time
        print last_update_time, DT

    if feature_detector.poll():
        if num_imu_measure > 1:
            # print "clock time: ", clock.now()
            feature_measurement = feature_detector.latest()
            print "feature message stamp: ", feature_measurement.true_time
            DTi = (feature_measurement.true_time - last_update_time)
            print "Last update time: ", last_update_time
            ##Go forwards in time until the camera measurement time.
            # Propagate orientation

            # w_hat = AngularRate(w - bias_g_est)
            # w_hat = w - bias_g_est
            q_Ik_to_I = Quat(np.dot((1. / 2) * DTi * w_hat.Omega() + np.eye(4), Quat.eye.asColVector()))
            q_Ik_to_I.normalize()
            q_W_to_I = q_Ik_to_I * q_W_to_Ik
            q_W_to_I.normalize()

            # Integrate acceleration
            # s_hat = AngularRate(s - bias_a_est)

            v_W_hat = v_W_hat + (np.dot(q_W_to_I.asDCM().T, s_hat) + g_W) * DTi
            # print "Velocity",v_W_hat
            # print "\n"

            # Integrate velocity
            p_IW_hat = p_IW_hat + v_W_hat * DTi

            # # Form the covariance differential equation
            F = np.zeros(15 * 15).reshape(15, 15)

            # Populate F
            # First row
            F[0:3, 0:3] = -skew_symmetric(w_hat)
            F[0:3, 9:12] = -np.eye(3)
            # Second row
            F[3:6, 6:9] = np.eye(3)
            # Third row
            F[6:9, 0:3] = -q_W_to_I.asDCM().T.dot(s_hat.skew())
            F[6:9, 12:15] = -q_W_to_I.asDCM().T

            # Populate G
            G = np.zeros(15 * 12).reshape(15, 12)
            # First row
            G[0:3, 0:3] = -np.eye(3)
            # Third row
            G[6:9, 3:6] = -q_W_to_I.asDCM().T
            # Fourth row
            G[9:12, 6:9] = np.eye(3)
            # Last row
            G[12:15, 9:12] = np.eye(3)

            # Reset the problem to estimate next state
            q_W_to_Ik = q_W_to_I
            # print "\n"
            P += (F.dot(P) + P.dot(F.T) + G.dot(Q).dot(G.T)) * DTi

            # Compute bound errors from the covariance matrix
            P_position_diag = np.diagonal(P[3:6, 3:6]).reshape(3, 1)

            base_image = np.zeros((512, 640), dtype=np.uint8)
            base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)

            y_hat_len = 2 * m
            p_fW_len = 3 * m
            H = np.zeros(2 * m * 15).reshape(2 * m, 15)
            # Preallocate vectors for measurements and position features.
            y_f_hat = np.zeros(y_hat_len).reshape(y_hat_len, 1)
            y_measurement = np.zeros(y_hat_len).reshape(y_hat_len, 1)
            # p_fW = np.zeros(p_fW_len).reshape(p_fW_len, 1)

            for f_id, feature in enumerate(feature_measurement.features[:m]):
                row_slice2 = slice(2 * f_id, 2 * (f_id + 1))
                # row_slice3 = slice(3 * f_id, 3 * (f_id + 1))

                # Populate vectors
                y_measurement[row_slice2, :] = np.array([feature.u, feature.v]).reshape(2, 1)
                tt = np.array([feature.u, feature.v]).reshape(2, 1)
                cv2.undistort(np.array([feature.u, feature.v]), cameraMatrix, distorsion_coeffs)
                # print tt
                y_measurement[row_slice2, :] = tt

                uv_measured_list.append(y_measurement[row_slice2, :].flatten())

                # p_fW[row_slice3, :] = np.array([feature.p_f_w]).reshape(3, 1)

                # Compute H with the current slice, this corresponds to compute the i_th matrix H.

                # Features on camera frame.
                p_fC_i = Hest(q_ic.asDCM().dot(q_W_to_I.asDCM().dot(feature.p_f_w - p_IW_hat) - p_c_i))
                h_hat_i = p_fC_i

                # Features pixel coordinates estimate.
                y_f_hat_i = np.array(
                    [[fx * (h_hat_i.x / h_hat_i.z) + cx], [fy * (h_hat_i.y / h_hat_i.z) + cy]])
                y_f_hat[row_slice2, :] = y_f_hat_i
                uv_plot = y_f_hat_i.flatten()
                uv_plot_list.append(uv_plot)
                # cv2.circle(base_image,(int(feature.u),int(feature.v)), 4, (0, 255, 0), -1)
                # cv2.circle(base_image,(int(uv_plot[0]),int(uv_plot[1])), 4, (0, 0, 255), 1)
                # cv2.imshow("measurements",base_image)
                # cv2.waitKey(1)

                # partial (y_measurement/p_fC)
                partial_y_measure_p_fC = (1 / h_hat_i.z) * np.array([[fx, 0, -fx * h_hat_i.x / h_hat_i.z],
                                                                     [0, fy, -fy * h_hat_i.y / h_hat_i.z]]).reshape(2,
                                                                                                                    3)

                q_block = AngularRate(
                    q_W_to_I.asDCM().dot(np.array([p_fC_i.x, p_fC_i.y, p_fC_i.z]).reshape(3, 1) - p_IW_hat)).skew()
                pos_block = -q_W_to_I.asDCM()
                vel_block = np.zeros((3, 3))
                biasg_block = np.zeros(9).reshape(3, 3)
                biasa_block = np.zeros(9).reshape(3, 3)

                partial_p_FC_x_error = q_ic.asDCM().dot(
                    np.hstack((q_block, pos_block, vel_block, biasg_block, biasa_block)))

                H[row_slice2, 0:15] = partial_y_measure_p_fC.dot(partial_p_FC_x_error)

            # # Kalman Gain
            K = P.dot(H.T).dot(np.linalg.inv(H.dot(P).dot(H.T) + R))

            # Update covariance matrix P
            P = (np.eye(15) - K.dot(H)).dot(P)

            state_error = Correction(K.dot(y_measurement - y_f_hat))
            # print state_error.data

            # # Apply the corrections
            q_vec = np.append(0.5 * state_error.delq, 1)
            q_error = Quat(q_vec)
            #
            q_W_to_I = q_error * q_W_to_I
            q_W_to_I.normalize()
            p_IW_hat += state_error.delx.reshape(3, 1)
            v_W_hat += state_error.delv.reshape(3, 1)
            bias_g_est += state_error.delbg.reshape(3, 1)
            bias_a_est += state_error.delba.reshape(3, 1)

            last_update_time = feature_measurement.true_time

            if camera != None and camera.poll():
                print camera.latest().true_time
                base_image = camera.latest().image.copy()
                base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
                dst = base_image

                base_image = cv2.undistort(base_image, cameraMatrix, distorsion_coeffs)

                #             #overlap predicted measurements on the image.
# for uv_plot,uv_measured in zip(uv_plot_list,uv_measured_list):
#                     # cv2.circle(base_image, (int(uv_measured[0]), int(uv_measured[1])), 4, (0, 255, 0), -1)
#                     # cv2.circle(base_image, (int(uv_plot[0]), int(uv_plot[1])), 4, (0, 0, 255), 1)
#                     cv2.imshow("predicted measurements", base_image)
#                 video_out.write(base_image)
#                 cv2.waitKey(1)
# video_out.release()
# cv2.destroyAllWindows()
############
# INTERPOLATE
##############

est_data_list = [p_W_history, q_W_to_I_history]
ref_data_list = [ref_position_history, ref_attitude_history]
error_data_list = []
for est_data, ref_data in zip(est_data_list, ref_data_list):
    interp_fun = interp1d(sim_time, est_data, axis=1, bounds_error=False)
    interp_vals = interp_fun(ref_time)
    error = ref_data - interp_vals
    error_data_list.append(error)

pos_error = error_data_list[0]
q_error = error_data_list[1]
biasg_error = bias_true_history - bias_g_est_history
# if camera != None:
#     camera.close()

######
# PLOT
######

# if camera != None:
#     camera.close()

# setup plotting
plt.close("all")

plotter_list = []

# create and add data to plotter for position
pos_plotter = MultiPlotter((3, 1), (6, 6), "Position", dict(sharex=True))
pos_plotter.add_data("all", ref_time, ref_position_history.T, labels="Reference")
pos_plotter.add_data("all", sim_time, p_W_history.T, labels="Estimate")
pos_plotter.set_axis_titles(2, "Time", "")
pos_plotter.set_plot_titles("all", ["x", "y", "z"])
plotter_list.append(pos_plotter)

#
# # create and add data to plotter for w
# w_plotter = MultiPlotter((3, 1), (6, 6), "w", dict(sharex=True))
# w_plotter.add_data("all", imu_time_history, w_history.T, labels="Reference")
# w_plotter.add_data("all", sim_time, w_hat_history.T, labels="Estimate")
# w_plotter.set_axis_titles(2, "Time", "")
# w_plotter.set_plot_titles("all", ["x", "y", "z"])
#
# # create and add data to plotter for gyro bias
# bias_g_est_plotter = MultiPlotter((3, 1), (6, 6), "Gyro bias", dict(sharex=True))
# bias_g_est_plotter.add_data("all", sim_time, bias_g_est_history.T, labels="Estimate")
# bias_g_est_plotter.set_axis_titles(2, "Time", "")
# bias_g_est_plotter.set_plot_titles("all", ["x", "y", "z"])
#
# # create and add data to plotter for attitude
att_plotter = MultiPlotter((2, 2), (6, 6), "Attitude", dict(sharex=True))
att_plotter.add_data("all", ref_time, ref_attitude_history.T, labels="Reference")
att_plotter.add_data("all", sim_time, q_W_to_I_history.T, labels="Estimate")
att_plotter.set_axis_titles((2, 3), "Time", "")
att_plotter.set_plot_titles("all", ["x", "y", "z", "w"])
plotter_list.append(att_plotter)

# # create and add data to plotter for covariance and error
error_plotter = MultiPlotter((3, 1), (6, 6), "Error", dict(sharex=True))
error_plotter.add_data("all", ref_time, pos_error.T, labels="Error")
style = dict(color="red")
error_plotter.add_data("all", sim_time, sigmas_position_history.T, labels=r"$3\sigma$", line_styles=style)
error_plotter.add_data("all", sim_time, -sigmas_position_history.T, labels=r"$3\sigma$", line_styles=style)
error_plotter.set_axis_titles(2, "Time", "")
error_plotter.set_plot_titles("all", ["x", "y", "z"])
plotter_list.append(error_plotter)

# # create and add data to plotter for covariance and error for gyro bias.
# error_plotter = MultiPlotter((3, 1), (6, 6), "Error gyro bias", dict(sharex=True))
# error_plotter.add_data("all", sim_time, biasg_error.T, labels="Error")
# style = dict(color="red")
# error_plotter.add_data("all", sim_time, sigmas_biasg_history.T, labels=r"$3\sigma$", line_styles=style)
# error_plotter.add_data("all", sim_time, -sigmas_biasg_history.T, labels=r"$3\sigma$", line_styles=style)
# error_plotter.set_axis_titles(2, "Time", "")
# error_plotter.set_plot_titles("all", ["x", "y", "z"])
# #
# print q_W_to_I_history

# plotter_list.append(w_plotter)

# save_dir = "/home/humberto/final_project/fig/"
#
# save_args = {}
# # for Latex, use
# save_args["format"] = "pdf"
#
# # pos_plotter.save(save_directory=save_dir,save_args=save_args)
# # att_plotter.save(save_directory=save_dir,save_args=save_args)
# # error_plotter.save(save_directory=save_dir,save_args=save_args)
#
plt.ion()
# apply some settings and display all plotters
for plotter in plotter_list:
    plotter.add_grid("all")
    plotter.add_figure_legend(legend_space_inches=0.4, legend_args=dict(ncol=2))
    plotter.display(hold=True)
    # plt.show(block=True)
