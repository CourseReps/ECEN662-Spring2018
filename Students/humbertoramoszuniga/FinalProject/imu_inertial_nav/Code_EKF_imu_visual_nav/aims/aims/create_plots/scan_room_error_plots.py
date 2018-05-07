import matplotlib.pyplot as pyplot

import os
from aims import *
import matplotlib.pylab as plt
from multiplot2d import MultiPlotter

from analysis import *


data1= ("scan_room_final_2",  [9.6,72] )
data2 = ("scan_room_up_down_final_2",  [9.6,72] )

plot_size = (6,3.5)
subplot_args=dict(sharex=True)
error_list = []
############################
for data in (data2, data1):
    use_database = data[0]
    database_path = "/Dropbox/aims_db/" + use_database + ".odb"
    tlims=data[1]
    t0=tlims[0]

    
    # pull data from odb
    plot_db = database_from_file(os.environ['HOME']+database_path)
    est_history = plot_db.estimate_history
    est_history.time -= tlims[0]
    three_sigma = plot_db.three_sigma
    three_sigma.time -= tlims[0]
    
    ref_pose = plot_db.idb.reference.kinematics

    
    ref_pose.time -= tlims[0]
    tlims[1] -= tlims[0]
    tlims[0] = 0 
    
    est_history.euler_321 = quat_2_euler_321(est_history.attitude)*180./np.pi
    ref_pose.euler_321 = quat_2_euler_321(ref_pose.attitude)*180./np.pi

    # interpolate estimate history so that x-axis for estimates and reference are same
    comps = ["position","euler_321"]
    fig_names = ["position", "attitude"]
    est_history_interp = interpolate_estimates(est_history, ref_pose,comps)
    # get additive errors
    add_errors = additive_error(est_history_interp, ref_pose,comps)

    
    error_list.append(add_errors)

error_index_list = [(3,6),(0,3)]
state_titles = [("x [m]", "y [m]", "z [m]")]
state_titles.append( ("Yaw [deg]", "Pitch [deg]", "Roll [deg]") )
t_label_plots = (4,5)
subplot_args=dict(sharex=True)
error_plotter = MultiPlotter((3,2),plot_size,"scan_room_error",subplot_args=subplot_args)

labels = ["Errors: With Guidance", "Errors: No Guidance" ]
for add_errors, label in zip(error_list, labels):

    error_plotter.add_data((0,2,4),add_errors.time,add_errors.position.T,labels=label)
    error_plotter.add_data((1,3,5),add_errors.time,add_errors.euler_321.T,labels=label)
    error_plotter.set_axis_titles((0,2,4),y_titles=state_titles[0])
    error_plotter.set_axis_titles((1,3,5),y_titles=state_titles[1])

error_plotter.set_axis_titles(t_label_plots,"Time [s]", "")
subplot_list = error_plotter.get_plots("all")
for plot in subplot_list:
    plot.locator_params( nbins=5)

error_plotter.add_grid("all")
error_plotter.set_limits("all",x_limits=tlims)
error_plotter.add_figure_legend(legend_space_inches=0.15)


error_plotter.display(False)
save_dir = os.environ['HOME']+"/Dropbox/whitten_thesis/tex/fig/" 
save_args = {}
save_args["format"] = "pdf"
save_args["bbox_inches"] = "tight"
save_args["pad_inches"] = 0.05
error_plotter.save(save_directory=save_dir,save_args=save_args)
plt.ion()
plt.show()

