
#data = ("space_lab_final", "space_lab", [17,94],False, 20, True, 0)
data = ("space_lab", "space_lab", [17,94],False, 20, True, 0)

#data = ("orange_2", "orange_2", [16.21,50],False, 20, True, 0)
#data = ("sim_space_lab", "sim_space_lab", [7.4,84.4], True, 20, True, 0)
#data = ("scan_room_final_2", "scan_room", [9.6,72], True, 20, False, 1)
#data = ("scan_room_up_down_final_2", "scan_room_up_down", [9.6,72], True, 30, False, 1)
import matplotlib.pyplot as pyplot

import os
from aims import *
import matplotlib.pylab as plt
from multiplot2d import MultiPlotter

from analysis import *

############################

use_database = data[0]
save_sub_dir = data[1]
simulated = data[3]
database_path = "/Dropbox/aims_db/" + use_database + ".odb"
tlims=data[2]
t0=tlims[0]
aux_plot_y_lim = data[4]
show_error = data[5]
aux_plot_type = data[6]

#plot_size = (7,4.5)
plot_size = (6,3.5)

subplot_args=dict(sharex=True)
p_error_ylims = [0.06,0.05,0.006]
three_sigma_style = dict(color="#999999",ls="--")
error_style = dict(color="#E24A33")
truth_style = dict(color="k")

est_style = dict(color="#E24A33") #red
#est_style = dict(color="#348ABD") # blue



# pull data from odb
plot_db = database_from_file(os.environ['HOME']+database_path)
est_history = plot_db.estimate_history
est_history.time -= tlims[0]
three_sigma = plot_db.three_sigma
three_sigma.time -= tlims[0]

if simulated:
    ref_pose = plot_db.idb.reference.kinematics
    truth_name = "Ground Truth"
else:
    ref_pose = plot_db.idb.reference.pose
    truth_name = "Motion Capture"

ref_pose.time -= tlims[0]
tlims[1] -= tlims[0]
tlims[0] = 0 

est_history.euler_321 = quat_2_euler_321(est_history.attitude)*180./np.pi
ref_pose.euler_321 = quat_2_euler_321(ref_pose.attitude)*180./np.pi
quat_error_est = np.vstack((0.5*three_sigma.imu_state[0:3,:], np.ones((1,three_sigma.imu_state.shape[1])) ))
three_sigma.imu_state[0:3,:] = quat_2_euler_321(quat_error_est)*180./np.pi


# interpolate estimate history so that x-axis for estimates and reference are same
comps = ["position","euler_321"]
fig_names = ["position", "attitude"]
est_history_interp = interpolate_estimates(est_history, ref_pose,comps)
# get additive errors
add_errors = additive_error(est_history_interp, ref_pose,comps)
#three_sigma = interpolate_three_sigma(three_sigma,np.arange(three_sigma.time[0],tlims[1],0.1))

ref_pose.distance_traveled = np.zeros(ref_pose.position.length())
for k in xrange(ref_pose.position.length()):
    ref_pose.distance_traveled[k] = np.sum(np.linalg.norm(np.diff(ref_pose.position[:,:k],1,1),axis=0))

pos_error_norm = np.linalg.norm(add_errors.position,axis=0)

# setup plotting
plt.close("all")
plt.ioff()
plt.style.use("dwplot")
all_plotters = []

error_index_list = [(3,6),(0,3)]
state_titles = [("x [m]", "y [m]", "z [m]")]
state_titles.append( ("Yaw [deg]", "Pitch [deg]", "Roll [deg]") )

t_label_plots = (4,5)

if show_error:
    
    for comp, error_indices, fig_name, state_title in zip(comps,error_index_list,fig_names,state_titles):
        state_range = (0,2,4)
        state_plotter = MultiPlotter((3,2),plot_size,fig_name,subplot_args=subplot_args)
        est_data = getattr(est_history,comp).T
        ref_data = getattr(ref_pose,comp).T
        
        state_plotter.add_data(state_range,ref_pose.time,ref_data,labels=truth_name,line_styles=truth_style)
        state_plotter.add_data(state_range,est_history.time,est_data,labels="Estimate",line_styles=est_style)
        state_plotter.set_axis_titles(state_range,y_titles=state_title)
        
        err_range = (1,3,5)
        data = getattr(add_errors,comp).T
        state_plotter.add_data(err_range,add_errors.time,data,labels="Error",line_styles=error_style)
        three_sigma_data = three_sigma.imu_state[error_indices[0]:error_indices[1],:].T
        state_plotter.add_data(err_range,three_sigma.time,three_sigma_data,line_styles=three_sigma_style,labels="$3\hat\sigma$")
        state_plotter.add_data(err_range,three_sigma.time,-three_sigma_data,line_styles=three_sigma_style,labels="$3\hat\sigma$")   
        
        state_plotter.set_axis_titles(t_label_plots,"Time [s]", "")
        state_plotter.add_figure_legend(legend_space_inches=0.15)
        subplot_list = state_plotter.get_plots("all")
        for plot in subplot_list:
            plot.locator_params( nbins=5)
        all_plotters.append(state_plotter)  
    
else:
    state_range = "all"
    pos_range = (0,2,4)
    att_range = (1,3,5)
    indices_list = [pos_range,att_range]
    state_plotter = MultiPlotter((3,2),plot_size,"states",subplot_args=subplot_args)

    
    for comp, indices, state_title in zip(comps,indices_list,state_titles): 
        est_data = getattr(est_history,comp).T
        ref_data = getattr(ref_pose,comp).T
            
        state_plotter.add_data(indices,ref_pose.time,ref_data,labels=truth_name,line_styles=truth_style)
        state_plotter.add_data(indices,est_history.time,est_data,labels="Estimate",line_styles=est_style)
        state_plotter.set_axis_titles(indices,y_titles=state_title)        
        
        state_plotter.set_axis_titles(t_label_plots,"Time [s]", "")
        state_plotter.add_figure_legend(legend_space_inches=0.15)
    subplot_list = state_plotter.get_plots("all")
    for plot in subplot_list:
        plot.locator_params( nbins=5)
    all_plotters.append(state_plotter)  
 

if aux_plot_type == 0:
    aux_plotter = MultiPlotter((2,1),(4,3.),"features_distance",subplot_args=subplot_args)
    aux_plotter.add_data(0,est_history.time, plot_db.feature_count.visible_features,labels="Visible Features")
    aux_plotter.add_data(0,est_history.time, plot_db.feature_count.features_in_map,labels="Features in Map")
    aux_plotter.set_limits(0,y_limits=(0,110))
    aux_plotter.add_legend(0,dict(loc="upper center",ncol=2))    
    aux_plotter.add_data(1,ref_pose.time,ref_pose.distance_traveled, "Linear Distance Traveled")
    aux_plotter.add_data(1,ref_pose.time,pos_error_norm,"Position Error")
    aux_plotter.set_limits(1,y_limits=(-1,aux_plot_y_lim))
    aux_plotter.add_legend(1,dict(loc="upper left"))
    
if aux_plot_type == 1:
    aux_plotter = MultiPlotter(1,(5,2.5),"criteria",subplot_args=subplot_args)
    aux_plotter.add_data(0,est_history.time, plot_db.feature_count.features_in_map,labels="Features in Map")

    corner_start_times = np.array([25, 35.5, 48.0, 58.5])-t0
    corner_end_times = corner_start_times + 7

    guidance_alt = est_history.position.z.copy()
    for i, t_i in enumerate(est_history.time):
        use = False
        for k in xrange(4):
            t0 = corner_start_times[k]
            tf = corner_end_times[k]
            if t_i >= t0 and t_i <= tf:
                use = True
        if not use:
            guidance_alt[i] = np.nan        
    
    altitude_plot = aux_plotter.get_plots(0)[0].twinx()
    
    altitude_plot.plot(est_history.time, est_history.position.z,ls=":",color="k",label="Altitude: Nominal")
    
    altitude_plot.plot(est_history.time, guidance_alt,color="k",label="Altitude: Guided")
    aux_plotter.set_limits(0,y_limits=(0,170))
    altitude_plot.set_ylim(0,3.5)
    aux_plotter.add_data(0,est_history.time,np.ones(est_history.time.shape[0])*40,r"Potential Threshold $\Gamma$",line_styles=dict(ls="-"))

#    aux_plotter.set_limits(0,y_limits=(0,100))
#    altitude_plot.set_ylim(0,2.)
    
    altitude_plot.legend(loc="upper right")
    aux_plotter.add_legend(0,dict(loc="upper left"))

aux_plotter.set_axis_titles(-1,"Time [s]")
all_plotters.append(aux_plotter)

for plotter in all_plotters:
    plotter.add_grid("all")
    
    plotter.set_limits("all",x_limits=tlims)
    plotter.display(False)
    save_dir = os.environ['HOME']+"/Dropbox/whitten_thesis/tex/fig/" + save_sub_dir 
    save_args = {}
    save_args["format"] = "pdf"
    save_args["bbox_inches"] = "tight"
    save_args["pad_inches"] = 0.05
#    plotter.save(save_directory=save_dir,save_args=save_args)

plt.ion()
plt.show()

