Implementation of an Extended Kalman filter for Vision-Aided Inertial Navigation
based on known world landmarks

///////////////////GENERAL DESCRIPTION OF THE SOFTWARE FRAMEWORK

The open-source robotics framework ROS is used to collect and store data from the IMU, camera,
Aruco process, and Vicon motion capture system. ROS stores the data in a binary format known as a rosbag.
The LASR lab library RPM-SAT was used to convert the data from the rosbag into a format that could be sequentially post-processed in Python,
while maintaining the original data timing.

The open-source library Quatematics provided a Quaternion class that was used to implement equations
relating to quaternions. The open source libraries matplotlib and Multiplot2D were used to plot the results. OpenCV was used to create the measurement visualization.
Quatematics, Multiplot2D, and RPM-SAT are included in the package along with this report. All other libraries mentioned before, must be installed using a package manager.
Many additional dependences are required to install all of these libraries.
//////////////////////////////////////////////////////////////////////////

BEFORE BEGIN

To be able to run the code for this project Linux Ubuntu 14.04 is needed. Otherwise, it may be possible that additional libraries are needed. Although different
operative systems were tested.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


1 INSTALL THE FOLLOWING LIBRARIES IN FULL version

  *ROS Desktop
  *Python and numpy (it is possible that some other dependencies are needed in the system in which this is intended to be run, so install what is required).
  *OpenCV 2.4 full
  *Quatematics (included with this package).
  *Multiplot2D (included with this package).
  *RPM-SAT (included with this package).

  For the last three open source libraries (Quatematics, Multiplot2D, and RPM-SAT ), follow the instructions inside. There is a README file that contains the instructions for installing them as well.

  Make sure the bashrc file contains the following lines at the end.

  source /opt/ros/indigo/setup.bash
  export PYTHONPATH=/opt/ros/indigo/lib/python2.7/dist-packages/:/home/humberto/src/quatematics/:/home/humberto/src/imu_inertial_nav/:/home/humberto/src/multiplot2d/:/home/humberto/src/aims:/home/humberto/src/rpmsat/

   *****Substitute the words "humberto" by anything that makes to match the directory in which the libraries are copied.*******
   The same directory is highly recomended, you could create a humberto folder.

  2 GENERATE A WORKINGSPACE FOR THE ROS ENVIRONMENT

    *According to this site http://wiki.ros.org/cn/catkin/Tutorials/create_a_workspace create a /catkin_ws directory. Inside this directory, the folder called imu_intertial_nav must be placed.
    IMPORTANT: This system was designed to be used with ROS Indigo Version.


  3 COMPILE THE PROGRAM IN SOME IDE
      *The code is written in Python. The IDE is not important. However, make sure the IDE runs from a terminal.
      *By default the program will look for the data to be processed at /home/data (this path structure is for linux).
      So place the folder data at /home. This folder contains different trajectories recorded. Notice that the program
      processes the data as it is coming in real time.
      * The generation of new trajectories can be made but just in the prescence of Vicon motion capture sytem.
      *The MAIN PROGRAM is called run_idb.py and it is inside imu_inertial_nav/scripts. By running this program after about 50 seconds (since process like in real time),
      the plots with the results must pop up.

  4 TROUBLESHOOTING
        *The only thing that can go wrong is the insuficiency of dependencies. This highly depends on your systems and how updated is.
        Commonly dependency errors appear in the console in which the program or IDE shows information. Also it can be shown in a terminal
        if the program is running from there, at any rate, just install the dependencies that are missing.
        This happens if Python or ROS were not completely installed during the instalation process, or if it comes to third library dependencies
        or when the system is not updated.

        *Some times when bashrc is not sourced correclty and although the dependencies and libraries are where they are supposed to, the system does not find them.
        It is worth it to try to source it again as
            source ~/.bashrc

        * The same issue can ocurr for the libraries that are inside the catkin_ws folder, to source them again by cleaning the catkin workspace by
        executing
                   catkin_make clean
        and then
                  catkin_make make
