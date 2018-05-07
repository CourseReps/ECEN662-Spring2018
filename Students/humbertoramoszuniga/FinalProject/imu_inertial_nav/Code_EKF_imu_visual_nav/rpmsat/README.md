# ROS Python Multi-Sensor Analysis Tools (RPM-SAT)

Robot Operating System (ROS) is a widely-used framework for robotics and multi-sensor data fusion work.
ROS contains many tools for getting, using, and saving data from distributed sensors and systems.
ROS is designed for and excels in hardware applications.
However, by default ROS is not as well suited for the development and/or implementation of new algorithms that require sensor data.
Users accustomed to a MATLAB-style development environment may find that the ROS infrastructure adds significant complexity to their development process or may be completely incompatible.

This library attempts to bridge this gap to some extent.
To understand the purpose of this library, consider this example.
A user wants to develop and implement a mulit-sensor Python-based navigation algorithm.
The navigation system requires input from an inertial measurement unit (IMU), monocular camera, and rangefinder.
The hardware implementation of such a system naturally fits within the ROS framework. However, long before hardware implementation, the engineer desires to develop and validate the system using simulations, where every variable can be precisely controlled.
In addition, the developer wants to run the system using hardware data offline to permit detailed analysis.
This library presents a unified framework for this development process so that an engineer can easily implement and test their system while seamlessly switching between simulated and hardware data.

RPM-SAT is a relatively small and generic library.
The example folder contains an example about how to use this library in an application.
To run the example scripts you will need to have [quatematics](https://github.com/dwhit15/quatematics) and [multiplot2d](https://github.com/dwhit15/multiplot2d) installed.

<!-- This framework was initially conceived and implemented for a master's thesis project. Though RPM-SAT could be useful to many people for many projects, some parts of the current package contains code that is somewhat specific to the project it was initially developed for.
A completely generic and documented version of RPM-SAT is far out of the scope of the original developer.
Instead, this package is likely to be primarily useful in bits and pieces.
User's are encouraged to take the ideas and useful portions of code from this package and apply them to their own application.  -->

## Installation

Clone this repository to a location of your choice.
Add rpmsat to the PYTHONPATH by running setup.sh.

For example:

```bash
git clone https://github.com/dwhit15/rpmsat
cd rpmsat
sh setup.sh
```
