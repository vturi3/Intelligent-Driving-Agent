# Autonomous Vehicle Driving Project with CARLA Simulator

<div align="center">
  <img src="carla.jpg" alt="CARLA Simulator">
</div>
## Description 
This repository contains the code and related documentation for a group project focused on the development of an autonomous driving system using CARLA simulator. The use of a simulator like this has the benefit of providing powerful and realistic virtual environment that allows the development of an autonomous driving system in a safe and controlled manner. The simulator offered a comprehensive set of simplified models and representations of various elements, including road infrastructure, traffic dynamics, and sensor simulations. This allowed the development of the system to focus on specific aspects such as control algorithms and decision-making strategies, without being overwhelmed by the complexity and variability of real-world scenarios, having the guarantee that at all times it is possible to know from the sorrounding world all the information useful for decision-making. This last aspect is not possible in real-world scenarios where, to give a trivial example, it is not possible to easily trace in real-time the position, speed and acceleration of other vehicles present in the world or to easily have the precise location and status of every object. It is important to specify this to emphasize that the obtained results, which are extremely promising, would be the same only if such information were also obtainable in the real world, making our system safely applicable.
The goal of the proposed implementation was to achieve a system that would take into account the maximum number of situations and that would be as general as possible, using the extra routes, not used during the code implementation, as "test set". 
Finally, the implemented algorithm and the results obtained should also be affected by some anomalous situations that occur within the simulator, but that are unrealistic, such as: excessive speed of some spawn actors when dropped into the world, obstructionism of some actors to the completion of overtaking, accumulation of infractions if the average speed of the vehicle is lower than the average speed of the other actors in the scene, etc. 
The results obtained on the optional routes are clearly improvable in future developments.

## Repository Structure
- `Project_code/`:
    - `AVD_PID_Implementation/`: contains the developed software using PID controller as lateral controller;
    - `AVD_Stanley_Implementation/`: contains the developed software using Stanley controller as lateral controller.
    - `Highlights/`: contains a series of short videos depicting the operation of the system in particular situations of the various routes.
    - `Result/`: contains the json file of the results obtained on extra routes.
    - `Test/`: contains the json file of the results obtained on the mandatory routes, both using PID and Stanley lateral controllers, running the developed system on various machines, in order to make the results more reliable, as stated in the report.
    - `Project_report.pdf`: a report including a comprehensive analysis of the implementation logic of developed features and obtained results.

## Implemented Features
Two different implementations are proposed that refer to the use of the Stanley controller and the PID controller as lateral controllers. For both versions, the following features has been implemented:
- **Braking management**, starting from a certain distance deemed as dangerous, avoiding abrupt actions if it is possible;
- **Vertical sign management**, like traffic lights or stop signs;
- **Detection of potential collisions**, considering the occupancy of our vehicle and of the other vehicles in the route, managing different actors like pedestrians, bikers, other motorized vehicles or static objects;
- **Overtaking**, evaluating the feasibility of this action depending on the required space, avoiding collision with other actors in the scene, taking into account their direction and motion, and trying not to go off-road and to occupy the adjacent lane for as little space and time as possible;
- **Junctions management**, avoiding collision with other actors in the scene taking into account their actual and future direction and motion;
- **Car following**, adjusting the speed of the autonomous car based on the behaviour of the leading vehicle (a vehicle with the same direction in front of the car), ensuring safe and smooth driving.

For more details about the project, including a comprehensive analysis of the implementation logic of these features and results, refer to the final report available in the [Project Report](Project_Report.pdf) file.

## Demo
[Here](https://www.youtube.com/watch?v=Uzu7N8bdBuM) there is a demo of the developed system.

## Feedback
For any feedback, questions or inquiries, please contact the project maintainers listed in the Contributors section.
