# Robotic_project_final
Mandatory final project repository.

## Repository Structure

| Folder | Purpose |
|--------|---------|
| `final_project_of_the_course/` | Gazebo simulation workspace |
| `dts/` | Duckietown Stack deployment |

## Particle Filter Visualizer (`pf_visualizer.py`)

This node visualizes the robot's real path, estimated path, landmarks, and particles on a 2D grid.

### Subscribed Topics
* **`/odom`** (`nav_msgs/Odometry`): Robot's raw wheel odometry trajectory (drawn as a **red line**).
* **`/pf/estimated_pose`** (`geometry_msgs/PoseStamped`): Robot's estimated pose from the particle filter (drawn as a **green line**).

### Inputs / Connected Data
* **`publish_particles(particles_list)`**: Python method to feed particle states (`x`, `y`, `theta`, `weight`). Drawn as arrows (blue-to-red based on particle weight).
* **AR Tags**: Hardcoded landmarks in the room (drawn as **blue squares**).

### Published Topics
* **`/vis/pf_image`** (`sensor_msgs/Image`): The final output visualization image, published at 10Hz.
