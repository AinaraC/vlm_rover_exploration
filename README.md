# VLM Rover Exploration 🤖 🚀

This repository contains the code and resources required to deploy and evaluate the autonomous exploration of a lunar rover using Vision-Language Models (VLMs) in ROS 2 and a Finite State Machine in YASMIN.

## System Prerequisites

To run this project, the system must meet the following requirements:

- **Operating System**: Ubuntu 24.04.
- **Robotics Framework**: ROS 2 Jazzy.
- **Hardware Acceleration**: CUDA-compatible NVIDIA GPU (required to compile and run VLM models with GPU acceleration via `GGML_CUDA`).

## Installation

Follow these steps to set up your workspace and build the project:

1. **Create the ROS 2 workspace**:
   ```bash
   mkdir -p ~/ros2_ws
   cd ~/ros2_ws
   ```

2. **Clone this repository** into the workspace:
   ```bash
   git clone git@github.com:AinaraCampoDuran/vlm_rover_exploration.git
   ```

3. **Install repository dependencies**:
   The project uses external repositories specified in the `dependencies.repos` file. These dependencies include pinned versions (in this case, the `main` branch) to ensure compatibility:
   - `yasmin` (version: `main`)

   To import them into the workspace, run:
   ```bash
   vcs import src < src/vlm_rover_exploration/dependencies.repos
   ```

4. **Install system and ROS dependencies**:
   ```bash
   sudo apt update
   sudo apt upgrade
   sudo apt install ros-jazzy-gz-ros2-control ros-jazzy-ros2-control ros-jazzy-ros2-controllers ros-dev-tools python3-vcstool
   
   rosdep update
   rosdep install --from-paths src --ignore-src -y
   ```

5. **Build the project**:
   It is essential to build the workspace with the `-DGGML_CUDA=ON` flag enabled to activate hardware acceleration for the vision-language models:
   ```bash
   colcon build --cmake-args -DGGML_CUDA=ON
   ```

6. **Source the environment**:
   ```bash
   source install/setup.bash
   ```

## Usage and Execution

Once installed and built, you can run the system using the provided scripts. Make sure to always be in the workspace root directory and to have sourced the environment (`source install/setup.bash`).

### Launch the Simulator
To launch only the Gazebo simulation environment and test the rover, use the simulation launch script:

```bash
ros2 launch test_sim.launch.py world_script:=low_moon.launch.py
```

### Run Experiments and VLM Evaluations
To automatically launch evaluations and repetitions across the VLM models (MiniCPM, Qwen3-VL, InternVL3), use the experiment script:

```bash
./run_experiments.sh
```

This script iterates through the different models, runs them under a predefined time limit, and cleans up residual ROS 2 nodes and Gazebo processes between each execution using the `cleanup_ros.sh` script.

### Run a Model Manually
If you want to launch the exploration pipeline with a specific model manually instead of using the automated experiment script, you can run the launch file directly:

```bash
ros2 launch vlm_rover_exploration_bringup vlm_rover_exploration.launch.py vlm_model:=MiniCPM.yaml
```

### Results Analysis and Visualization
After completing experiment runs, you can process the collected data and generate comparative plots using the scripts provided in the `vlm_rover_exploration` package.

Make sure to run both scripts from the **workspace root** (`~/ros2_ws`):

1. **Process metrics (Benchmark)**:
   This script searches for all raw results files (`raw_metrics_*.json`), calculates detailed statistics, and generates summaries for each model.
   ```bash
   python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/benchmark.py
   ```

2. **Generate plots and visualizations**:
   Once metrics are processed, use this script to generate bar charts, boxplots, and strategy frequencies for the models. The resulting figures will be saved in a new directory called `benchmark_visualizations/`.
   ```bash
   python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/visualize_results.py
   ```