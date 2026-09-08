# VLM Rover Exploration: Autonomous Planetary Mapping via Vision-Language Models & Hierarchical State Machines 🤖 🚀

An end-to-end research and benchmarking framework for evaluating **Vision-Language Models (VLMs)** as zero-shot high-level planners for **autonomous planetary rover exploration** in **ROS 2 Jazzy** and **Gazebo Sim**. 

This project integrates local multimodal LLM inference (`llama.cpp` / `llama_ros` with CUDA acceleration) into a discrete frontier-exploration pipeline orchestrated by a robust **YASMIN** (Yet Another State MachINe) Finite State Machine. The system empowers a lunar rover to visually analyze its expanding occupancy grid map, select strategic frontier waypoints through structured 8-step Chain-of-Thought (CoT) reasoning, and navigate autonomously using Nav2.

---

### Key Capabilities & Highlights

1. **Multimodal VLM Strategic Decision-Making**: Closed-loop frontier target selection using state-of-the-art vision-language architectures (**MiniCPM-o 4.5**, **Qwen3.5-VL**, **InternVL3.5**). Prompts enforce structured 8-step Chain-of-Thought (CoT) reasoning and strict JSON schema output via grammar-constrained sampling.
2. **Hierarchical Robotics Architecture (ROS 2 & YASMIN)**: Clear decoupling of perception, high-level cognitive deliberation , and path execution managed through an inspectable YASMIN FSM.
3. **Quantitative Exploration & Trajectory Metrics**: Multi-faceted evaluation framework measuring total explored area ($m^2$), coverage ratio ($\text{CR}$ in $m^2/m$), decision efficiency ($m^2/\text{step}$), target jump distance ($\Delta d_{\text{jump}}$), proximity ranking ($R_p$), and odometry drift ($E_{\text{odom}}$).
4. **Hardware Telemetry & Performance Profiling**: Real-time profiling of computational load during inference and navigation, logging per-step CPU usage (%), GPU utilization (%), VRAM allocation (MB), system RAM (MB), inference latency ($s/\text{step}$), and navigation transit time.
5. **Automated Campaign & Visualization Suite**: Ready-to-use batch evaluation orchestration (`run_experiments.sh` with process hygiene and timeouts) and statistical analysis tooling producing publication-quality composite bar charts, boxplot distributions, and strategy frequency graphs.

---

## Table of Contents

- [System Architecture & Workflow](#system-architecture--workflow)
- [Repository Structure](#repository-structure)
- [Prerequisites & System Requirements](#prerequisites--system-requirements)
  - [Hardware Requirements](#hardware-requirements)
  - [Software & Frameworks](#software--frameworks)
- [Installation & Build Guide](#installation--build-guide)
  - [1. Workspace Setup](#1-workspace-setup)
  - [2. Clone Repository & External Repositories](#2-clone-repository--external-repositories)
  - [3. Install System & ROS Dependencies](#3-install-system--ros-dependencies)
  - [4. Build Workspace with CUDA Hardware Acceleration](#4-build-workspace-with-cuda-hardware-acceleration)
  - [5. Environment Configuration](#5-environment-configuration)
- [Step-by-Step Usage Guide](#step-by-step-usage-guide)
  - [1. Standalone Simulation Bringup](#1-standalone-simulation-bringup)
  - [2. Dynamic Obstacle Generation](#2-dynamic-obstacle-generation)
  - [3. Running Autonomous Exploration (Manual Run)](#3-running-autonomous-exploration-manual-run)
  - [4. Automated Multi-Model Benchmark Campaign](#4-automated-multi-model-benchmark-campaign)
  - [5. Live FSM Telemetry (YASMIN Web Viewer)](#5-live-fsm-telemetry-yasmin-web-viewer)
  - [6. Post-Mission Benchmark & Statistical Aggregation](#6-post-mission-benchmark--statistical-aggregation)
  - [7. Visualization & Comparative Analysis](#7-visualization--comparative-analysis)
- [Exploration Pipeline & Decision-Making Framework](#exploration-pipeline--decision-making-framework)
  - [1. 2D Polar-Cartesian Map Representation](#1-2d-polar-cartesian-map-representation)
  - [2. 8-Step Structured Chain-of-Thought (CoT) Prompting](#2-8-step-structured-chain-of-thought-cot-prompting)
  - [3. Grammar-Constrained JSON Sampling](#3-grammar-constrained-json-sampling)
- [Evaluation Metrics & Mathematical Formulation](#evaluation-metrics--mathematical-formulation)
  - [1. Exploration Efficiency Metrics](#1-exploration-efficiency-metrics)
  - [2. Navigation Dynamics & Odometry Drift](#2-navigation-dynamics--odometry-drift)
  - [3. Cognitive & Spatial Decision Metrics](#3-cognitive--spatial-decision-metrics)
  - [4. Hardware Profiling & Latency Breakdown](#4-hardware-profiling--latency-breakdown)
- [Experimental Results & Benchmark Summary](#experimental-results--benchmark-summary)
  - [Multi-Model Comparative Performance (10-Run Averages)](#multi-model-comparative-performance-10-run-averages)
  - [Key Findings & Behavioral Analysis](#key-findings--behavioral-analysis)
- [Supported VLM Models & Configs](#supported-vlm-models--configs)
- [Citations & Tool References](#citations--tool-references)

---

## System Architecture & Workflow

```text
       +-------------------------------------------------------------+
       |               Gazebo Simulation Environment                 |  
       +------------------------------+------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |                  Perception & SLAM (ROS 2)                  |
       +------------------------------+------------------------------+
                                      |
                                      v
+---------------------------------------------------------------------------+
|               YASMIN Finite State Machine (`exploration_sm`)              |
|                                                                           |
|  [GENERATING_MAP_IMAGE]                                                   |
|    - Center map on initial pose, filter frontiers along boundary          |
|    - Render Polar-Cartesian crosshair, scale markers, and numbered IDs    |
|    - Compute exact Euclidean distances from rover to all frontiers        |
|                               |                                           |
|                               v                                           |
|  [GENERATING_NEXT_WP] (llama_ros / llava_node)                           |
|    - Multimodal query with image + distances + strategy memory            |
|    - 8-Step Structured CoT: Global Pattern -> Candidates -> Finalist     |
|    - Enforce valid JSON grammar: {"reasoning", "strategy", "target"}      |
|    - Sample hardware telemetry (GPU %, CPU %, RAM, VRAM) via psutil       |
|                               |                                           |
|                               v                                           |
|  [PROCESSING_RESPONSE]                                                    |
|    - Parse JSON response, validate frontier ID existence                  |
|    - Compute target heading (yaw) & proximity ranking                     |
|    - Export debug maps (`map_N.png`, `full_route_history.png`)            |
|                               |                                           |
|                               v                                           |
|  [DRIVING_TO_WAYPOINT] (Nav2 Action)                                      |
|    - Dispatch target pose to Nav2 (`/navigate_to_pose`)                   |
|    - Track transit duration, success status, and navigation retries       |
|                               |                                           |
|                               v (Loop or Mission Complete)                |
|  [SHOW_METRICS]                                                           |
|    - Aggregate telemetry & trajectory -> Save `raw_metrics_<model>.json`  |
+---------------------------------------------------------------------------+
                                      |
                                      v
       +-------------------------------------------------------------+
       |             Benchmarking & Visualization Suite              |
       |  - `benchmark.py`: Aggregates runs -> `average_benchmark.json` |
       |  - `visualize_results.py`: Multi-panel bar charts, boxplots |
       |  - Strategy frequency distribution & decision stability     |
       +-------------------------------------------------------------+
```

---

## Repository Structure

```text
vlm_rover_exploration/
├── average_benchmark_InternVL3.json      # Aggregated 10-run benchmark statistics (InternVL3.5)
├── average_benchmark_MiniCPM.json        # Aggregated 10-run benchmark statistics (MiniCPM-o 4.5)
├── average_benchmark_Qwen3-VL.json       # Aggregated 10-run benchmark statistics (Qwen3.5-VL)
│
├── benchmark_visualizations/             # Generated publication-quality comparative plots
│   ├── grouped_averages_exploration.png     # Exploration metrics bar charts (Mean ± Std)
│   ├── grouped_averages_performance.png     # Hardware consumption bar charts (CPU, GPU, RAM, VRAM)
│   ├── grouped_averages_time.png            # Latency and duration bar charts (Inference vs Nav)
│   ├── grouped_distributions_exploration.png# Boxplot distributions for exploration metrics
│   ├── grouped_distributions_performance.png# Boxplot distributions for hardware utilization
│   ├── grouped_distributions_time.png       # Boxplot distributions for step durations
│   └── strategy_frequencies.png             # Stacked bar chart of chosen global strategies (%)
│
├── run_experiments.sh                    # Automated campaign runner (iterates models, applies timeout)
├── cleanup_ros.sh                        # Process hygiene script (terminates stale ROS 2 / GZ nodes)
│
├── src/
│   ├── vlm_rover_exploration/            # Core exploration package
│   │   ├── dependencies.repos            # External VCS repository specifications (YASMIN, etc.)
│   │   │
│   │   ├── vlm_rover_exploration/        # Python library and state implementations
│   │   │   ├── scripts/
│   │   │   │   ├── benchmark.py          # Scans raw run JSONs, computes stats & saves averages
│   │   │   │   ├── visualize_results.py  # Plots composite figures from benchmark JSONs
│   │   │   │   └── spawn_random_obstacles.py # Spawns procedural Gazebo enclosures/obstacles
│   │   │   │
│   │   │   └── vlm_rover_exploration/
│   │   │       ├── exploration_sm.py     # Main FSM entry point, blackboard, and MetricTracker
│   │   │       └── states/
│   │   │           ├── generate_map_image_state.py # Occupancy grid crop, crosshairs, frontier IDs
│   │   │           ├── llama_state.py              # Multimodal prompt, CoT, grammar, hardware telemetry
│   │   │           ├── process_response_state.py   # JSON validation, yaw computation, debug renders
│   │   │           ├── drive_state.py              # Nav2 NavigateToPose execution & timing
│   │   │           └── show_metrics_state.py       # Metrics serialization to disk
│   │   │
│   │   └── vlm_rover_exploration_bringup/ # Bringup launch scripts & model configs
│   │       ├── launch/
│   │       │   └── vlm_rover_exploration.launch.py # Master launch file (Gazebo + Nav2 + LLaMA + FSM)
│   │       └── models/                   # GGUF model parameter YAMLs
│   │           ├── MiniCPM.yaml          # OpenBMB MiniCPM-o 4.5 Q4_K_M config
│   │           ├── Qwen3-VL.yaml         # Qwen3.5-VL 4B Q4_K_M config
│   │           ├── InternVL3.yaml        # InternVL3.5 4B Q4_K_M config
│   │           ├── SpaceThinker.yaml     # SpaceQwen3-VL 2B Thinking config
│   │           ├── llava-mistral.yaml    # LLaVA Mistral 7B baseline config
│   │           └── gemma4.yaml           # Gemma multimodal baseline config
│   │
│   ├── llama_ros/                        # ROS 2 integration node for llama.cpp (submodule/imported)
│   └── ros2_rover/                       # Rover simulation, Nav2 stack, URDF & Gazebo worlds
│
└── README.md                             # Project documentation
```

---

## Prerequisites & System Requirements

### Used Hardware Setup
- **NVIDIA GPU**: RTX 5060
- **VRAM**: 8 GB
- **CPU**:  Intel Core i9-14900HX.
- **RAM**: 32 GB

### Software & Frameworks
- **Operating System**: Ubuntu 24.04 LTS (Noble Numbat).
- **Robotics Middleware**: ROS 2 Jazzy Jalisco.
- **Simulator**: Gazebo Sim (Harmonic / gz-sim).
- **CUDA Toolkit**: CUDA 12.x with cuDNN and compatible NVIDIA display drivers.

---

## Installation & Build Guide

Follow these steps to configure your ROS 2 workspace, download dependencies, and compile with full GPU hardware acceleration:

### 1. Workspace Setup
Create a dedicated ROS 2 workspace directory:
```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
```

### 2. Clone Repository & External Repositories
Clone `vlm_rover_exploration` into your `src/` directory:
```bash
cd ~/ros2_ws/src
git clone https://github.com/AinaraCampoDuran/vlm_rover_exploration.git
```

Import external pinned dependencies (such as **YASMIN**) defined in `dependencies.repos`:
```bash
cd ~/ros2_ws
vcs import src < src/vlm_rover_exploration/dependencies.repos
```

### 3. Install System & ROS Dependencies
Install essential ROS 2 build tools, Gazebo ros2_control packages, and resolve package dependencies via `rosdep`:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-nav2-bringup \
  ros-jazzy-navigation2 \
  ros-jazzy-rtabmap-ros \
  ros-dev-tools \
  python3-vcstool \
  python3-pip \
  python3-psutil \
  python3-seaborn \
  python3-pandas

rosdep update
rosdep install --from-paths src --ignore-src -y -r
```

### 4. Build Workspace with CUDA Hardware Acceleration
> [!IMPORTANT]
> It is essential to pass `-DGGML_CUDA=ON` to `colcon build`. This instructs `llama_ros` and `llama_cpp_vendor` to compile the llama.cpp backend with native CUDA kernel acceleration, offloading VLM transformer and vision-projector layers directly to the NVIDIA GPU.

```bash
cd ~/ros2_ws
colcon build --symlink-install --cmake-args -DGGML_CUDA=ON
```

### 5. Environment Configuration
Source the workspace overlay:
```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
```
*(Tip: Add `source ~/ros2_ws/install/setup.bash` to your `~/.bashrc` for convenience).*

---

## Step-by-Step Usage Guide

### 1. Standalone Simulation Bringup
To launch the Gazebo simulation environment alone (inspecting the rover chassis, terrain elevation, cameras, and lidar sensors):

```bash
# Launch Moon environment
ros2 launch rover_gazebo moon.launch.py
```

### 2. Dynamic Obstacle Generation - NOT SUPPORTED
To test the rover's obstacle-avoidance and local detour reasoning, spawn procedural square enclosures or barrier walls dynamically inside the exploration radius ($8\,\text{m}$ from origin):

```bash
python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/spawn_random_obstacles.py
```

---

### 3. Running Autonomous Exploration (Manual Run)
To execute the complete autonomous exploration pipeline with a specific VLM model:

```bash
ros2 launch vlm_rover_exploration_bringup vlm_rover_exploration.launch.py vlm_model:=MiniCPM.yaml
```

Supported model parameters:
- `vlm_model:=MiniCPM.yaml` (Recommended: MiniCPM-o 4.5 Q4_K_M)
- `vlm_model:=Qwen3-VL.yaml` (Qwen3.5-VL 4B Q4_K_M)
- `vlm_model:=InternVL3.yaml` (InternVL3.5 4B Q4_K_M)

This launch sequence automatically starts:
1. `llama_bringup` hosting the VLM on the ROS 2 Action Server `/llama/generate_chat_completions`.
2. Gazebo simulation with rover dynamics, sensor streams, and TF transforms.
3. Nav2 navigation stack configured with Regulated Pure Pursuit (RPP) and SmacHybrid planner.
4. YASMIN exploration state machine node (`exploration_sm`).
5. YASMIN web viewer node (port `5000`).

---

### 4. Automated Multi-Model Benchmark Campaign
To conduct rigorous comparative evaluations across multiple models with automated repetitions and timeout safety:

```bash
cd ~/ros2_ws
./run_experiments.sh
```

**Campaign Details**:
- Executes **10 repetitions per model** (`MiniCPM.yaml`, `Qwen3_VL.yaml`, `InternVL3.yaml`).
- Enforces a hard **20-minute timeout** per mission execution.
- Invokes `./cleanup_ros.sh` between consecutive runs to eliminate hanging background nodes, Gazebo instances (`gz sim`, `ruby`), Nav2 lifecycle nodes, and memory leaks.
- Saves timestamped output directories for every run containing raw telemetry, maps, and responses.

---

### 5. Live FSM Telemetry (YASMIN Web Viewer)
During execution, monitor state transitions and blackboard data in real time:
1. Open your web browser.
2. Navigate to: **`http://localhost:5000`**
3. View the active state (`GENERATING_MAP_IMAGE` $\to$ `GENERATING_NEXT_WP` $\to$ `PROCESSING_RESPONSE` $\to$ `DRIVING_TO_WAYPOINT`), execution times, and blackboard state variables.

---

### 6. Post-Mission Benchmark & Statistical Aggregation
After completing experimental trials, process raw run logs to compute statistical metrics:

```bash
cd ~/ros2_ws
python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/benchmark.py
```

**Outputs Produced**:
- Generates `benchmark_results_<model>.json` inside every individual run directory.
- Computes aggregated averages, standard deviations, min, and max values across all 10 runs per model.
- Exports master summary files:
  - `average_benchmark_MiniCPM.json`
  - `average_benchmark_Qwen3-VL.json`
  - `average_benchmark_InternVL3.json`

---

### 7. Visualization & Comparative Analysis
Generate comparative publication charts directly from the aggregated benchmark results:

```bash
cd ~/ros2_ws
python3 src/vlm_rover_exploration/vlm_rover_exploration/scripts/visualize_results.py
```

All figures will be rendered at $300\,\text{DPI}$ into `benchmark_visualizations/`:
- **`grouped_averages_exploration.png`**: Multi-panel bar charts of Explored Area ($m^2$), Coverage Ratio ($m^2/m$), Total Decision Steps, Proximity Rank, and Navigation Failures.
- **`grouped_averages_performance.png`**: Hardware consumption (CPU %, GPU %, RAM MB, VRAM MB).
- **`grouped_averages_time.png`**: Latency breakdown (Mission Duration, Inference Time per Step, Navigation Time per Step).
- **`grouped_distributions_*.png`**: Boxplots overlaid with jittered individual-run points.
- **`strategy_frequencies.png`**: Stacked percentage bar chart illustrating global strategy distribution per model (`spiral`, `perimeter`, `sweep`, `zig-zag`).

---

## Exploration Pipeline & Decision-Making Framework

### 1. 2D Polar-Cartesian Map Representation
The state machine converts the dynamic OccupancyGrid into an intuitive image representation centered on the rover's initial position ($x_0, y_0$):
- **Visual Palette**: Explored space is rendered in white ($255$), unexplored terrain in gray ($100$), and detected obstacles in bright yellow ($50$).
- **Spatial Reference**: A cyan crosshair aligned with the world frame marks `TOP`, `BOTTOM`, `LEFT`, and `RIGHT`, augmented by diagonal dividers for 8 directional quadrants (`TOP-LEFT`, `TOP-RIGHT`, etc.).
- **Frontier Detection & Centroid Labeling**: Boundaries between explored white cells and unexplored gray cells are clustered into discrete frontier candidates.
- **Euclidean Distance Projection**: Exact metric distances from the rover's current pose to all candidate IDs are mathematically calculated and provided directly to the model's context window.

```text
                  [TOP]
              \     |     /
      [TOP-LEFT]\   |   /[TOP-RIGHT]
                  \ | /
    [LEFT] --------(ROBOT)-------- [RIGHT]
                  / | \
   [BOTTOM-LEFT]/   |   \[BOTTOM-RIGHT]
              /     |     \
                 [BOTTOM]
```

---

### 2. 8-Step Structured Chain-of-Thought (CoT) Prompting
To prevent hallucinations and guarantee methodical multi-criteria spatial reasoning, the system prompt strictly mandates an 8-step deliberation template before selecting a waypoint:

1. **Step 1: Robot State Description**: Describe rover position and orientation relative to the global crosshair.
2. **Step 2: Initial Candidates**: List all visible frontier IDs detected on the current map.
3. **Step 3: Unexplored Terrain Inventory**: Identify and characterize all distinct gray unexplored regions (shape, scale, direction).
4. **Step 4: Global Strategy Selection**: Choose **one** systematic exploration pattern (`spiral`, `perimeter`, `sweep`, or `zig-zag`) and **one** direction (`clockwise` or `counter-clockwise`), maintaining strategic continuity from prior steps unless a justified pivot is required.
5. **Step 5: Directional Pruning**: Select up to 3 candidate IDs advancing the global pattern (`GLOBAL_CANDIDATES`).
6. **Step 6: Multi-Criteria Trade-Off (`GLOBAL_FINALIST`)**: Evaluate candidates against:
   - *Information Gain*: Volume of open unexplored space accessible from the frontier.
   - *Travel Cost*: Precise Euclidean distance.
   - *Heading Alignment*: Angular deviation between rover heading and target waypoint.
7. **Step 7: Residual Patch / Hole Scanning (`LOCAL_FINALIST`)**: Check for isolated, lingering pockets of unexplored gray space that break the global pattern. Select the closest candidate to prevent costly backtracking later.
8. **Step 8: Cost-Benefit Arbitration & Final Decision**: Perform an economic trade-off between the `GLOBAL_FINALIST` and `LOCAL_FINALIST` to output the winning target ID.

---

## Evaluation Metrics & Mathematical Formulation

### 1. Exploration Efficiency Metrics

#### Explored Area ($A_{\text{expl}}$)
Total metric surface area discovered by the rover within the operational radius:
$$A_{\text{expl}} = N_{\text{explored\_cells}} \times \Delta r^2 \quad [\text{m}^2]$$
where $\Delta r$ is the map grid resolution in meters per cell ($\Delta r = 0.05\,\text{m}$).

#### Coverage Ratio ($\text{CR}$)
The ratio of newly mapped terrain surface area per meter of physical travel:
$$\text{CR} = \frac{A_{\text{expl}}}{d_{\text{real}}} \quad \left[\frac{\text{m}^2}{\text{m}}\right]$$
- **Higher $\text{CR}$**: Demonstrates efficient path selection without redundant loops or self-intersecting trajectories.

#### Area Explored per Step ($\eta_{\text{step}}$)
$$\eta_{\text{step}} = \frac{A_{\text{expl}}}{N_{\text{steps}}} \quad \left[\frac{\text{m}^2}{\text{step}}\right]$$

---

### 2. Navigation Dynamics & Odometry Drift

#### Trajectory Length & Odometry Error ($E_{\text{odom}}$)
Measures the cumulative difference between wheel odometry estimates ($d_{\text{est}}$) and true simulation ground-truth position ($d_{\text{real}}$):
$$E_{\text{odom}} = \left| d_{\text{est}} - d_{\text{real}} \right| \quad [\text{m}]$$

#### Target Jump Distance ($\Delta d_{\text{jump}}$)
Euclidean distance between consecutive frontier selections:
$$\Delta d_{\text{jump}} = \sqrt{(x_k - x_{k-1})^2 + (y_k - y_{k-1})^2} \quad [\text{m}]$$

---

### 3. Cognitive & Spatial Decision Metrics

#### Proximity Rank ($R_p$)
Measures whether the VLM favors greedy local frontiers ($R_p = 1$) or long-range strategic targets ($R_p > 1$). When sorted ascendingly by Euclidean distance from the rover:
$$R_p = \text{Index of selected target ID in } \text{sort}_{\text{dist}}(\text{Frontiers}) \in \{1, 2, \dots, K\}$$

#### Navigation Failure Count ($N_{\text{fail}}$)
Total occurrences where Nav2 aborted path tracking due to unrecoverable obstacles or local trapped configurations:
$$N_{\text{fail}} = \sum_{k=1}^{N_{\text{steps}}} \mathbf{1}(\text{status}_k = \text{failed})$$

---

### 4. Hardware Profiling & Latency Breakdown
- **Inference Latency ($\bar{t}_{\text{inf}}$)**: Average wall-clock time spent inside the multimodal forward pass per step.
- **Navigation Latency ($\bar{t}_{\text{nav}}$)**: Average transit duration from waypoint dispatch until Nav2 arrival.
- **Resource Footprint**: Process-specific sampling of CPU usage ($\%$), GPU compute load ($\%$), system RAM ($\text{MB}$), and allocated VRAM ($\text{MB}$).

---

## Supported VLM Models & Configs

Configurations are located in `src/vlm_rover_exploration/vlm_rover_exploration_bringup/models/`:

| Model Identifier | Hugging Face Repository | Quantization | Context Window ($n_{\text{ctx}}$) | GPU Offload Layers | Chat Template |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **MiniCPM.yaml** | `openbmb/MiniCPM-o-4_5-gguf` | Q4_K_M | $8,192$ | $35$ | ChatML |
| **Qwen3-VL.yaml** | `unsloth/Qwen3.5-4B-GGUF` | Q4_K_M | $8,192$ | $30$ | ChatML |
| **InternVL3.yaml** | `mradermacher/InternVL3_5-4B-GGUF` | Q4_K_M | $8,192$ | $33$ | ChatML |

---

## Citations & Tool References

If you utilize this framework or benchmark methodology in your research, please cite the underlying components:

```bibtex
@article{yasmin2024,
  title={YASMIN: Yet Another State MachINe for ROS 2},
  author={Gonz{\'a}lez-Santamarta, Miguel {\'A}ngel and Camino-Castro, Roberto and Garc{\'i}a-P{\'e}rez, Diego and Rodriguez-Lera, Francisco J.},
  journal={SoftwareX},
  volume={26},
  pages={101683},
  year={2024},
  publisher={Elsevier}
}

@article{llamaros2023,
  title={LLaMA-ROS: A ROS 2 wrapper for LLaMA and multimodal models},
  author={Gonz{\'a}lez-Santamarta, Miguel {\'A}ngel and Camino-Castro, Roberto and Rodriguez-Lera, Francisco J.},
  journal={arXiv preprint arXiv:2311.12781},
  year={2023}
}

@software{llamacpp,
  title={{llama.cpp}: Port of Facebook's LLaMA model in C/C++},
  author={Gerganov, Georgi and contributors},
  url={https://github.com/ggerganov/llama.cpp},
  year={2023}
}

@inproceedings{minicpm2024,
  title={MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies},
  author={Hu, Shengding and Tu, Yuge and Han, Xu and He, Chaoqun and Wang, Cui and Zhao, Weilin and others},
  booktitle={arXiv preprint arXiv:2404.06395},
  year={2024}
}

@article{qwen2024,
  title={Qwen2.5-VL: Technical Report},
  author={Qwen Team},
  journal={arXiv preprint arXiv:2502.13923},
  year={2025}
}

@article{internvl2024,
  title={InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks},
  author={Chen, Zhe and Wu, Jiannan and Wang, Wenhai and Su, Weiyun and Chen, Guo and Xing, Sen and others},
  journal={arXiv preprint arXiv:2312.14238},
  year={2023}
}
```

---