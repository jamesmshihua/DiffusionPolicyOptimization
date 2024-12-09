# Diffusion Policy Policy Optimization -- Tensorflow Version

## Installation
1. Clone the repository
```console
git clone https://github.com/jamesmshihua/DiffusionPolicyOptimization.git
cd dppo
```

2. Install core dependencies with a conda environment on a Linux machine with a Nvidia GPU.
```console
conda create -n dppo python=3.12 -y
conda activate dppo
pip install 'tensorflow[and-cuda]'
pip install tensorflow-probability tf-keras tf-agents
pip install -e .[gym]
```

3. Install MuJoCo for Gym. Instructions are available [here](https://github.com/openai/mujoco-py).

4. Set environment variables for data and logging directory (default is `data/` and `log/`), and set WandB entity (username or team name)
```console
source script/set_path.sh
```

## Usage - Pre-training

**Note**: As the pytorch checkpoint files cannot be directly used for `tensorflow` models, you are advised to pretrain the model locally.

First create a directory as the parent directory of the pre-training data and set the environment variable for it.
```console
export DPPO_DATA_DIR=/path/to/data
```

Pre-training data for all tasks are pre-processed and can be found at [here](https://drive.google.com/drive/folders/1AXZvNQEKOrp0_jk1VLepKh_oHCg_9e3r?usp=drive_link). Pre-training script will download the data (including normalization statistics) automatically to the data directory.

The configuration file can be found under `cfg/<env>/pretrain/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
```console
# Gym - hopper/walker2d/halfcheetah
python script/run.py --config-name=pre_diffusion_mlp --config-dir=cfg/gym/pretrain/hopper-medium-v2
```

## Usage - Fine-tuning
Two configuration files can be found under `cfg/<env>/finetune/`. A new WandB project may be created based on `wandb.project` in the config file; set `wandb=null` in the command line to test without WandB logging.
```console
# Gym - hopper/walker2d/halfcheetah (debug)
python script/run.py --config-name=ft_ppo_diffusion_mlp --config-dir=cfg/gym/finetune/hopper-v2

# Gym - hopper/walker2d/halfcheetah (optimized run)
python script/run.py --config-name=ft_ppo_diffusion_mlp_run --config-dir=cfg/gym/finetune/hopper-v2
```
