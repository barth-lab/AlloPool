# AlloPool

## Installation

Start by cloning this github repo

```
git clone https://github.com/barth-lab/AlloPool.git
cd AlloPool
```

Next, set up a conda environment using the environment file provided in the root directory. The code was tested on `python 3.11.4` with `CUDA 11.4` and `torch 2.0.1`. 
```
conda env create --name allopool -f environment.yml
conda activate allopool
```

## Usage

### Setting up your data

Before training the model you need to set up a data folder. AlloPool takes as input a set of MD trajectories (xtc or dcd files) and their corresponding protein structure (pdb or gro). Place your files in a new folder and copy its path {PATH_TO_YOUR_DATA}.

### Creating additional directories

In order to run AlloPool and store its results, you must first create additional directories in the root folder. Navigate to the root directory and execute
```
mkdir cache/{run_name} checkpoints/{run_name} results/{run_name}_md runs/{run_name}
```
Substitute `{run_name}` by your desired run name, **keeping it consistent across all new folders.**

### Specifying argument files

AlloPool used two json files to specify some of the model's hyperparameters. The first one is placed inside `cache/test_md/{args_file}.json`
```
{
	"sim_type": null,
	"timesteps": 1,
	"last_frame": null,
	"first_frame": null,
	"replicates": null,
	"dir": "{PATH_TO_YOUR_DATA}"
}
```

Substitute the *PATH_TO_YOUR_DATA* for that you saved in the first step. The second json file has to be at `results/test_md/{args_file}.json`

```
{
	"accumulation_step": 16,
	"batch_size": 5,
	"cache_name": "{run_name}",
	"contact_coef": 0,
	"cst_lr": false,
	"decoder_depth": 4,
	"decoder_dim": 128,
	"decoder_dim_head": 64,
	"decoder_heads": 8,
	"drop_angles": false,
	"drop_cbs": false,
	"drop_x": false,
	"input_dim": 3,
	"lr": 0.001,
	"momentum": 0.9,
	"n_epochs": 100,
	"no_cuda": null,
	"no_shuffle": false,
	"normalize": false,
	"num_time_steps": 1,
	"overlap": 1,
	"persistence": 0.5,
	"pool_attn_heads": 1,
	"pool_depth": 4,
	"pool_in": 128,
	"pool_ratio": 0.75,
	"rmsd_coef": 0,
	"run_name": "{run_name}",
	"seed": null,
	"seq_len": 286,
	"stride": 10,
	"temporal_in": 128,
	"temporal_out": 128,
	"test_only": false,
	"test_replicates": null,
	"threshold": 1.0,
	"transformer_depth": 4,
	"transformer_heads": 4,
	"transl_coef": 1,
	"weight_decay": 0.1
}
```

Make sure to substitute `{run_name}` by the same folder name you used when creating the additional directories 

### Training AlloPool

Once you setup your data, the new directories and their corresponding json files, you are ready to train AlloPool. Launch the training script from the root directory

```
python train.py --load-args {args_file}.json --no-cache --dir {PATH_TO_YOUR_DATA}
```

## Common errors
<details>
<summary><b> Mismatch between number of atoms in trajectory and structure files</b></summary>
<br>
This error happens when the trajectory contains additional atoms besides the protein. For example, in a simulation of a protein-ligand it is possible that when saving the trajectory of the protein you also include the ligand coordinates, in this case you will have an extra set of atoms with respect to the protein structure. Make sure that you are only using the protein coordinates as input for the model
</details>
