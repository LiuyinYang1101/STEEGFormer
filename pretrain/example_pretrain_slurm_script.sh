#!/bin/bash -l
#SBATCH --cluster xxxx
#SBATCH --partition xxxx
#SBATCH --time=01:00:00
#SBATCH --account=xxxx
#SBATCH --job-name="mae_large_2GPU"
### e.g. request 1 nodes with 2 gpu each, totally 2 gpus (WORLD_SIZE==2)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="xxxx"

#SBATCH --chdir=xxxx
#SBATCH --output=xxxx/%x-%j.out
#SBATCH --error=xxxx/%x-%j.error

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=13416
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

#set up your Python env
module load Python/3.11.5-GCCcore-13.2.0


# Print paths for debugging
echo "Python executable: $(which python)"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# Run the distributed training script with srun
srun --mpi=pmi2 --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(which python) ddp_train_eeg.py --model 'mae_vit_large_patch16' --resume 'latest' --output_dir './checkpoint/experiment5_large' --config_path './global_setting_pretrainDEBUG.json' --distributed true --device 'cuda' --batch_size 128 --accum_iter 4 --num_workers 8 --epochs 401
