#!/bin/bash -l
#SBATCH --cluster dodrio
#SBATCH --partition gpu_rome_a100_40
#SBATCH --time=10:30:00
#SBATCH --account=2025_500
#SBATCH --job-name="challenge1_vit_small_all_simple_lora_peft_0_lora_rank_8_no_dropout"
### e.g. request 1 nodes with 2 gpu each, totally 4 gpus (WORLD_SIZE==4)
### Note: --gres=gpu:x should equal to ntasks-per-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --mail-type="END,FAIL,TIME_LIMIT"
#SBATCH --mail-user="liuyin.yang@kuleuven.be"

#SBATCH --chdir=/dodrio/scratch/projects/2025_500/shared/new_lora_exp
#SBATCH --output=/dodrio/scratch/projects/2025_500/shared/new_lora_exp/output_files/%x-%j.out
#SBATCH --error=/dodrio/scratch/projects/2025_500/shared/new_lora_exp/output_files/%x-%j.error

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=25102
export WORLD_SIZE=2

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load timm/1.0.8-foss-2023a-CUDA-12.1.1
module load wandb/0.16.1-GCC-12.3.0
module load einops/0.7.0-GCCcore-12.3.0
module load h5py/3.9.0-foss-2023a

source "/dodrio/scratch/projects/2025_500/venvs/peft-foss2023a/bin/activate"


# (Optional) Make PyTorch verbose for DDP/debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print paths for debugging
echo "Python executable: $(which python)"
echo "PYTHONPATH head: $(echo $PYTHONPATH | tr ':' '\n' | head -n 3)"

# Run the distributed training script with srun
srun --mpi=pmi2 --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(which python) ddp_finetune_eeg.py --challenge 'challenge1' --model 'vit_small_patch16' --output_dir '/dodrio/scratch/projects/2025_500/shared/new_lora_exp/exp_out/challenge1' --vit_pretrained_model_dir '/dodrio/scratch/projects/2025_038/mae/checkpoint/experiment3_small/checkpoint-336.pth' --use_lora --lora_last_n 0 --lora_r 8 --lora_wd_zero --lora_lr_scale 1.0 --lora_no_lrd
