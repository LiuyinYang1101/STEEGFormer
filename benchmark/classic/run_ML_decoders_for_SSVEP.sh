#!/bin/bash -l
#SBATCH --cluster wice # genius
#SBATCH --partition bigmem # batch bigmem
#SBATCH --account=user1 # your HPC account
#SBATCH --job-name=ssvep_task%a # set a task name

#SBATCH --array=0

#SBATCH --chdir="/../fundation_model" # location of the project
#SBATCH --ntasks=72
#SBATCH --mem-per-cpu=28000M
#SBATCH --time=48:00:00
#SBATCH --error="/logs/%x_%a.e"
#SBATCH --output="/logs/%x_%a.o"
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=xxx.xx@xx.com # your email to receive the task notice

# 1) Activate conda environment
export PATH="/../miniconda/bin:$PATH"
source /../miniconda/etc/profile.d/conda.sh
conda activate env2 # environment for each downstream tasks

# 2) Define grid
downstream_tasks=(binocular_ssvep)
evaluation_schemes=(per-subject) # population leave-one-out-finetuning per-subject
models=(trca) # xdawn_lda xdawncov_mdm xdawncov_ts_svm erpcov_mdm dcpm

N1=${#downstream_tasks[@]}     # = 1
N2=${#evaluation_schemes[@]}   # = 3
N3=${#models[@]}               # = 7

# Map SLURM_ARRAY_TASK_ID → i, j, k
idx=$SLURM_ARRAY_TASK_ID
i=$(( idx / (N2 * N3) ))
rem=$(( idx % (N2 * N3) ))
j=$(( rem / N3 ))
k=$(( rem % N3 ))

downstream_task=${downstream_tasks[$i]}
evaluation_scheme=${evaluation_schemes[$j]}
model=${models[$k]}

echo "[$(date)] Combo #$idx: model=$model, task=$downstream_task, scheme=$evaluation_scheme"

# 3) Launch training
python run_ML_decoders_for_SSVEP.py \
  --model "$model" \
  --downstream_task "$downstream_task" \
  --evaluation_scheme "$evaluation_scheme" \
  --seed 3407