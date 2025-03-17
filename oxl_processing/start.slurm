#!/bin/bash
# Job name:
#SBATCH --job-name=objaverse_xl_rendering
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=15
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
# #SBATCH --output=res_objaverse_xl_rendering.txt
#SBATCH --output=./logs/slurm-%A.%a.out # stdout file
#SBATCH --error=./logs/slurm-%A.%a.err  # stderr file
# #SBATCH --array=0-2

## Command(s) to run (example):
export PATH=/mnt/damian/miniconda3/bin:$PATH 
eval "$(conda shell.bash hook)"
conda activate dfn
cd /mnt/damian/Projects/Diffus3D/02_multi-view-rendering

# Save the content of CUDA_VISIBLE_DEVICES to a file
echo $CUDA_VISIBLE_DEVICES > ~/cuda_visible_devices.txt

python main.py
# python objaverse_xl_batched_renderer.py --sample_size 5000 --n_processes 14 --batch_size 50 --gpu_batch_size 40

# python objaverse_xl_batched_renderer.py --sample_size 10 --n_processes 10 --batch_size 10 --gpu_batch_size 10

# python objaverse_xl_batched_renderer.py --start_index 157050 --sample_size 950 --n_processes 8 --batch_size 50 --gpu_batch_size 40

wait