#!/bin/sh
#SBATCH -J UNet_res
#SBATCH -p cas_v100nv_8
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH --gres=gpu:8
#SBATCH --comment pytorch

module load python/3.7.1
source activate UNet

python train_res.py

exit 0
