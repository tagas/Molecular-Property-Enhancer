#!/bin/bash
#SBATCH --job-name=propen
#SBATCH --error ./sbatch_output/%J.err
#SBATCH --output ./sbatch_output/%J.out
#SBATCH --nodes 1
#SBATCH --cpus-per-task 8
#SBATCH --ntasks-per-node 1
#SBATCH --partition gpu2
#SBATCH --gpus 1

source ~/.bashrc
micromamba activate mdev3110

python run_propen.py --experiment_name henri_c1_strict \
    --input_file s3://prescient-data-dev/sandbox/dwyerh/propen/c1_train_strict.csv \
    --n_samples 200 \
    --property affinity_pkd_raw \
    --same_epitope_threshold 12 \
    --property_th 0.3 \
    --heavy_light 0 \
    --target C1s \
    --temps 0.1,0.3,0.5,0.7,1,1.5,2 \
    --weight_decay 1e-4 \
    --lr 1e-4 \
    --output_dir s3://prescient-data-dev/sandbox/dwyerh/propen/results_strict_10_lr_1e-4_wd_1e-4_adam_h/
