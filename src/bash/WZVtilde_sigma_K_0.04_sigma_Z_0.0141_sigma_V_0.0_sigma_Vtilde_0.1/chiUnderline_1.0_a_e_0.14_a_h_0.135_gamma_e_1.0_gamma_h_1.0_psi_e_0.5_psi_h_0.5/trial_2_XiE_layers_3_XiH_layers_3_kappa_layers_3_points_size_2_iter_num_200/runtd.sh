#!/bin/bash

#SBATCH --job-name=runtd
#SBATCH --output=./job-outs/WZVtilde_sigma_K_0.04_sigma_Z_0.0141_sigma_V_0.0_sigma_Vtilde_0.1/chiUnderline_1.0_a_e_0.14_a_h_0.135_gamma_e_1.0_gamma_h_1.0_psi_e_0.5_psi_h_0.5/trial_2_XiE_layers_3_XiH_layers_3_kappa_layers_3_points_size_2_iter_num_200/runtd.out
#SBATCH --error=./job-outs/WZVtilde_sigma_K_0.04_sigma_Z_0.0141_sigma_V_0.0_sigma_Vtilde_0.1/chiUnderline_1.0_a_e_0.14_a_h_0.135_gamma_e_1.0_gamma_h_1.0_psi_e_0.5_psi_h_0.5/trial_2_XiE_layers_3_XiH_layers_3_kappa_layers_3_points_size_2_iter_num_200/runtd.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=56G

module load tensorflow/2.1
module unload python
module load python/anaconda-2021.05

python3 NN_structure.py    --XiE_layers 3 --XiH_layers 3 --kappa_layers 3
python3 standard_BFGS.py   --chiUnderline 1.0 --a_e 0.14 --a_h 0.135 --gamma_e 1.0 --gamma_h 1.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.1 --XiE_layers 3 --XiH_layers 3 --kappa_layers 3 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 200 --trial 2 --chi_position_tolerance 0.0 --chi_value_tolerance 0.0 --chi_max_iterations 500 --W_fix 49 --Z_fix 14 --V_fix 0 --Vtilde_fix 14
python3 standard_variable.py   --chiUnderline 1.0 --a_e 0.14 --a_h 0.135 --gamma_e 1.0 --gamma_h 1.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.1 --XiE_layers 3 --XiH_layers 3 --kappa_layers 3 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 200 --trial 2 --chi_position_tolerance 0.0 --chi_value_tolerance 0.0 --chi_max_iterations 500 --W_fix 49 --Z_fix 14 --V_fix 0 --Vtilde_fix 14
python3 standard_moments.py   --chiUnderline 1.0 --a_e 0.14 --a_h 0.135 --gamma_e 1.0 --gamma_h 1.0 --psi_e 0.5 --psi_h 0.5 --nWealth 100 --nZ 30 --nV 0 --nVtilde 30 --V_bar 1.0 --Vtilde_bar 1.0 --sigma_V_norm 0.0 --sigma_Vtilde_norm 0.1 --XiE_layers 3 --XiH_layers 3 --kappa_layers 3 --weight1 0.0 --boundary1 2 --weight2 0.0 --boundary2 5 --points_size 2 --iter_num 200 --trial 2 --chi_position_tolerance 0.0 --chi_value_tolerance 0.0 --chi_max_iterations 500 --W_fix 49 --Z_fix 14 --V_fix 0 --Vtilde_fix 14 

