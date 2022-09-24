#! /bin/bash

nWealth=20
nZ=10
nV=10
nVtilde=10
V_bar=1.0
Vtilde_bar=1.0
sigma_K_norm=0.04
sigma_Z_norm=0.0141
sigma_V_norm=0.132
sigma_Vtilde_norm=0.1
trial=2
weight1=0.0
boundary1=2
weight2=0.0
boundary2=5
chi_position_tolerance=0.0
chi_value_tolerance=0.0
chi_max_iterations=500

if (( $(echo "$sigma_Vtilde_norm == 0.0" |bc -l) )); then
    domain_folder=WZV_sigma_K_${sigma_K_norm}_sigma_Z_${sigma_Z_norm}_sigma_V_${sigma_V_norm}_sigma_Vtilde_${sigma_Vtilde_norm}
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
elif (( $(echo "$sigma_V_norm == 0.0" |bc -l) )); then
    domain_folder=WZVtilde_sigma_K_${sigma_K_norm}_sigma_Z_${sigma_Z_norm}_sigma_V_${sigma_V_norm}_sigma_Vtilde_${sigma_Vtilde_norm}
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
else
    domain_folder=WZVVtilde_sigma_K_${sigma_K_norm}_sigma_Z_${sigma_Z_norm}_sigma_V_${sigma_V_norm}_sigma_Vtilde_${sigma_Vtilde_norm}
    mkdir -p ./job-outs/$domain_folder
    mkdir -p ./bash/$domain_folder
fi

for chiUnderline in 0.5 1.0 0.2
do 
    for a_e in 0.15
    do
        for a_h in 0.1 0.05
        do
            for gamma_e in 1.0
            do
                for gamma_h in 3.0
                do
                    for psi_e in 0.5 1.5
                    do
                        for psi_h in 0.5 1.5
                        do
                            model_folder=chiUnderline_${chiUnderline}_a_e_${a_e}_a_h_${a_h}_gamma_e_${gamma_e}_gamma_h_${gamma_h}_psi_e_${psi_e}_psi_h_${psi_h}
                            mkdir -p ./job-outs/$domain_folder/$model_folder
                            mkdir -p ./bash/$domain_folder/$model_folder

                            for points_size in 2
                            do
                                for iter_num in 15
                                do                                                                                                
                                    for XiE_layers in 3
                                    do 
                                        for XiH_layers in 3
                                        do  
                                            for kappa_layers in 3
                                            do
                                                for W_fix in 14
                                                do                                                                                                
                                                    for Z_fix in 4
                                                    do 
                                                        for V_fix in 4
                                                        do  
                                                            for Vtilde_fix in 4
                                                            do
                                                                layer_folder=trial_${trial}_XiE_layers_${XiE_layers}_XiH_layers_${XiH_layers}_kappa_layers_${kappa_layers}_points_size_${points_size}_iter_num_${iter_num}
                                                                mkdir -p ./job-outs/$domain_folder/$model_folder/$layer_folder
                                                                mkdir -p ./bash/$domain_folder/$model_folder/$layer_folder

                                                                touch ./bash/$domain_folder/$model_folder/$layer_folder/run.sh
                                                                tee ./bash/$domain_folder/$model_folder/$layer_folder/run.sh << EOF
#!/bin/bash

#SBATCH --job-name=run
#SBATCH --output=./job-outs/$domain_folder/$model_folder/$layer_folder/run.out
#SBATCH --error=./job-outs/$domain_folder/$model_folder/$layer_folder/run.err
#SBATCH --time=0-24:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=56G

module load tensorflow/2.1
module unload python
module load python/anaconda-2021.05

python3 NN_structure.py    --XiE_layers ${XiE_layers} --XiH_layers ${XiH_layers} --kappa_layers ${kappa_layers}
python3 standard_BFGS.py   --chiUnderline ${chiUnderline} --a_e ${a_e} --a_h ${a_h} --gamma_e ${gamma_e} --gamma_h ${gamma_h} --psi_e ${psi_e} --psi_h ${psi_h} --nWealth ${nWealth} --nZ ${nZ} --nV ${nV} --nVtilde ${nVtilde} --V_bar ${V_bar} --Vtilde_bar ${Vtilde_bar} --sigma_V_norm ${sigma_V_norm} --sigma_Vtilde_norm ${sigma_Vtilde_norm} --XiE_layers ${XiE_layers} --XiH_layers ${XiH_layers} --kappa_layers ${kappa_layers} --weight1 ${weight1} --boundary1 ${boundary1} --weight2 ${weight2} --boundary2 ${boundary2} --points_size ${points_size} --iter_num ${iter_num} --trial ${trial} --chi_position_tolerance ${chi_position_tolerance} --chi_value_tolerance ${chi_value_tolerance} --chi_max_iterations ${chi_max_iterations} --W_fix ${W_fix} --Z_fix ${Z_fix} --V_fix ${V_fix} --Vtilde_fix ${Vtilde_fix}

EOF
                                                                sbatch ./bash/$domain_folder/$model_folder/$layer_folder/run.sh
                                                            done
                                                        done
                                                    done
                                                done        
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done
