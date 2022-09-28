import json
import tensorflow as tf 
import pandas as pd
import time 
import os
from standard_para import *
from standard_training import *
from standard_plotting import *
import argparse
float_formatter = "{0:.4f}"

## Parameter parser
parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--nWealth",type=int,default=100)
parser.add_argument("--nZ",type=int,default=30)
parser.add_argument("--nV",type=int,default=30)
parser.add_argument("--nVtilde",type=int,default=0)
parser.add_argument("--V_bar",type=float,default=1.0)
parser.add_argument("--Vtilde_bar",type=float,default=0.0)
parser.add_argument("--sigma_K_norm",type=float,default=0.04)
parser.add_argument("--sigma_Z_norm",type=float,default=0.0141)
parser.add_argument("--sigma_V_norm",type=float,default=0.132)
parser.add_argument("--sigma_Vtilde_norm",type=float,default=0.0)

parser.add_argument("--chiUnderline",type=float,default=1.0)
parser.add_argument("--a_e",type=float,default=0.14)
parser.add_argument("--a_h",type=float,default=0.135)
parser.add_argument("--gamma_e",type=float,default=1.0)
parser.add_argument("--gamma_h",type=float,default=1.0)
parser.add_argument("--psi_e",type=float,default=1.0)
parser.add_argument("--psi_h",type=float,default=1.0)

parser.add_argument("--XiE_layers",type=int,default=5)
parser.add_argument("--XiH_layers",type=int,default=5)
parser.add_argument("--kappa_layers",type=int,default=5)
parser.add_argument("--points_size",type=int,default=2)
parser.add_argument("--iter_num",type=int,default=10)
parser.add_argument("--trial",type=int,default=1)

parser.add_argument("--weight1",type=float,default=30.0)
parser.add_argument("--boundary1",type=int,default=2)
parser.add_argument("--weight2",type=float,default=100.0)
parser.add_argument("--boundary2",type=int,default=5)
parser.add_argument("--chi_position_tolerance",type=float,default=0.0)
parser.add_argument("--chi_value_tolerance",type=float,default=0.0)
parser.add_argument("--chi_max_iterations",type=int,default=500)

parser.add_argument("--W_fix",type=int,default=5)
parser.add_argument("--Z_fix",type=int,default=5)
parser.add_argument("--V_fix",type=int,default=5)
parser.add_argument("--Vtilde_fix",type=int,default=5)
args = parser.parse_args()

## Domain parameters
nWealth           = args.nWealth
nZ                = args.nZ
nV                = args.nV
nVtilde           = args.nVtilde
V_bar             = args.V_bar
Vtilde_bar        = args.Vtilde_bar
sigma_K_norm      = args.sigma_K_norm
sigma_Z_norm      = args.sigma_Z_norm
sigma_V_norm      = args.sigma_V_norm
sigma_Vtilde_norm = args.sigma_Vtilde_norm
domain_list       = [nWealth, nZ, nV, nVtilde, V_bar, Vtilde_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, sigma_Vtilde_norm]
sigma_K, sigma_Z, sigma_V, sigma_Vtilde = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in [sigma_K_norm, sigma_Z_norm, sigma_V_norm, sigma_Vtilde_norm]]
if sigma_Vtilde_norm == 0:
  domain_folder = 'WZV' + '_sigma_K_' + sigma_K + '_sigma_Z_' + sigma_Z + '_sigma_V_' + sigma_V + '_sigma_Vtilde_' + sigma_Vtilde
  nDims = 3
elif sigma_V_norm == 0:
  domain_folder = 'WZVtilde' + '_sigma_K_' + sigma_K + '_sigma_Z_' + sigma_Z + '_sigma_V_' + sigma_V + '_sigma_Vtilde_' + sigma_Vtilde
  nDims = 3
else:
  domain_folder = 'WZVVtilde' + '_sigma_K_' + sigma_K + '_sigma_Z_' + sigma_Z + '_sigma_V_' + sigma_V + '_sigma_Vtilde_' + sigma_Vtilde
  nDims = 4

## Model parameters
chiUnderline      = args.chiUnderline
a_e               = args.a_e
a_h               = args.a_h
gamma_e           = args.gamma_e
gamma_h           = args.gamma_h
psi_e             = args.psi_e
psi_h             = args.psi_h
parameter_list    = [chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h]
chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

## NN layer parameters
XiE_layers        = args.XiE_layers
XiH_layers        = args.XiH_layers
kappa_layers      = args.kappa_layers
points_size       = args.points_size
iter_num          = args.iter_num
trial             = args.trial
layer_folder =  'trial_' + str(trial) + '_XiE_layers_' + str(XiE_layers) +'_XiH_layers_' + str(XiH_layers) +'_kappa_layers_'+ str(kappa_layers) + '_points_size_' + str(points_size) + '_iter_num_' + str(iter_num)

w1                = args.weight1
b1                = args.boundary1
w2                = args.weight2
b2                = args.boundary2
chi_ptol          = args.chi_position_tolerance
chi_vtol          = args.chi_value_tolerance
chi_max_iter      = args.chi_max_iterations

## Plot parameters
W_fix               = args.W_fix
Z_fix               = args.Z_fix
V_fix               = args.V_fix
Vtilde_fix          = args.Vtilde_fix

## Working directory
workdir = os.path.dirname(os.getcwd())
srcdir = workdir + '/src/'
datadir = workdir + '/data/' + domain_folder + '/' + model_folder + '/'
outputdir = workdir + '/output/' + domain_folder + '/' + model_folder + '/' + layer_folder + '/'
docdir = workdir + '/doc/' + domain_folder + '/' + model_folder + '/'+ layer_folder + '/'
os.makedirs(datadir,exist_ok=True)
os.makedirs(docdir,exist_ok=True)
os.makedirs(outputdir,exist_ok=True)

## Load MFR SUite results back from memory
mfrSuite_resdir = os.path.dirname(workdir) + '/mfrSuite_solution/' + domain_folder + '/' + model_folder + '/'
if nDims == 3:
  with open(mfrSuite_resdir+'status.json') as f:
    status= json.load(f)
  status = status['status']
else:
  status = '3'

## Load parameter set
with open(datadir + 'parameters_NN.json') as json_file:
    paramsFromFile = json.load(json_file)
params = setModelParametersFromFile(paramsFromFile, nDims, chi_ptol, chi_vtol, chi_max_iter)

load_list = ['W_NN','Z_NN','V_NN','Vtilde_NN','logXiE_NN','logXiH_NN','XiE_NN','XiH_NN','kappa_NN','q_NN',\
             'sigmaK_NN', 'sigmaZ_NN','sigmaV_NN','sigmaVtilde_NN','muK_NN','muZ_NN', 'muV_NN','muVtilde_NN','chi_NN',\
             'sigmaW_NN','sigmaQ_NN','sigmaR_NN','deltaE_NN','deltaH_NN','PiH_NN','PiE_NN','betaE_NN','betaH_NN',
             'muW_NN', 'muQ_NN','r_NN','dent_NN',\
             'Fe_NN', 'firstCoefsE_NN', 'secondCoefsE_NN', 'HJB_E_NN','Fh_NN', 'firstCoefsH_NN', 'secondCoefsH_NN', 'HJB_H_NN','kappa_min_NN']

moment_list = ['W','Z','V','Vtilde','logXiE','logXiH','XiE','XiH','kappa','q', 'chi','sigmaR_first_shock','sigmaR_second_shock','sigmaR_third_shock','PiH_first_shock','PiH_second_shock','PiH_third_shock','PiE_first_shock','PiE_second_shock','PiE_third_shock','r','dent']

W_NN, Z_NN, V_NN, Vtilde_NN, logXiE_NN, logXiH_NN, XiE_NN, XiH_NN, kappa_NN, q_NN,\
sigmaK_NN, sigmaZ_NN, sigmaV_NN, sigmaVtilde_NN, muK_NN, muZ_NN, muV_NN, muVtilde_NN, chi_NN,\
sigmaW_NN, sigmaQ_NN, sigmaR_NN, deltaE_NN, deltaH_NN, PiH_NN, PiE_NN, betaE_NN, betaH_NN,\
muW_NN, muQ_NN, r_NN, dent_NN,\
Fe_NN, firstCoefsE_NN, secondCoefsE_NN, HJB_E_NN, Fh_NN, firstCoefsH_NN, secondCoefsH_NN, HJB_H_NN, kappa_min_NN = [np.load(outputdir +i+'.npy')  for i in load_list]
V_NN = V_NN.round(decimals=14)

NN_variables = pd.DataFrame([W_NN, Z_NN, V_NN, Vtilde_NN, logXiE_NN[:,0], logXiH_NN[:,0], XiE_NN[:,0], XiH_NN[:,0], kappa_NN[:,0], q_NN[:,0], chi_NN[:,0], sigmaR_NN[:,0], sigmaR_NN[:,1], sigmaR_NN[:,2], PiH_NN[:,0], PiH_NN[:,1], PiH_NN[:,2], PiE_NN[:,0], PiE_NN[:,1], PiE_NN[:,2], r_NN[:,0], dent_NN],\
                            index = moment_list).T

NN_variables_norm = pd.DataFrame([HJB_E_NN[:,0], HJB_H_NN[:,0],kappa_min_NN[:,0]],index = ['HJB_E','HJB_H','kappa_min']).T
NN_variables_norm = pd.concat([NN_variables,NN_variables_norm],axis=1)

W_grid = NN_variables['W'].unique()
Z_grid = NN_variables['Z'].unique()
V_grid = NN_variables['V'].unique()
Vtilde_grid = NN_variables['Vtilde'].unique()

W_boundary_points = int(round(W_grid.shape[0]*0.05,0)) if int(round(W_grid.shape[0]*0.05,0))!=0 else 1
W_inner_points = np.sort(np.unique(W_NN))[W_boundary_points:-W_boundary_points]

Z_boundary_points = int(round(Z_grid.shape[0]*0.05,0)) if int(round(Z_grid.shape[0]*0.05,0))!=0 else 1
Z_inner_points = np.sort(np.unique(Z_NN))[Z_boundary_points:-Z_boundary_points]

if sigma_Vtilde_norm == 0:
  V_boundary_points = int(round(V_grid.shape[0]*0.05,0)) if int(round(V_grid.shape[0]*0.05,0))!=0 else 1
  V_inner_points = np.sort(np.unique(V_NN))[V_boundary_points:-V_boundary_points]
elif sigma_V_norm == 0:
  Vtilde_boundary_points = int(round(Vtilde_grid.shape[0]*0.05,0)) if int(round(Vtilde_grid.shape[0]*0.05,0))!=0 else 1
  Vtilde_inner_points = np.sort(np.unique(Vtilde_NN))[Vtilde_boundary_points:-Vtilde_boundary_points]
else:
  V_boundary_points = int(round(V_grid.shape[0]*0.05,0)) if int(round(V_grid.shape[0]*0.05,0))!=0 else 1
  V_inner_points = np.sort(np.unique(V_NN))[V_boundary_points:-V_boundary_points]

  Vtilde_boundary_points = int(round(Vtilde_grid.shape[0]*0.05,0)) if int(round(Vtilde_grid.shape[0]*0.05,0))!=0 else 1
  Vtilde_inner_points = np.sort(np.unique(Vtilde_NN))[Vtilde_boundary_points:-Vtilde_boundary_points]


fix_dict = {'W':W_fix,'Z':Z_fix,'V':V_fix,'Vtilde':Vtilde_fix}


if sigma_Vtilde_norm == 0:

    NN_variables_inner = NN_variables[(NN_variables['W'].isin(W_inner_points)) & (NN_variables['Z'].isin(Z_inner_points))& (NN_variables['V'].isin(V_inner_points))]
    NN_variables_norm_inner = NN_variables_norm[(NN_variables_norm['W'].isin(W_inner_points)) & (NN_variables_norm['Z'].isin(Z_inner_points))& (NN_variables_norm['V'].isin(V_inner_points))]

    fixed_variable = ['V','Z','W']
    fixed_points = [fix_dict[i] for i in fixed_variable]
    NN_slice = [NN_variables.loc[(NN_variables[v] == NN_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
    NN_slice_inner = [NN_variables_inner.loc[(NN_variables_inner[v] == NN_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
    fixed_values = [v + ' fixed at ' + str("{0:.4f}".format(NN_variables[v].unique()[fixed_points[fixed_variable.index(v)]])) for v in fixed_variable]
    
    fixed_variable = [['W','Z'],['W','V'],['Z','V']]
    fixed_points = [[fix_dict[i] for i in j] for j in fixed_variable]
    NN_slice_2d = [NN_variables.loc[(NN_variables[i[0]]==NN_variables[i[0]].unique()[fixed_points[fixed_variable.index(i)][0]]) &(NN_variables[i[1]]==NN_variables[i[1]].unique()[fixed_points[fixed_variable.index(i)][1]])].drop(i,axis=1).reset_index(drop=True) for i in fixed_variable]
    fixed_values_2d = [v[0] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])) + ', ' + v[1] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])) for v in fixed_variable]

    NN_first_moment_conditional_WZ, NN_second_moment_conditional_WZ = calc_moment(NN_variables, ['W','Z'], moment_list)
    NN_first_moment_conditional_WV, NN_second_moment_conditional_WV = calc_moment(NN_variables, ['W','V'], moment_list)
    NN_first_moment_conditional_ZV, NN_second_moment_conditional_ZV = calc_moment(NN_variables, ['Z','V'], moment_list)

    NN_first_moment_conditional_W, NN_second_moment_conditional_W = calc_moment(NN_variables, ['W'], moment_list)
    NN_first_moment_conditional_Z, NN_second_moment_conditional_Z = calc_moment(NN_variables, ['Z'], moment_list)
    NN_first_moment_conditional_V, NN_second_moment_conditional_V = calc_moment(NN_variables, ['V'], moment_list)

    NN_first_moments = [NN_first_moment_conditional_WZ, NN_first_moment_conditional_WV, NN_first_moment_conditional_ZV]
    NN_second_moments = [NN_second_moment_conditional_WZ, NN_second_moment_conditional_WV, NN_second_moment_conditional_ZV]

    NN_first_moments_2d = [NN_first_moment_conditional_W, NN_first_moment_conditional_Z, NN_first_moment_conditional_V]
    NN_second_moments_2d = [NN_second_moment_conditional_W, NN_second_moment_conditional_Z, NN_second_moment_conditional_V]

    NN_marginal_density = [NN_variables.groupby(['W','Z']).sum().reset_index(drop=False), NN_variables.groupby(['W','V']).sum().reset_index(drop=False),NN_variables.groupby(['V','Z']).sum().reset_index(drop=False)]
    NN_marginal_density_2d = [NN_variables.groupby(['W']).sum().reset_index(drop=False), NN_variables.groupby(['Z']).sum().reset_index(drop=False),NN_variables.groupby(['V']).sum().reset_index(drop=False)]

elif sigma_V_norm == 0:

    new_columns = NN_variables.columns.tolist()
    NN_variables = NN_variables[new_columns[:2] + [new_columns[3]]+ [new_columns[2]] +new_columns[4:]]
    NN_variables_inner = NN_variables[(NN_variables['W'].isin(W_inner_points)) & (NN_variables['Z'].isin(Z_inner_points))& (NN_variables['Vtilde'].isin(Vtilde_inner_points))]
    NN_variables_norm_inner = NN_variables_norm[(NN_variables_norm['W'].isin(W_inner_points)) & (NN_variables_norm['Z'].isin(Z_inner_points))& (NN_variables_norm['Vtilde'].isin(Vtilde_inner_points))]

    fixed_variable = ['Vtilde','Z','W']
    fixed_points = [fix_dict[i] for i in fixed_variable]
    NN_slice = [NN_variables.loc[(NN_variables[v] == NN_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
    NN_slice_inner = [NN_variables_inner.loc[(NN_variables_inner[v] == NN_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]

    fixed_values = [v + ' fixed at ' + str("{0:.4f}".format(NN_variables[v].unique()[fixed_points[fixed_variable.index(v)]])) for v in fixed_variable]
    
    fixed_variable = [['W','Z'],['W','Vtilde'],['Z','Vtilde']]
    fixed_points = [[fix_dict[i] for i in j] for j in fixed_variable]
    NN_slice_2d = [NN_variables.loc[(NN_variables[i[0]]==NN_variables[i[0]].unique()[fixed_points[fixed_variable.index(i)][0]]) &(NN_variables[i[1]]==NN_variables[i[1]].unique()[fixed_points[fixed_variable.index(i)][1]])].drop(i,axis=1).reset_index(drop=True) for i in fixed_variable]
    fixed_values_2d = [v[0] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])) + ', ' + v[1] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])) for v in fixed_variable]

    NN_first_moment_conditional_WZ, NN_second_moment_conditional_WZ = calc_moment(NN_variables, ['W','Z'], moment_list)
    NN_first_moment_conditional_WVtilde, NN_second_moment_conditional_WVtilde = calc_moment(NN_variables, ['W','Vtilde'], moment_list)
    NN_first_moment_conditional_ZVtilde, NN_second_moment_conditional_ZVtilde = calc_moment(NN_variables, ['Z','Vtilde'], moment_list)

    NN_first_moment_conditional_W, NN_second_moment_conditional_W = calc_moment(NN_variables, ['W'], moment_list)
    NN_first_moment_conditional_Z, NN_second_moment_conditional_Z = calc_moment(NN_variables, ['Z'], moment_list)
    NN_first_moment_conditional_Vtilde, NN_second_moment_conditional_Vtilde = calc_moment(NN_variables, ['Vtilde'], moment_list)

    NN_first_moments = [NN_first_moment_conditional_WZ, NN_first_moment_conditional_WVtilde, NN_first_moment_conditional_ZVtilde]
    NN_second_moments = [NN_second_moment_conditional_WZ, NN_second_moment_conditional_WVtilde, NN_second_moment_conditional_ZVtilde]

    NN_first_moments_2d = [NN_first_moment_conditional_W, NN_first_moment_conditional_Z, NN_first_moment_conditional_Vtilde]
    NN_second_moments_2d = [NN_second_moment_conditional_W, NN_second_moment_conditional_Z, NN_second_moment_conditional_Vtilde]

    NN_marginal_density = [NN_variables.groupby(['W','Z']).sum().reset_index(drop=False), NN_variables.groupby(['W','Vtilde']).sum().reset_index(drop=False),NN_variables.groupby(['Vtilde','Z']).sum().reset_index(drop=False)]
    NN_marginal_density_2d = [NN_variables.groupby(['W']).sum().reset_index(drop=False), NN_variables.groupby(['Z']).sum().reset_index(drop=False),NN_variables.groupby(['Vtilde']).sum().reset_index(drop=False)]

else:

  NN_variables_inner = NN_variables[(NN_variables['W'].isin(W_inner_points)) & (NN_variables['Z'].isin(Z_inner_points))& (NN_variables['V'].isin(V_inner_points))& (NN_variables['Vtilde'].isin(Vtilde_inner_points))]
  NN_variables_norm_inner = NN_variables_norm[(NN_variables_norm['W'].isin(W_inner_points)) & (NN_variables_norm['Z'].isin(Z_inner_points))& (NN_variables_norm['V'].isin(V_inner_points))& (NN_variables_norm['Vtilde'].isin(Vtilde_inner_points))]

  fixed_variable = [['W','Z'],['W','V'],['W','Vtilde'],['Z','V'],['Z','Vtilde'],['V','Vtilde']]
  fixed_points = [[fix_dict[i] for i in j] for j in fixed_variable]
  NN_slice = [NN_variables.loc[(NN_variables[v[0]] == NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])&(NN_variables[v[1]] == NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
  NN_slice_inner = [NN_variables_inner.loc[(NN_variables_inner[v[0]] == NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])&(NN_variables_inner[v[1]] == NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
  fixed_values = [v[0] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])) + ', ' + v[1] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])) for v in fixed_variable]

  fixed_variable = [['W','Z','V'],['W','Z','Vtilde'],['W','V','Vtilde'],['Z','V','Vtilde']]
  fixed_points = [[fix_dict[i] for i in j] for j in fixed_variable]
  NN_slice_2d = [NN_variables.loc[(NN_variables[v[0]] == NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])&(NN_variables[v[1]] == NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])&(NN_variables[v[2]] == NN_variables[v[2]].unique()[fixed_points[fixed_variable.index(v)][2]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
  fixed_values_2d = [v[0] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[0]].unique()[fixed_points[fixed_variable.index(v)][0]])) + ', ' + v[1] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[1]].unique()[fixed_points[fixed_variable.index(v)][1]])) + ', '+ v[2] + ' fixed at ' + str("{0:.4f}".format(NN_variables[v[2]].unique()[fixed_points[fixed_variable.index(v)][2]])) for v in fixed_variable]

  NN_first_moment_conditional_WZ, NN_second_moment_conditional_WZ = calc_moment(NN_variables, ['W','Z'], moment_list)
  NN_first_moment_conditional_WV, NN_second_moment_conditional_WV = calc_moment(NN_variables, ['W','V'], moment_list)
  NN_first_moment_conditional_WVtilde, NN_second_moment_conditional_WVtilde = calc_moment(NN_variables, ['W','Vtilde'], moment_list)
  NN_first_moment_conditional_ZV, NN_second_moment_conditional_ZV = calc_moment(NN_variables, ['Z','V'], moment_list)
  NN_first_moment_conditional_ZVtilde, NN_second_moment_conditional_ZVtilde = calc_moment(NN_variables, ['Z','Vtilde'], moment_list)
  NN_first_moment_conditional_VVtilde, NN_second_moment_conditional_VVtilde = calc_moment(NN_variables, ['V','Vtilde'], moment_list)

  NN_first_moment_conditional_W, NN_second_moment_conditional_W = calc_moment(NN_variables, ['W'], moment_list)
  NN_first_moment_conditional_Z, NN_second_moment_conditional_Z = calc_moment(NN_variables, ['Z'], moment_list)
  NN_first_moment_conditional_V, NN_second_moment_conditional_V = calc_moment(NN_variables, ['V'], moment_list)
  NN_first_moment_conditional_Vtilde, NN_second_moment_conditional_Vtilde = calc_moment(NN_variables, ['Vtilde'], moment_list)

  NN_first_moments = [NN_first_moment_conditional_WZ, NN_first_moment_conditional_WV, NN_first_moment_conditional_WVtilde, NN_first_moment_conditional_ZV, NN_first_moment_conditional_ZVtilde, NN_first_moment_conditional_VVtilde]
  NN_second_moments = [NN_second_moment_conditional_WZ, NN_second_moment_conditional_WV, NN_second_moment_conditional_WVtilde, NN_second_moment_conditional_ZV, NN_second_moment_conditional_ZVtilde, NN_second_moment_conditional_VVtilde]

  NN_first_moments_2d = [NN_first_moment_conditional_W, NN_first_moment_conditional_Z, NN_first_moment_conditional_V, NN_first_moment_conditional_Vtilde]
  NN_second_moments_2d = [NN_second_moment_conditional_W, NN_second_moment_conditional_Z, NN_second_moment_conditional_V, NN_second_moment_conditional_Vtilde]

  NN_marginal_density = [NN_variables.groupby(['W','Z']).sum().reset_index(drop=False), NN_variables.groupby(['W','V']).sum().reset_index(drop=False), NN_variables.groupby(['W','Vtilde']).sum().reset_index(drop=False),NN_variables.groupby(['Z','V']).sum().reset_index(drop=False),NN_variables.groupby(['Z','Vtilde']).sum().reset_index(drop=False),NN_variables.groupby(['V','Vtilde']).sum().reset_index(drop=False)]
  NN_marginal_density_2d = [NN_variables.groupby(['W']).sum().reset_index(drop=False), NN_variables.groupby(['Z']).sum().reset_index(drop=False),NN_variables.groupby(['V']).sum().reset_index(drop=False),NN_variables.groupby(['Vtilde']).sum().reset_index(drop=False)]

if status != '3':

    W_MFR                        = np.genfromtxt(mfrSuite_resdir + '/W.dat')
    Z_MFR                        = np.genfromtxt(mfrSuite_resdir + '/Z.dat')
    V_MFR                        = np.genfromtxt(mfrSuite_resdir + '/V.dat')
    Vtilde_MFR                   = np.genfromtxt(mfrSuite_resdir + '/Vtilde.dat')
    V_MFR = V_MFR.round(decimals=14)
    print(np.sum(W_MFR == W_NN),np.sum(Z_MFR == Z_NN),np.sum(V_MFR == V_NN),np.sum(Vtilde_MFR == Vtilde_NN))

    if status == '0':
        logXiE_MFR                  = np.genfromtxt(mfrSuite_resdir + '/xi_e_final.dat').reshape(-1,1)
        logXiH_MFR                  = np.genfromtxt(mfrSuite_resdir + '/xi_h_final.dat').reshape(-1,1)
        XiE_MFR                     = np.exp(logXiE_MFR)
        XiH_MFR                     = np.exp(logXiH_MFR)
        kappa_MFR                   = np.genfromtxt(mfrSuite_resdir + '/kappa_final.dat').reshape(-1,1)
        q_MFR                       = np.genfromtxt(mfrSuite_resdir + '/q_final.dat').reshape(-1,1)

        sigmaK_MFR                  = np.genfromtxt(mfrSuite_resdir + '/sigmaK_final.dat').reshape(3,-1).T        
        sigmaZ_MFR                  = np.genfromtxt(mfrSuite_resdir + '/sigmaZ_final.dat').reshape(3,-1).T
        muK_MFR                     = np.genfromtxt(mfrSuite_resdir + '/muK_final.dat').reshape(-1,1)
        muZ_MFR                     = np.genfromtxt(mfrSuite_resdir + '/muZ_final.dat').reshape(-1,1)
        chi_MFR                     = np.genfromtxt(mfrSuite_resdir + '/chi_final.dat').reshape(-1,1)

        if sigma_Vtilde_norm == 0:
          muV_MFR                     = np.genfromtxt(mfrSuite_resdir + '/muV_final.dat').reshape(-1,1)
          sigmaV_MFR                  = np.genfromtxt(mfrSuite_resdir + '/sigmaV_final.dat').reshape(3,-1).T
        elif sigma_V_norm == 0:
          muVtilde_MFR                = np.genfromtxt(mfrSuite_resdir + '/muH_final.dat').reshape(-1,1)
          sigmaVtilde_MFR             = np.genfromtxt(mfrSuite_resdir + '/sigmaH_final.dat').reshape(3,-1).T

        sigmaQ_MFR                  = np.genfromtxt(mfrSuite_resdir + '/sigmaQ_final.dat').reshape(3,-1).T
        sigmaW_MFR                  = np.genfromtxt(mfrSuite_resdir + '/sigmaw_final.dat').reshape(3,-1).T
        sigmaR_MFR                  = np.genfromtxt(mfrSuite_resdir + '/sigmaR_final.dat').reshape(3,-1).transpose()
        betaE_MFR                   = np.genfromtxt(mfrSuite_resdir + '/betaE_final.dat').reshape(-1,1)
        betaH_MFR                   = np.genfromtxt(mfrSuite_resdir + '/betaH_final.dat').reshape(-1,1)
        deltaE_MFR                  = np.genfromtxt(mfrSuite_resdir + '/deltaE_final.dat').reshape(-1,1)
        deltaH_MFR                  = np.genfromtxt(mfrSuite_resdir + '/deltaH_final.dat').reshape(-1,1)
        PiH_MFR                     = np.genfromtxt(mfrSuite_resdir + '/PiH_final.dat').reshape(3,-1).transpose()
        PiE_MFR                     = np.genfromtxt(mfrSuite_resdir + '/PiE_final.dat').reshape(3,-1).transpose()
        muW_MFR                     = np.genfromtxt(mfrSuite_resdir + '/muw_final.dat').reshape(-1,1)
        muQ_MFR                     = np.genfromtxt(mfrSuite_resdir + '/muQ_final.dat').reshape(-1,1)
        r_MFR                       = np.genfromtxt(mfrSuite_resdir + '/r_final.dat').reshape(-1,1)        
        dent_MFR                    = np.genfromtxt(mfrSuite_resdir + '/dent.txt').reshape(-1,1)
        
        Fe_MFR                      = np.genfromtxt(mfrSuite_resdir + '/Fe_final.dat').reshape(-1,1)
        Fh_MFR                      = np.genfromtxt(mfrSuite_resdir + '/Fh_final.dat').reshape(-1,1)
        firstCoefsE_MFR             = np.genfromtxt(mfrSuite_resdir + '/firstCoefsE_final.dat').reshape(3,-1).transpose()
        firstCoefsH_MFR             = np.genfromtxt(mfrSuite_resdir + '/firstCoefsH_final.dat').reshape(3,-1).transpose()
        secondCoefsE_MFR            = np.genfromtxt(mfrSuite_resdir + '/secondCoefsE_final.dat').reshape(3,-1).transpose()
        secondCoefsH_MFR            = np.genfromtxt(mfrSuite_resdir + '/secondCoefsH_final.dat').reshape(3,-1).transpose()
                
        MFR_variables = pd.DataFrame([W_MFR,Z_MFR,V_MFR,Vtilde_MFR,logXiE_MFR,logXiH_MFR,XiE_MFR,XiH_MFR,kappa_MFR,q_MFR[:,0],chi_MFR[:,0], sigmaR_MFR[:,0],sigmaR_MFR[:,1],sigmaR_MFR[:,2],PiH_MFR[:,0],PiH_MFR[:,1],PiH_MFR[:,2],PiE_MFR[:,0],PiE_MFR[:,1],PiE_MFR[:,2],r_MFR[:,0],dent_MFR],\
                            index = moment_list).T
        MFR_variables = MFR_variables.astype(np.float64)
        MFR_variables_norm = pd.DataFrame([np.zeros(W_MFR.shape),np.zeros(W_MFR.shape),np.zeros(W_MFR.shape)],index = ['HJB_E','HJB_H','kappa_min']).T
        MFR_variables_norm = pd.concat([MFR_variables,MFR_variables_norm],axis=1)
        
        if sigma_Vtilde_norm == 0:
            MFR_variables_inner = MFR_variables[(MFR_variables['W'].isin(W_inner_points)) & (MFR_variables['Z'].isin(Z_inner_points))& (MFR_variables['V'].isin(V_inner_points))]
            MFR_variables_norm_inner = MFR_variables_norm[(MFR_variables_norm['W'].isin(W_inner_points)) & (MFR_variables_norm['Z'].isin(Z_inner_points))& (MFR_variables_norm['V'].isin(V_inner_points))]
            fixed_variable = ['V','Z','W']
            fixed_points = [fix_dict[i] for i in fixed_variable]
            MFR_slice = [MFR_variables.loc[(MFR_variables[v] == MFR_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
            MFR_slice_inner = [MFR_variables_inner.loc[(MFR_variables_inner[v] == MFR_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
            
            fixed_variable = [['W','Z'],['W','V'],['Z','V']]
            fixed_points = [[fix_dict[i] for i in j] for j in fixed_variable]
            MFR_slice_2d = [MFR_variables.loc[(MFR_variables[i[0]]==MFR_variables[i[0]].unique()[fixed_points[fixed_variable.index(i)[0]]]) &(MFR_variables[i[1]]==MFR_variables[i[1]].unique()[fixed_points[fixed_variable.index(i)[1]]])].drop(i,axis=1).reset_index(drop=True) for i in fixed_variable]

            MFR_first_moment_conditional_WZ, MFR_second_moment_conditional_WZ = calc_moment(MFR_variables, ['W','Z'], moment_list)
            MFR_first_moment_conditional_WV, MFR_second_moment_conditional_WV = calc_moment(MFR_variables, ['W','V'], moment_list)
            MFR_first_moment_conditional_ZV, MFR_second_moment_conditional_ZV = calc_moment(MFR_variables, ['Z','V'], moment_list)

            MFR_first_moment_conditional_W, MFR_second_moment_conditional_W = calc_moment(MFR_variables, ['W'], moment_list)
            MFR_first_moment_conditional_Z, MFR_second_moment_conditional_Z = calc_moment(MFR_variables, ['Z'], moment_list)
            MFR_first_moment_conditional_V, MFR_second_moment_conditional_V = calc_moment(MFR_variables, ['V'], moment_list)

            MFR_first_moments = [MFR_first_moment_conditional_WZ, MFR_first_moment_conditional_WV, MFR_first_moment_conditional_ZV]
            MFR_second_moments = [MFR_second_moment_conditional_WZ, MFR_second_moment_conditional_WV, MFR_second_moment_conditional_ZV]

            MFR_first_moments_2d = [MFR_first_moment_conditional_W, MFR_first_moment_conditional_Z, MFR_first_moment_conditional_V]
            MFR_second_moments_2d = [MFR_second_moment_conditional_W, MFR_second_moment_conditional_Z, MFR_second_moment_conditional_V]

            MFR_marginal_density = [MFR_variables.groupby(['W','Z']).sum().reset_index(drop=False), MFR_variables.groupby(['W','V']).sum().reset_index(drop=False),MFR_variables.groupby(['V','Z']).sum().reset_index(drop=False)]
            MFR_marginal_density_2d = [MFR_variables.groupby(['W']).sum().reset_index(drop=False), MFR_variables.groupby(['Z']).sum().reset_index(drop=False),MFR_variables.groupby(['V']).sum().reset_index(drop=False)]

        elif sigma_V_norm == 0:
            new_columns = MFR_variables.columns.tolist()
            MFR_variables = MFR_variables[new_columns[:2] + [new_columns[3]]+ [new_columns[2]] +new_columns[4:]]
            MFR_variables_inner = MFR_variables[(MFR_variables['W'].isin(W_inner_points)) & (MFR_variables['Z'].isin(Z_inner_points))& (MFR_variables['Vtilde'].isin(Vtilde_inner_points))]
            MFR_variables_norm_inner = MFR_variables_norm[(MFR_variables['W'].isin(W_inner_points)) & (MFR_variables_norm['Z'].isin(Z_inner_points))& (MFR_variables_norm['Vtilde'].isin(Vtilde_inner_points))]

            fixed_variable = ['Vtilde','Z','W']
            fixed_points = [fix_dict[i] for i in fixed_variable]
            MFR_slice = [MFR_variables.loc[(MFR_variables[v] == MFR_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
            MFR_slice_inner = [MFR_variables_inner.loc[(MFR_variables_inner[v] == MFR_variables[v].unique()[fixed_points[fixed_variable.index(v)]])].drop(v,axis=1).reset_index(drop=True) for v in fixed_variable]
            
            fixed_variable = [['W','Z'],['W','Vtilde'],['Z','Vtilde']]
            fixed_points = [[fix_dict[i] for i in j] for j in fixed_variable]
            MFR_slice_2d = [MFR_variables.loc[(MFR_variables[i[0]]==MFR_variables[i[0]].unique()[fixed_points[fixed_variable.index(i)][0]]) &(MFR_variables[i[1]]==MFR_variables[i[1]].unique()[fixed_points[fixed_variable.index(i)][1]])].drop(i,axis=1).reset_index(drop=True) for i in fixed_variable]

            MFR_first_moment_conditional_WZ, MFR_second_moment_conditional_WZ = calc_moment(MFR_variables, ['W','Z'], moment_list)
            MFR_first_moment_conditional_WVtilde, MFR_second_moment_conditional_WVtilde = calc_moment(MFR_variables, ['W','Vtilde'], moment_list)
            MFR_first_moment_conditional_ZVtilde, MFR_second_moment_conditional_ZVtilde = calc_moment(MFR_variables, ['Z','Vtilde'], moment_list)

            MFR_first_moment_conditional_W, MFR_second_moment_conditional_W = calc_moment(MFR_variables, ['W'], moment_list)
            MFR_first_moment_conditional_Z, MFR_second_moment_conditional_Z = calc_moment(MFR_variables, ['Z'], moment_list)
            MFR_first_moment_conditional_Vtilde, MFR_second_moment_conditional_Vtilde = calc_moment(MFR_variables, ['Vtilde'], moment_list)

            MFR_first_moments = [MFR_first_moment_conditional_WZ, MFR_first_moment_conditional_WVtilde, MFR_first_moment_conditional_ZVtilde]
            MFR_second_moments = [MFR_second_moment_conditional_WZ, MFR_second_moment_conditional_WVtilde, MFR_second_moment_conditional_ZVtilde]

            MFR_first_moments_2d = [MFR_first_moment_conditional_W, MFR_first_moment_conditional_Z, MFR_first_moment_conditional_Vtilde]
            MFR_second_moments_2d = [MFR_second_moment_conditional_W, MFR_second_moment_conditional_Z, MFR_second_moment_conditional_Vtilde]
             
            MFR_marginal_density = [MFR_variables.groupby(['W','Z']).sum().reset_index(drop=False), MFR_variables.groupby(['W','Vtilde']).sum().reset_index(drop=False),MFR_variables.groupby(['Vtilde','Z']).sum().reset_index(drop=False)]
            MFR_marginal_density_2d = [MFR_variables.groupby(['W']).sum().reset_index(drop=False), MFR_variables.groupby(['Z']).sum().reset_index(drop=False),MFR_variables.groupby(['Vtilde']).sum().reset_index(drop=False)]

    else:
        MFR_slice = []; MFR_slice_inner = []; MFR_first_moments = []; MFR_second_moments = []; MFR_slice_2d = []; MFR_first_moments_2d = []; MFR_second_moments_2d = []; MFR_marginal_density = []; MFR_marginal_density_2d = []
        MFR_variables_norm = NN_variables_norm.copy()
        MFR_variables_norm_inner = NN_variables_norm_inner.copy()
        MFR_variables_norm['HJB_E'] = 0.0
        MFR_variables_norm['HJB_H'] = 0.0
        MFR_variables_norm_inner['HJB_E'] = 0.0
        MFR_variables_norm_inner['HJB_H'] = 0.0
else:
    MFR_slice = []; MFR_slice_inner = []; MFR_first_moments = []; MFR_second_moments = []; MFR_slice_2d = []; MFR_first_moments_2d = []; MFR_second_moments_2d = []; MFR_marginal_density = []; MFR_marginal_density_2d = []
    MFR_variables_norm = NN_variables_norm.copy()
    MFR_variables_norm_inner = NN_variables_norm_inner.copy()
    MFR_variables_norm['HJB_E'] = 0.0
    MFR_variables_norm['HJB_H'] = 0.0
    MFR_variables_norm_inner['HJB_E'] = 0.0
    MFR_variables_norm_inner['HJB_H'] = 0.0

plot_results_slice = [MFR_slice, NN_slice]
plot_results_slice_inner = [MFR_slice_inner, NN_slice_inner]
plot_results_first_moments = [MFR_first_moments, NN_first_moments]
plot_results_second_moments = [MFR_second_moments, NN_second_moments]
plot_results_slice_2d = [MFR_slice_2d, NN_slice_2d]
plot_results_first_moments_2d = [MFR_first_moments_2d, NN_first_moments_2d]
plot_results_second_moments_2d = [MFR_second_moments_2d, NN_second_moments_2d]
plot_results_density = [MFR_marginal_density, NN_marginal_density]
plot_results_density_2d = [MFR_marginal_density_2d, NN_marginal_density_2d]

var_names = [['logXiE'], ['logXiH'], ['XiE'], ['XiH'], ['kappa'], ['q'], ['chi'], ['sigmaR_first_shock', 'sigmaR_second_shock', 'sigmaR_third_shock'], ['PiH_first_shock', 'PiH_second_shock', 'PiH_third_shock'], ['PiE_first_shock', 'PiE_second_shock', 'PiE_third_shock'], ['r']]
plot_contents = ['Log Experts Value Function',  'Log Households Value Function', 'Experts Value Function', 'Households Value Function', 'Kappa Policy Function', 'Capital Price', 'Chi Policy Function', 'Local Capital Return Volatility', 'Households Risk Price', 'Experts Risk Price', 'Short Term Interest Rate']

# for i in range(len(plot_contents)):
#     var_name = var_names[i]
#     height = 1200 if len(var_name) > 1 else 700
#     width_3d = 3500 if nDims == 4 else 1500
#     width_2d = 2300 if nDims == 4 else 1500
#     spacing = 0.05 if nDims == 4 else 0.1

#     plot_content = plot_contents[i]
#     generateMomentPlots(status, plot_results_slice, var_name, plot_content, parameter_list, fixed = fixed_values, fix_dict = fix_dict, z_adjust = True, height=height, width=width_3d, spacing = spacing, path = docdir)
#     plot_content = plot_contents[i] + ' Interior'
#     generateMomentPlots(status, plot_results_slice_inner, var_name, plot_content, parameter_list, fixed = fixed_values, fix_dict = fix_dict, z_adjust = True, height=height, width=width_3d, spacing = spacing, path = docdir)    
#     # plot_content = plot_contents[i] + ' 2d'
#     # generateMomentPlots_2d(status, plot_results_slice_2d, var_name, plot_content, parameter_list, fixed = fixed_values_2d, fix_dict = fix_dict, y_adjust = True, height=height, width=width_2d, spacing = spacing, path = docdir)
#     plot_content = 'Conditional Expectation of ' + plot_contents[i]
#     generateMomentPlots(status, plot_results_first_moments, var_name, plot_content, parameter_list, z_adjust = True, height=height, width=width_3d, spacing = spacing, path = docdir)
#     # plot_content = 'Conditional Variance of ' + plot_contents[i]
#     # generateMomentPlots(status, plot_results_second_moments, var_name, plot_content, parameter_list, z_adjust = True, height=height, width=width_3d, spacing = spacing, path = docdir)
#     plot_content = 'Conditional Expectation of ' + plot_contents[i] + ' 2d'
#     generateMomentPlots_2d(status, plot_results_first_moments_2d, var_name, plot_content, parameter_list, y_adjust = True, height=height, width=width_2d, spacing = spacing, path = docdir)
#     # plot_content = 'Conditional Variance of ' + plot_contents[i] + ' 2d'
#     # generateMomentPlots_2d(status, plot_results_second_moments_2d, var_name, plot_content, parameter_list, y_adjust = True, height=height, width=width_2d, spacing = spacing, path = docdir)

# var_name = ['dent']
# plot_content = 'Stationary Densities'
# generateMomentPlots(status, plot_results_slice, var_name, plot_content, parameter_list, fixed = fixed_values, fix_dict = fix_dict, z_adjust = True, height=700, width=width_3d, spacing = spacing, path = docdir)
# plot_content = 'Stationary Densities' + ' 2d'
# generateMomentPlots_2d(status, plot_results_slice_2d, var_name, plot_content, parameter_list, fixed = fixed_values_2d, fix_dict = fix_dict, y_adjust = True, height=700, width=width_2d, spacing = spacing, path = docdir)
# plot_content = 'Marginal Stationary Densities'
# generateMomentPlots(status, plot_results_density, var_name, plot_content, parameter_list, z_adjust = True, height=700, width=width_3d, spacing = spacing, path = docdir)
# plot_content = 'Marginal Stationary Densities'+ ' 2d'
# generateMomentPlots_2d(status, plot_results_density_2d, var_name, plot_content, parameter_list, y_adjust = True, height=700, width=width_3d, spacing = spacing, path = docdir)


two_norm = []
sup_norm = []
two_norm_inner = []
sup_norm_inner = []
var_num = NN_variables_norm.shape[1]
for i in range(var_num):
  two_norm.append(np.linalg.norm(NN_variables_norm.iloc[:,i]-MFR_variables_norm.iloc[:,i]))
  sup_norm.append(np.linalg.norm(NN_variables_norm.iloc[:,i]-MFR_variables_norm.iloc[:,i], np.inf))
  two_norm_inner.append(np.linalg.norm(NN_variables_norm_inner.iloc[:,i]-MFR_variables_norm_inner.iloc[:,i]))
  sup_norm_inner.append(np.linalg.norm(NN_variables_norm_inner.iloc[:,i]-MFR_variables_norm_inner.iloc[:,i], np.inf))

norm = pd.DataFrame([two_norm,sup_norm,two_norm_inner,sup_norm_inner],columns = NN_variables_norm.columns,index = ['Two norm','Sup Norm','Interior Two Norm','Interior Sup Norm']).T
norm.to_csv(docdir + 'norm.csv')

varibles_list = load_list.copy()
[varibles_list.pop(varibles_list.index(i)) for i in ['HJB_E_NN','HJB_H_NN','kappa_min_NN']]
varibles_list = [i[:-3] for i in varibles_list]

if sigma_Vtilde_norm == 0:
  varibles_list.pop(varibles_list.index('muVtilde'))
  varibles_list.pop(varibles_list.index('sigmaVtilde'))
elif sigma_V_norm == 0:
  varibles_list.pop(varibles_list.index('muV'))
  varibles_list.pop(varibles_list.index('sigmaV'))

  # norms = []
  # for variable in varibles_list:
  #   if variable == 'sigmaK':
  #     norms.append(np.linalg.norm(eval(variable+'_MFR')[:,0] - eval(variable+'_NN')[:,0]))
  #   elif variable == 'sigmaZ':
  #     norms.append(np.linalg.norm(eval(variable+'_MFR')[:,1] - eval(variable+'_NN')[:,1]))
  #   elif variable == 'sigmaV':
  #     norms.append(np.linalg.norm(eval(variable+'_MFR')[:,2] - eval(variable+'_NN')[:,2]))
  #   elif variable == 'sigmaVtilde':
  #     norms.append(np.linalg.norm(eval(variable+'_MFR')[:,2] - eval(variable+'_NN')[:,2]))
  #   elif variable in ['sigmaQ','sigmaR','sigmaW','PiH','PiE']:
  #     norms.append(np.linalg.norm(eval(variable+'_MFR') - eval(variable+'_NN')[:,0:3]))
  #   elif variable in ['firstCoefsE','secondCoefsE','firstCoefsH','secondCoefsH']:
  #     norms.append(np.linalg.norm(eval(variable+'_MFR').sum(axis=1) - eval(variable+'_NN')[:,0]))
  #   else:
  #     norms.append(np.linalg.norm(eval(variable+'_MFR').reshape([-1,1])- eval(variable+'_NN').reshape([-1,1])))
  
  # norms.append(np.linalg.norm(HJB_E_NN-np.zeros(HJB_E_NN.shape)))
  # norms.append(np.linalg.norm(HJB_H_NN-np.zeros(HJB_H_NN.shape)))
  # norms.append(np.linalg.norm(kappa_min_NN-np.zeros(kappa_min_NN.shape)))
  # [varibles_list.append(i) for i in ['HJB_E','HJB_H','kappa_min']]
  # norms = pd.DataFrame([varibles_list, norms]).T
  # norms.columns = ['Variables','Norms']
  # norms.set_index('Variables')
  # norms.to_csv(docdir + 'norms.csv')