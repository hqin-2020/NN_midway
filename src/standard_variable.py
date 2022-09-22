import json
import tensorflow as tf 
import pandas as pd
import time 
import os
import mfr.sdm as sdm
from standard_para import *
from standard_training import *
import argparse

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

## Load parameter set
with open(datadir + 'parameters_NN.json') as json_file:
    paramsFromFile = json.load(json_file)
params = setModelParametersFromFile(paramsFromFile, nDims, chi_ptol, chi_vtol, chi_max_iter)

## Load trained NNs back from memory
logXiH_NN_tf = tf.saved_model.load(outputdir   + 'logXiH_NN')
logXiE_NN_tf = tf.saved_model.load(outputdir   + 'logXiE_NN')
kappa_NN_tf = tf.saved_model.load(outputdir    + 'kappa_NN')

## Calculate and dump equlibrium variables
if sigma_Vtilde_norm == 0:
  total_points = nWealth * nZ * nV 
  W_NN      = np.tile(np.linspace(params['wMin'],params['wMax'], num = nWealth),int(total_points/nWealth))
  Z_NN      = np.tile(np.repeat(np.linspace(params['zMin'],params['zMax'], num = nZ),nWealth),nV)
  V_NN      = np.repeat(np.linspace(params['vMin'],params['vMax'], num = nV),int(total_points/nV))
  Vtilde_NN = np.repeat(Vtilde_bar,total_points)
elif sigma_V_norm == 0:
  total_points = nWealth * nZ * nVtilde 
  W_NN      = np.tile(np.linspace(params['wMin'],params['wMax'], num = nWealth),int(total_points/nWealth))
  Z_NN      = np.tile(np.repeat(np.linspace(params['zMin'],params['zMax'], num = nZ),nWealth),nVtilde)
  V_NN      = np.repeat(V_bar,total_points)
  Vtilde_NN = np.repeat(np.linspace(params['VtildeMin'],params['VtildeMax'], num = nVtilde),int(total_points/nVtilde))
else:
  total_points = nWealth * nZ * nV * nVtilde
  W_NN      = np.tile(np.linspace(params['wMin'],params['wMax'], num = nWealth),int(total_points/nWealth))
  Z_NN      = np.tile(np.repeat(np.linspace(params['zMin'],params['zMax'], num = nZ),nWealth),nV*nVtilde)
  V_NN      = np.tile(np.repeat(np.linspace(params['vMin'],params['vMax'], num = nV), nWealth*nZ),nVtilde)
  Vtilde_NN = np.repeat(np.linspace(params['VtildeMin'],params['VtildeMax'], num = nVtilde),int(total_points/nVtilde))

X      = np.array([W_NN,Z_NN,V_NN,Vtilde_NN]).transpose()
X_var  = tf.Variable(X, dtype=tf.float64)

## Calculte NN approximations and equlibrium variables
variables = calc_var(logXiH_NN_tf, logXiE_NN_tf, kappa_NN_tf, W_NN.reshape(-1,1), Z_NN.reshape(-1,1), V_NN.reshape(-1,1), Vtilde_NN.reshape(-1,1), params)

logXiE_NN       = variables['logXiE'].numpy();  
logXiH_NN       = variables['logXiH'].numpy();  
XiE_NN          = np.exp(logXiE_NN)
XiH_NN          = np.exp(logXiH_NN)
kappa_NN        = variables['kappa'].numpy()
q_NN            = variables['Q'].numpy();            
sigmaK_NN       = variables['sigmaK'].numpy();    
sigmaZ_NN       = variables['sigmaZ'].numpy();    
sigmaV_NN       = variables['sigmaV'].numpy();      
sigmaVtilde_NN  = variables['sigmaVtilde'].numpy(); 
muK_NN          = variables['muK'].numpy();        
muZ_NN          = variables['muZ'].numpy();          
muV_NN          = variables['muV'].numpy();          
muVtilde_NN     = variables['muVtilde'].numpy();  
chi_NN          = variables['chi'].numpy(); 
sigmaQ_NN       = variables['sigmaQ'].numpy();  
sigmaR_NN       = variables['sigmaR'].numpy();    
sigmaW_NN       = variables['sigmaW'].numpy();    
deltaE_NN       = variables['deltaE'].numpy();     
deltaH_NN       = variables['deltaH'].numpy();
PiH_NN          = variables['PiH'].numpy();        
PiE_NN          = variables['PiE'].numpy();          
betaE_NN        = variables['betaE'].numpy();      
betaH_NN        = variables['betaH'].numpy(); 
muW_NN          = variables['muW'].numpy();        
muQ_NN          = variables['muQ'].numpy();          
muX_NN          = np.matrix(variables['muX']);       
sigmaX_NN       = [np.matrix(el) for el in variables['sigmaX']];  
r_NN            = variables['r'].numpy();                                   

Fe_NN, firstCoefsE_NN, secondCoefsE_NN, HJB_E_NN = calc_HJB_E(W_NN.reshape(-1,1), Z_NN.reshape(-1,1),V_NN.reshape(-1,1), Vtilde_NN.reshape(-1,1), params, variables)
Fh_NN, firstCoefsH_NN, secondCoefsH_NN, HJB_H_NN = calc_HJB_H(W_NN.reshape(-1,1), Z_NN.reshape(-1,1),V_NN.reshape(-1,1), Vtilde_NN.reshape(-1,1), params, variables)
kappa_min_NN = calc_con_kappa(W_NN.reshape(-1,1), Z_NN.reshape(-1,1),V_NN.reshape(-1,1), Vtilde_NN.reshape(-1,1), params, variables)

Fe_NN           = Fe_NN.numpy(); 
firstCoefsE_NN  = firstCoefsE_NN.numpy(); 
secondCoefsE_NN = secondCoefsE_NN.numpy(); 
HJB_E_NN        = HJB_E_NN.numpy();
Fh_NN           = Fh_NN.numpy(); 
firstCoefsH_NN  = firstCoefsH_NN.numpy(); 
secondCoefsH_NN = secondCoefsH_NN.numpy(); 
HJB_H_NN        = HJB_H_NN.numpy();
kappa_min_NN    = kappa_min_NN.numpy()

if sigma_Vtilde_norm == 0:
  muX_NN = np.delete(muX_NN,-1, axis=1)
  sigmaX_NN.pop(-1)
  dent_NN, FKmat_NN, stateGrid = sdm.computeDent(np.matrix(np.array([W_NN,Z_NN,V_NN]).transpose()), {'muX': muX_NN, 'sigmaX': sigmaX_NN})
elif sigma_V_norm == 0:
  muX_NN = np.delete(muX_NN,-2, axis=1)
  sigmaX_NN.pop(-2)
  dent_NN, FKmat_NN, stateGrid = sdm.computeDent(np.matrix(np.array([W_NN,Z_NN,Vtilde_NN]).transpose()), {'muX': muX_NN, 'sigmaX': sigmaX_NN})
else:
  dent_NN, FKmat_NN, stateGrid = sdm.computeDent(np.matrix(np.array([W_NN,Z_NN,V_NN,Vtilde_NN]).transpose()), {'muX': muX_NN, 'sigmaX': sigmaX_NN})

dump_list = ['W_NN','Z_NN','V_NN','Vtilde_NN','logXiE_NN','logXiH_NN','XiE_NN','XiH_NN','kappa_NN','q_NN',\
             'sigmaK_NN', 'sigmaZ_NN','sigmaV_NN','sigmaVtilde_NN','muK_NN','muZ_NN', 'muV_NN','muVtilde_NN','chi_NN',\
             'sigmaW_NN','sigmaQ_NN','sigmaR_NN','deltaE_NN','deltaH_NN','PiH_NN','PiE_NN','betaE_NN','betaH_NN',
             'muW_NN', 'muQ_NN','r_NN','dent_NN',\
             'Fe_NN', 'firstCoefsE_NN', 'secondCoefsE_NN', 'HJB_E_NN','Fh_NN', 'firstCoefsH_NN', 'secondCoefsH_NN', 'HJB_H_NN','kappa_min_NN']
[np.save(outputdir + i, eval(i)) for i in dump_list];