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

## Generate parameter set
setModelParameters(parameter_list, domain_list, nDims)
with open(datadir + 'parameters_NN.json') as json_file:
    paramsFromFile = json.load(json_file)
params = setModelParametersFromFile(paramsFromFile, nDims, chi_ptol, chi_vtol, chi_max_iter)

batchSize = 2048 * points_size
dimension = 4
units = 16
activation = 'tanh'
kernel_initializer = 'glorot_normal'

BFGS_maxiter  = 50
BFGS_maxfun   = 50000
BFGS_gtol     = 1.0 * np.finfo(float).eps
BFGS_maxcor   = 100
BFGS_maxls    = 100
BFGS_ftol     = 1.0 * np.finfo(float).eps

## NN structure
tf.keras.backend.set_floatx("float64") ## Use float64 by default

logXiE_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation= None,  kernel_initializer='glorot_normal')])

logXiH_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation= None , kernel_initializer='glorot_normal')])

kappa_NN = tf.keras.Sequential(
    [tf.keras.Input(shape=[dimension,]),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),
      tf.keras.layers.Dense(1,  activation='sigmoid', kernel_initializer='glorot_normal')])

## Training
start = time.time()
targets = tf.zeros(shape=(batchSize,1), dtype=tf.float64)
for iter in range(iter_num):
  W = tf.random.uniform(shape = (batchSize,1), minval = params['wMin'], maxval = params['wMax'], dtype=tf.float64)
  Z = tf.random.uniform(shape = (batchSize,1), minval = params['zMin'], maxval = params['zMax'], dtype=tf.float64)
  V = tf.random.uniform(shape = (batchSize,1), minval = params['vMin'], maxval = params['vMax'], dtype=tf.float64)
  Vtilde = tf.random.uniform(shape = (batchSize,1), minval = params['VtildeMin'], maxval = params['VtildeMax'], dtype=tf.float64)
  print('Iteration', iter)
  training_step_BFGS(logXiH_NN, logXiE_NN, kappa_NN, W, Z, V, Vtilde, params, targets,\
                    weight1 = w1, boundary1 = b1, weight2 = w2, boundary2 = b2,\
                    maxiter = BFGS_maxiter, maxfun = BFGS_maxfun, gtol = BFGS_gtol, maxcor = BFGS_maxcor, maxls = BFGS_maxls, ftol = BFGS_ftol)
end = time.time()
training_time = '{:.4f}'.format((end - start)/60)
print('Elapsed time for training {:.4f} sec'.format(end - start))

## Save trained neural network approximations and respective model parameters
tf.saved_model.save(logXiH_NN, outputdir   + 'logXiH_NN')
tf.saved_model.save(logXiE_NN, outputdir   + 'logXiE_NN')
tf.saved_model.save(kappa_NN,  outputdir   + 'kappa_NN')

NN_info = {'XiE_layers': XiE_layers, 'XiH_layers': XiH_layers, 'kappa_layers': kappa_layers, 'weight1': w1, 'boundary1': b1, 'weight2': w2, 'boundary2': b2,  'points_size': points_size,\
          'dimension': dimension, 'units': units, 'activation': activation, 'kernel_initializer': kernel_initializer, 'iter_num': iter_num, 'training_time': training_time,\
           'chi_position_tolerance':chi_ptol, 'chi_value_tolerance':chi_vtol, 'chi_max_iterations':chi_max_iter, 'batchSize':batchSize,\
           'BFGS_maxiter':BFGS_maxiter, 'BFGS_maxfun':BFGS_maxfun, 'BFGS_gtol':BFGS_gtol, 'BFGS_maxcor':BFGS_maxcor, 'BFGS_maxls':BFGS_maxls, 'BFGS_ftol':BFGS_ftol}

with open(outputdir + "/NN_info.json", "w") as f:
  json.dump(NN_info,f)