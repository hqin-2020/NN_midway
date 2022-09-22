import tensorflow as tf 
import mfr.modelSoln as m
import json
import os

def setModelParameters(parameter_list, domain_list, nDims):

  chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = parameter_list 
  nWealth, nZ, nV, nVtilde, V_bar, Vtilde_bar, sigma_K_norm, sigma_Z_norm, sigma_V_norm, sigma_Vtilde_norm = domain_list 

  params = m.paramsDefault.copy()

  ## Dimensionality params
  params['nDims'] = nDims;      params['nShocks'] = nDims;

  ## Grid parameters 
  params['nWealth'] = nWealth;  params['nZ']  = nZ; params['nV']  = nV; params['nVtilde'] = nVtilde;

  ## Domain params
  params['wMin']              = 0.01
  params['wMax']              = 0.99
  params['Z_bar']             = 0.0
  params['V_bar']             = V_bar
  params['Vtilde_bar']        = Vtilde_bar
  params['sigma_K_norm']      = sigma_K_norm
  params['sigma_Z_norm']      = sigma_Z_norm
  params['sigma_V_norm']      = sigma_V_norm
  params['sigma_Vtilde_norm'] = sigma_Vtilde_norm

  ## Economic params
  params['chiUnderline']      = chiUnderline
  params['a_e']               = a_e
  params['a_h']               = a_h
  params['gamma_e']           = gamma_e
  params['gamma_h']           = gamma_h
  params['rho_e']             = psi_e
  params['rho_h']             = psi_h

  ## Shock correlation params
  params['cov11'] = 1.0;    params['cov12'] = 0.0;    params['cov13'] = 0.0;    params['cov14'] = 0.0;
  params['cov21'] = 0.0;    params['cov22'] = 1.0;    params['cov23'] = 0.0;    params['cov24'] = 0.0;
  params['cov31'] = 0.0;    params['cov32'] = 0.0;    params['cov33'] = 1.0;    params['cov34'] = 0.0;
  params['cov41'] = 0.0;    params['cov42'] = 0.0;    params['cov43'] = 0.0;    params['cov44'] = 1.0 if nDims > 3 else 0.0;

  sigma_K, sigma_Z, sigma_V, sigma_Vtilde = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in [sigma_K_norm, sigma_Z_norm, sigma_V_norm, sigma_Vtilde_norm]]
  if params['sigma_Vtilde_norm'] == 0:
    domain_folder = 'WZV' + '_sigma_K_' + sigma_K + '_sigma_Z_' + sigma_Z + '_sigma_V_' + sigma_V + '_sigma_Vtilde_' + sigma_Vtilde
  elif params['sigma_V_norm'] == 0:
    domain_folder = 'WZVtilde' + '_sigma_K_' + sigma_K + '_sigma_Z_' + sigma_Z + '_sigma_V_' + sigma_V + '_sigma_Vtilde_' + sigma_Vtilde
  else:
    domain_folder = 'WZVVtilde' + '_sigma_K_' + sigma_K + '_sigma_Z_' + sigma_Z + '_sigma_V_' + sigma_V + '_sigma_Vtilde_' + sigma_Vtilde

  chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = [str("{:0.3f}".format(param)).replace('.', '', 1)  for param in parameter_list]
  model_folder = 'chiUnderline_' + chiUnderline + '_a_e_' + a_e + '_a_h_' + a_h  + '_gamma_e_' + gamma_e + '_gamma_h_' + gamma_h + '_psi_e_' + psi_e + '_psi_h_' + psi_h

  params['folderName']        = model_folder

  workdir = os.path.dirname(os.getcwd())
  datadir = workdir + '/data/' + domain_folder + '/' + model_folder + '/'
  with open(datadir + 'parameters_NN.json', 'w') as f:
      json.dump(params,f)

def setModelParametersFromFile(paramsFromFile, nDims, chi_position_tolerance, chi_value_tolerance, chi_max_iterations):

  params = {}
  ####### Model parameters #######
  NN_param_list = ['chiUnderline','a_e','a_h','gamma_e','gamma_h','psi_e','psi_h',\
                   'V_bar','Z_bar','Vtilde_bar','sigma_K_norm','sigma_Z_norm','sigma_V_norm','sigma_Vtilde_norm','lambda_d','lambda_Z','lambda_V','lambda_Vtilde',\
                   'nu_newborn','phi','rho_e','rho_h','equityIss','delta', 'numSds',\
                   'cov11','cov12','cov13','cov14','cov21','cov22','cov23','cov24','cov31','cov32','cov33','cov34','cov41','cov42','cov43','cov44']
  MFR_param_list = NN_param_list.copy()
  MFR_param_list[NN_param_list.index('psi_e')] = 'rho_e';     MFR_param_list[NN_param_list.index('psi_h')] = 'rho_h';
  MFR_param_list[NN_param_list.index('rho_e')] = 'delta_e';   MFR_param_list[NN_param_list.index('rho_h')] = 'delta_h';
  MFR_param_list[NN_param_list.index('delta')] = 'alpha_K'
  
  for i in range(len(NN_param_list)):
    params[NN_param_list[i]]             = tf.constant(paramsFromFile[MFR_param_list[i]], dtype=tf.float64)

  ########### Derived parameters
  ## Covariance matrices 
  params['sigmaK']                 = tf.concat([params['cov11'] * params['sigma_K_norm'],       params['cov12'] * params['sigma_K_norm'],       params['cov13'] * params['sigma_K_norm'],       params['cov14'] * params['sigma_K_norm']], 0)
  params['sigmaZ']                 = tf.concat([params['cov21'] * params['sigma_Z_norm'],       params['cov22'] * params['sigma_Z_norm'],       params['cov23'] * params['sigma_Z_norm'],       params['cov24'] * params['sigma_Z_norm']], 0)
  params['sigmaV']                 = tf.concat([params['cov31'] * params['sigma_V_norm'],       params['cov32'] * params['sigma_V_norm'],       params['cov33'] * params['sigma_V_norm'],       params['cov34'] * params['sigma_V_norm']], 0)
  params['sigmaVtilde']            = tf.concat([params['cov41'] * params['sigma_Vtilde_norm'],  params['cov42'] * params['sigma_Vtilde_norm'],  params['cov43'] * params['sigma_Vtilde_norm'],  params['cov44'] * params['sigma_Vtilde_norm']], 0) if nDims > 3 else\
                                     tf.concat([params['cov31'] * params['sigma_Vtilde_norm'],  params['cov32'] * params['sigma_Vtilde_norm'],  params['cov33'] * params['sigma_Vtilde_norm'],  params['cov34'] * params['sigma_Vtilde_norm']], 0) 

  ## Min and max of state variables
  ## min/max for W
  params['wMin'] = tf.constant(0.01, dtype=tf.float64)
  params['wMax'] = tf.constant(1 - params['wMin'], dtype=tf.float64)
  
  ## min/max for Z
  zVar  = tf.pow(params['V_bar'] * params['sigma_Z_norm'], 2) / (2 * params['lambda_Z'])
  params['zMin'] = params['Z_bar'] - params['numSds'] * tf.sqrt(zVar)
  params['zMax'] = params['Z_bar'] + params['numSds'] * tf.sqrt(zVar)

  ## min/max for V
  if params['sigma_V_norm'] == 0:
    params['vMin'] = params['V_bar']
    params['vMax'] = params['V_bar']
  else:
    shape = 2 * params['lambda_V'] * params['V_bar']  /  (tf.pow(params['sigma_V_norm'],2));
    rate = 2 * params['lambda_V'] / (tf.pow(params['sigma_V_norm'],2));
    params['vMin'] = tf.constant(0.00001, dtype=tf.float64)
    params['vMax'] = params['V_bar'] + params['numSds'] * tf.sqrt( shape / tf.pow(rate, 2));
  
  ## min/max for Vtilde
  if params['sigma_Vtilde_norm'] == 0:
    params['VtildeMin'] = params['Vtilde_bar']
    params['VtildeMax'] = params['Vtilde_bar']
  else:
    vtildeVar  = tf.pow(params['Vtilde_bar'] * params['sigma_Vtilde_norm'], 2) / (2 * params['lambda_Vtilde'])
    params['VtildeMin'] = tf.constant(0.00001, dtype=tf.float64)
    params['VtildeMax'] = params['Vtilde_bar'] + params['numSds'] * tf.sqrt(vtildeVar)

  params['chi_position_tolerance']  = chi_position_tolerance
  params['chi_value_tolerance']     = chi_value_tolerance
  params['chi_max_iterations']      = chi_max_iterations
  
  return params