import argparse
import os 
srcdir = os.getcwd()

parser = argparse.ArgumentParser(description="parameter settings")
parser.add_argument("--XiE_layers",type=int,default=5)
parser.add_argument("--XiH_layers",type=int,default=5)
parser.add_argument("--kappa_layers",type=int,default=5)
args = parser.parse_args()

with open(srcdir + '/BFGS.py', 'r') as f:
    lines = f.readlines()

XiE_layers = args.XiE_layers
XiH_layers = args.XiH_layers
kappa_NN_layers = args.kappa_layers
layer = "      tf.keras.layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer),\n"
i = lines.index('      ####### XiE structure #######\n') 
lines[i:i+1] = [layer for i in range(XiE_layers)]
i = lines.index('      ####### XiH structure #######\n') 
lines[i:i+1] = [layer for i in range(XiH_layers)]
i = lines.index('      ####### kappa structure #######\n') 
lines[i:i+1] = [layer for i in range(kappa_NN_layers)]

# with open(srcdir + '/standard_BFGS.py', 'w') as f:
#     f.writelines(lines)