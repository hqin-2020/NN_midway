import numpy as np
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os

def generateMomentPlots(status, plot_results, var_name, plot_content, parameter_list, fixed = None, fix_dict = None, z_adjust = True, float_formatter = "{0:.4f}", height=800, width=1200, spacing = 0.1, path = os.path.dirname(os.getcwd()) + '/doc/'):
    
    plot_row_dims      = len(var_name)
    plot_col_dims      = len(plot_results[1])

    plot_results_name  = ['MFR', "NN"]
    plot_color_style   = ['Viridis', 'Plasma']

    if status != '0':
        plot_results      = [plot_results[1]]
        plot_results_name = [plot_results_name[1]]
        plot_color_style  = [plot_color_style[1]] 

    subplot_titles = []
    subplot_types = []
    for row in range(plot_row_dims):
        subplot_type = []
        for col in range(plot_col_dims):
            if status == '0':
                twoNorm = float_formatter.format(np.linalg.norm(plot_results[1][col][var_name[row]] - plot_results[0][col][var_name[row]]))
                supNorm = float_formatter.format(np.linalg.norm((plot_results[1][col][var_name[row]] - plot_results[0][col][var_name[row]]), np.inf))
                try:
                  supNorm_percentage = float_formatter.format(np.linalg.norm(((plot_results[1][col][var_name[row]] - plot_results[0][col][var_name[row]])/plot_results[0][col][var_name[row]]*100), np.inf))
                except:
                  supNorm_percentage = np.nan
                if fixed ==None:
                    subplot_titles.append('Conditional on '+ plot_results[1][col].columns[0] +' and ' + plot_results[1][col].columns[1] + '<br> 2Norm:' + str(twoNorm) + ', supNorm:' + str(supNorm) + '<br> supNorm_pct: '+str(supNorm_percentage)+'%')
                else:
                    subplot_titles.append(fixed[col] + '<br> 2Norm:' + str(twoNorm) + ', supNorm:' + str(supNorm) + '<br> supNorm_pct: '+str(supNorm_percentage)+'%')
            else:
                if fixed ==None:
                    subplot_titles.append('Conditional on '+ plot_results[0][col].columns[0] +' and ' + plot_results[0][col].columns[1] + '<br> MFR Suite Not Solved.')
                else:
                    subplot_titles.append(fixed[col]  + '<br> MFR Suite Not Solved.')
            subplot_type.append({'type': 'surface'})
        subplot_types.append(subplot_type)

    fig = make_subplots(rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=spacing, vertical_spacing=spacing, subplot_titles=(subplot_titles), specs=subplot_types)
    z_max = np.max([[np.max(i[col][var_name].max()) for col in range(plot_col_dims)] for i in plot_results])
    z_min = np.min([[np.min(i[col][var_name].min()) for col in range(plot_col_dims)] for i in plot_results])
    z_max = z_max + (z_max - z_min)*0.15
    z_min = z_min - (z_max - z_min)*0.15 
    z_max = np.max([z_max, 10**(-4)])
    z_min = np.min([z_min, -10**(-4)])
    for row in range(plot_row_dims):
        for col in range(plot_col_dims):
            if z_adjust == True:
                fig.update_scenes(dict(xaxis_title=plot_results[0][col].columns[0], yaxis_title=plot_results[0][col].columns[1], zaxis_title=var_name[row], zaxis = dict(nticks=4, range=[z_min,z_max], tickformat= ".4f")), row = row+1, col = col+1)
            else:
                fig.update_scenes(dict(xaxis_title=plot_results[0][col].columns[0], yaxis_title=plot_results[0][col].columns[1], zaxis_title=var_name[row]), row = row+1, col = col+1)
            for i in range(len(plot_results)):
                # print(i,col,row)
                fig.add_trace(go.Surface(
                    
                    z=plot_results[i][col].pivot(index=plot_results[i][col].columns[1], columns=plot_results[i][col].columns[0], values=var_name[row]),
                    x=plot_results[i][col].iloc[:,0].unique(),
                    y=plot_results[i][col].iloc[:,1].unique(),
                    colorscale=plot_color_style[i], showscale=False, name= plot_results_name[i], showlegend=True), row = row+1, col = col+1)
    
    chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = parameter_list
    fig.update_layout(title= 'MRF vs NN - '+ plot_content +'<br><span style="font-size: 12px;"> chiUnderline = '+ str(chiUnderline)+\
              ', a_e = ' + str(a_e) + ', a_h = ' + str(a_h) + ', gamma_e = ' + str(gamma_e) + ', gamma_h = ' +str(gamma_h) + ', psi_e = ' + str(psi_e) + ', psi_h = ' + str(psi_h) +'</span>',\
              title_x = 0.5, title_y = 0.97, height=height, width=width, title_yanchor = 'top')
    fig.update_layout(margin=dict(t=150))
    if fixed ==None:
      fig.write_json(path + '/' + plot_content + '.json')
    else:
      dict_name = ' '.join([i + ' '  + str(fix_dict[i]) for i in fix_dict])
      fig.write_json(path + '/' + plot_content + ' ' + dict_name +'.json')

def generateMomentPlots_2d(status, plot_results, var_name, plot_content, parameter_list, fixed = None, fix_dict = None, y_adjust = True, float_formatter = "{0:.4f}", height=800, width=1200, spacing = 0.1, path = os.path.dirname(os.getcwd()) + '/doc/'):
    
    plot_row_dims      = len(var_name)
    plot_col_dims      = len(plot_results[1])

    plot_results_name  = ['MFR', "NN"]
    plot_color_style   = ['royalblue', 'firebrick']

    if status != '0':
        plot_results      = [plot_results[1]]
        plot_results_name = [plot_results_name[1]]
        plot_color_style  = [plot_color_style[1]] 

    subplot_titles = []
    for row in range(plot_row_dims):
        for col in range(plot_col_dims):
            if status == '0':
                twoNorm = float_formatter.format(np.linalg.norm(plot_results[1][col][var_name[row]] - plot_results[0][col][var_name[row]]))
                supNorm = float_formatter.format(np.linalg.norm((plot_results[1][col][var_name[row]] - plot_results[0][col][var_name[row]]), np.inf))
                try:
                  supNorm_percentage = float_formatter.format(np.linalg.norm(((plot_results[1][col][var_name[row]] - plot_results[0][col][var_name[row]])/plot_results[0][col][var_name[row]]*100), np.inf))
                except:
                  supNorm_percentage = np.nan                
                if fixed ==None:
                    subplot_titles.append('Conditional on '+ plot_results[1][col].columns[0] + '<br> 2Norm:' + str(twoNorm) + ', supNorm:' + str(supNorm) + '<br> supNorm_pct: '+str(supNorm_percentage)+'(%)')
                else:
                    subplot_titles.append(fixed[col] + '<br> 2Norm:' + str(twoNorm) + ', supNorm:' + str(supNorm) + '<br> supNorm_pct: '+str(supNorm_percentage)+'%')
            else:
                if fixed ==None:
                    subplot_titles.append('Conditional on '+ plot_results[0][col].columns[0]  + '<br> MFR Suite Not Solved.')
                else:
                    subplot_titles.append(fixed[col]  + '<br> MFR Suite Not Solved.')

    fig = make_subplots(rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=spacing, vertical_spacing=spacing, subplot_titles=(subplot_titles))
    y_max = np.max([[np.max(i[col][var_name].max()) for col in range(plot_col_dims)] for i in plot_results])
    y_min = np.min([[np.min(i[col][var_name].min()) for col in range(plot_col_dims)] for i in plot_results])
    y_max = y_max + (y_max - y_min)*0.15
    y_min = y_min - (y_max - y_min)*0.15
    y_max = np.max([y_max, 10**(-4)])
    y_min = np.min([y_min, -10**(-4)])
    for row in range(plot_row_dims):
        for col in range(plot_col_dims):
            for i in range(len(plot_results)):
                fig.add_trace(go.Scatter(
                    x=plot_results[i][col].iloc[:,0],
                    y=plot_results[i][col][var_name[row]],
                    line=dict(color=plot_color_style[i], width=4), name = plot_results_name[i], showlegend=True), row = row+1, col = col+1)
            fig.update_xaxes(title_text=plot_results[0][col].columns[0], row=row+1, col=col+1)
            if y_adjust == True:
                fig.update_yaxes(title_text=var_name[row], range = [y_min,y_max], row=row+1, col=col+1, tickformat= ".4f")
                
    chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = parameter_list
    fig.update_layout(title= 'MRF vs NN - '+ plot_content +'<br><span style="font-size: 12px;"> chiUnderline = '+ str(chiUnderline)+\
              ', a_e = ' + str(a_e) + ', a_h = ' + str(a_h) + ', gamma_e = ' + str(gamma_e) + ', gamma_h = ' +str(gamma_h) + ', psi_e = ' + str(psi_e) + ', psi_h = ' + str(psi_h) +'</span>',\
              title_x = 0.5, title_y = 0.97, height=height, width=width, title_yanchor = 'top')
    fig.update_layout(margin=dict(t=150))
    if fixed ==None:
      fig.write_json(path + '/' + plot_content + '.json')
    else:
      dict_name = ' '.join([i + ' '  + str(fix_dict[i]) for i in fix_dict])
      fig.write_json(path + '/' + plot_content + ' ' + dict_name +'.json')

# def generateSurfacePlots(domain_folder, status, plot_results, fixed_points, X, function_name, var_name, plot_content, parameter_list, z_adjust = True, float_formatter = "{0:.4f}", height=800, width=1200, path = os.path.dirname(os.getcwd()) + '/doc/'):
    
#     W, Z, V, Vtilde    = X[:,0], X[:,1], X[:,2], X[:,3]
#     if domain_folder == 'WZV':
#       fixed_var          = [Z, V]
#       fixed_inv_var      = [V, Z]
#       fixed_var_name     = ['Z','V']
#       fixed_inv_var_name = ['V','Z']
#     elif domain_folder == 'WZVtilde':
#       fixed_var          = [Z, Vtilde]
#       fixed_inv_var      = [Vtilde, Z]
#       fixed_var_name     = ['Z','Vtilde']
#       fixed_inv_var_name = ['Vtilde','Z']

#     plot_results_name  = ['MFR', "NN"]
#     plot_color_style   = ['Viridis', 'Plasma']

#     n_points           = np.unique(W).shape[0]
#     plot_row_dims      = len(plot_results_name)
#     plot_col_dims      = len(plot_results[1])

#     if status != '0':
#       plot_results      = [plot_results[1]]
#       plot_results_name = [plot_results_name[1]]
#       plot_color_style  = [plot_color_style[1]]

#     fixed_idx = [fixed_var[i] == np.unique(fixed_var[i])[fixed_points[i]]         for i in range(plot_row_dims)]
#     fixed_val = [float_formatter.format(np.unique(fixed_var[i])[fixed_points[i]]) for i in range(plot_row_dims)]
    
#     fixed_subplot_titles = []
#     subplot_types = []
#     for row in range(plot_row_dims):
#       subplot_type = []
#       for col in range(plot_col_dims):
#         if status == '0':
#           fixed_twoNorm = float_formatter.format(np.linalg.norm(plot_results[1][col][fixed_idx[row]] - plot_results[0][col][fixed_idx[row]]))
#           fixed_subplot_titles.append(function_name[col] + '. '+ fixed_var_name[row] +' fixed at '+ str(fixed_val[row]) +'. <br> ||diff.||_2 = ' + str(fixed_twoNorm))
#         else:
#           fixed_subplot_titles.append(function_name[col] + '. '+ fixed_var_name[row] +' fixed at '+ str(fixed_val[row]) +'. <br> MFR Suite Not Solved.')
#         subplot_type.append({'type': 'surface'})
#       subplot_types.append(subplot_type)

#     fig = make_subplots(
#         rows=plot_row_dims, cols=plot_col_dims, horizontal_spacing=.1, vertical_spacing=.1,
#         subplot_titles=(fixed_subplot_titles), specs=subplot_types)
    
#     for row in range(plot_row_dims):
#       for col in range(plot_col_dims):
#         if z_adjust:
#           z_max = np.max([np.max(i) for i in plot_results])
#           z_min = np.min([np.min(i) for i in plot_results])
#           z_max = z_max + (z_max - z_min)*0.1
#           z_min = z_min - (z_max - z_min)*0.1
#           z_max = np.max([z_max, 10**(-4)])
#           z_min = np.min([z_min, -10**(-4)]) 
#           fig.update_scenes(dict(xaxis_title='W', yaxis_title=fixed_inv_var_name[row], zaxis_title=var_name[col],zaxis = dict(nticks=4, range=[z_min,z_max], tickformat= ".4f")), row = row+1, col = col+1)
#         else:
#           fig.update_scenes(dict(xaxis_title='W', yaxis_title=fixed_inv_var_name[row], zaxis_title=var_name[col]), row = row+1, col = col+1)
#         for i in range(len(plot_results)):
#           fig.add_trace(go.Surface(
#             x=W[fixed_idx[row]].reshape([n_points, 30], order='F'),
#             y=fixed_inv_var[row][fixed_idx[row]].reshape([n_points, 30], order='F'),
#             z=plot_results[i][col][fixed_idx[row]].reshape([n_points, 30], order='F'),
#             colorscale=plot_color_style[i], showscale=False, name= plot_results_name[i], showlegend=True), row = row+1, col = col+1)
    
#     chiUnderline, a_e, a_h, gamma_e, gamma_h, psi_e, psi_h = parameter_list
#     fig.update_layout(title= 'MRF vs NN - '+ plot_content +' surface plots <br><span style="font-size: 15px;"> chiUnderline = '+ str(chiUnderline)+\
#                       ', a_e = ' + str(a_e) + ', a_h = ' + str(a_h) + ', gamma_e = ' + str(gamma_e) + ', gamma_h = ' +str(gamma_h) + ', psi_e = ' + str(psi_e) + ', psi_h = ' + str(psi_h) +'</span>',\
#                       title_x = 0.5, title_y = 0.97, height=height, width=width)
#     fig.write_json(path + '/' + plot_content + '.json')
#     fig.show()

# def generateScatterPlots(mfr_Results, nn_Results, function_name, height=20, width=7):
#   plot_row_dims     = len(mfr_Results)
#   n_points          = 90000
#   fig, axes = plt.subplots(plot_row_dims,1, figsize = (height,width))
#   for i, ax in enumerate(axes.flatten()):
#     ax.scatter(np.arange(0,n_points), mfr_Results[i], label = 'MFR', alpha = 0.3, s=0.1)
#     ax.scatter(np.arange(0,n_points), nn_Results[i], label = 'NN', alpha = 0.3, s=0.1)
#     ax.set_title(function_name[i])
#     ax.legend()
#   fig.tight_layout()
#   plt.show()