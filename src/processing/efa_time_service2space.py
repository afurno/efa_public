'''
stdbuf -i0 -o0 -e0 python efa_time2space.py -i /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-11-28/traffic_load.dat -l grand_paris_2016-09-05_2016-11-28_all_services_30min -o /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-11-28/results_time2space 1> /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-11-28/time2space_out.log 2> /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-11-28/time2space_err.log
stdbuf -i0 -o0 -e0 python efa_time2space.py -i /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-11-28/traffic_load.dat -l grand_paris_2016-09-05_2016-11-28_all_services_1h -o /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-11-28/results_time2space 1> /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-11-28/time2space_out.log 2> /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-11-28/time2space_err.log
'''
import argparse
import numpy as np
import rpy2.robjects as ro
from rpy2 import rinterface
from rpy2.robjects.packages import importr
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from datetime import datetime
from processing import utilities
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from matplotlib.colors import LogNorm
import os

'''def pca_princomp(snapshot_filename, cor=True):    
    print "\n\n\n***Performing factorial analysis with princomp...\n"
    ro.r('fit <- princomp(x, cor= ' + ("T" if cor else "F") + ')')
    print("\n")
    
    print "\nPrinting a summary of the fitting matrix with princomp...\n"
    ro.r('summary(fit)') # print variance accounted for 
    print("\n")
    
    print "\nPrinting the fitting matrix with princomp...\n" 
    ro.r('summary(fit)') # print variance accounted for     
    print("\n")
    
    print "\nPrinting the loadings matrix with princomp...\n"
    ro.r('print(loadings(fit), digits=2, cutoff=0.4)')
    print("\n")
    
    loadings = np.asarray(ro.r('fit$loadings'))
    factor_names = np.asarray(ro.r('colnames(fit$loadings)'))
    cell_ids = np.asarray(ro.r('rownames(fit$loadings)'))
   
    scores = np.asarray(ro.r('fit$scores'))
    population_names = np.asarray(ro.r('rownames(fit$scores)'))
    score_factor_names = np.asarray(ro.r('colnames(fit$scores)'))
    scores_df = analyze_scores(scores, population_names, score_factor_names, "pca_princomp")
    
    print "Plotting scores (princomp)...\n"
    plot_factor_scores(factor_names[0:30], scores_df, "pca_princomp", False)
    plot_factor_scores(factor_names[0:30], scores_df, "pca_princomp", True)
    
    print "Saving loadings to file (princomp)..\n"
    loadings_df = analyze_loadings(loadings, factor_names, cell_ids, "pca_princomp")
        
def pca_principal(snapshot_filename, nfact, rotation="none", cor=True):        
    print "\n\n\n***Performing factorial analysis with principal and " + rotation + " rotation...\n"
    ro.r('fit <- principal(x, nfactors=' + nfact + ', rotate="' + rotation + '", scores=T, covar= ' + ("F" if cor else "T") + ')')
    print("\n")
    
    print "\nPrinting a summary of the fitting matrix with principal and " + rotation + " rotation...\n" 
    ro.r('summary(fit)') # print variance accounted for 
    print("\n")
    
    print "\nPrinting the fitting matrix with principal and " + rotation + " rotation...\n" 
    ro.r('print(fit)')    
    print("\n")
    
    print "\nPrinting the loadings matrix with principal and " + rotation + " rotation...\n" 
    ro.r('print(loadings(fit), digits=2, cutoff=0.4)')
    print("\n")
    
    loadings = np.asarray(ro.r('fit$loadings'))
    factor_names = np.asarray(ro.r('colnames(fit$loadings)'))
    cell_ids = np.asarray(ro.r('rownames(fit$loadings)'))
    
    scores = np.asarray(ro.r('fit$scores'))
    #print "Dimension of varimax score matrix:\n%s" % str(ro.r('dim(fit$scores)'))
    #print "Head of varimax score matrix:\n%s" % str(ro.r('head(fit$scores)'))
    population_names = np.asarray(ro.r('rownames(fit$scores)'))
    score_factor_names = np.asarray(ro.r('colnames(fit$scores)'))
    #print "Cell names: %s" % str(cell_names)
    #print "Score factor names: %s" % str(score_factor_names)
    scores_df = analyze_scores(scores, population_names, score_factor_names, "pca_principal_" + rotation)
    
    print "Plotting scores (principal and " + rotation + " rotation)...\n"
    plot_factor_scores(factor_names, scores_df, "pca_principal_" + rotation, False)
    plot_factor_scores(factor_names, scores_df, "pca_principal_" + rotation, True)
    
    print "Saving loadings to file (principal and " + rotation + " rotation)...\n"
    loadings_df = analyze_loadings(loadings, factor_names, cell_ids, "pca_principal_" + rotation)'''

def efa(snapshot_filename, nfact, output_folder, rotation="none", method="ml", cor=True, period_min=60):
    #print "Dimension of x matrix:\n%s" % str(ro.r('dim(x)'))
    #print "Head of x matrix:\n%s" % str(ro.r('head(x)'))
    #ro.r('ev <- eigen(cor(x))') # get eigenvalues
    #ro.r('ap <- parallel(subject=nrow(x),var=ncol(x),rep=100,cent=.05)')
    #ro.r('nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)')
    #ro.r('plotnScree(nS)')
    #compute num factors
    
    print "\n\n\n***Performing factorial analysis with " + method + " fitting method, " + rotation + " rotational method...\n"
    ro.r('fit <- fa(r=x,nfactors=' + nfact + ',n.iter=1, min.err = 0.001,  max.iter = 50, \
            rotate="' + rotation + '", scores="regression", \
            residuals=TRUE, SMC=TRUE, missing=FALSE,impute="median",\
            warnings=TRUE, fm="' + method + '",\
            alpha=.1,p=.05,oblique.scores=FALSE,use="pairwise", \
            covar= ' + ("F" if cor else "T") + ', cor = "' + ("cor" if cor else "cov") + '")')
         
    print "\nPrinting a summary of the fitting matrix with method " + method + " and rotation " + rotation + "...\n" 
    ro.r('summary(fit)') # print variance accounted for 
    print("\n")
    
    print "\nPrinting the fitting matrix with method " + method + " and rotation " + rotation + "...\n" 
    ro.r('print(fit)')
    print("\n")
    
    print "\nPrinting the loadings matrix with method " + method + " and " + rotation + " rotation...\n" 
    ro.r('print(loadings(fit), digits=2, cutoff=0.4)')
    print("\n")
    
    loadings = np.asarray(ro.r('fit$loadings'))
    factor_names = np.asarray(ro.r('colnames(fit$loadings)'))
    cell_ids = np.asarray(ro.r('rownames(fit$loadings)'))
    
    traffic_matrix = np.asarray(ro.r('x'))
    traffic_snapshots = np.asarray(ro.r('rownames(x)'))
    traffic_base_stations = np.asarray(ro.r('colnames(x)'))

    full_scores_df = pd.DataFrame()
    for c in CATEGORIES:
        searched_category = ',' + c + ',' #2016-09-05.Monday,00:00:00,NI,
        output_subfolders = output_folder + "/" + method + "_" + rotation + "/" + c + "/"
        if not os.path.exists(os.path.dirname(output_subfolders)):
            os.makedirs(os.path.dirname(output_subfolders))
        found_elements = [(i, (v.split(",")[0]+','+v.split(",")[1])) for i, v in enumerate(traffic_snapshots) if searched_category in v]
        if len(found_elements) == 0:
            print("*** PROBLEM HERE: no element for category " + c)
            continue
        indices, c_traffic_snapshots = zip(*found_elements)
        indices = list(indices)
        c_traffic_snapshots = list(c_traffic_snapshots)
        c_traffic_matrix = traffic_matrix[indices]
        #print("*** For category " + c + "\nFound elements: " + str(found_elements) + "\nIndices: " + str(indices) + "\nc_traffic_matrix: " + str(c_traffic_matrix) + "\nCreated output folder: " + output_folder)
        signatures, signatures_df, _ = utilities.generate_signatures(c_traffic_matrix, c_traffic_snapshots, \
                                                     traffic_base_stations, loadings, factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/")
        
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, normalized_y=True)
        
        signatures, signatures_df, _ = utilities.generate_signatures(c_traffic_matrix, c_traffic_snapshots, \
                                                     traffic_base_stations, loadings, factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", threshold=0.4)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, threshold=0.4)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, normalized_y=True, threshold=0.4)
        
        signatures, signatures_df, _ = utilities.generate_signatures(c_traffic_matrix, c_traffic_snapshots, \
                                                     traffic_base_stations, loadings, factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", threshold=0.6)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, threshold=0.6)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, normalized_y=True, threshold=0.6)
        
        signatures, signatures_df, _ = utilities.generate_signatures(c_traffic_matrix, c_traffic_snapshots, \
                                                     traffic_base_stations, loadings, factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", threshold=0.8)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, threshold=0.8)
        utilities.plot_time_signatures(signatures, signatures_df["index"], factor_names, label, output_folder, method + "_" + rotation + "/" + c + "/", period_min, normalized_y=True, threshold=0.8)
        
        #Analyzing scores
        scores = np.asarray(ro.r('fit$scores'))
        population_names = np.asarray(ro.r('rownames(fit$scores)'))
        score_factor_names = np.asarray(ro.r('colnames(fit$scores)'))
        
        found_elements = [(i, (v.split(",")[0]+','+v.split(",")[1])) for i, v in enumerate(population_names) if searched_category in v]
        if len(found_elements) == 0:
            print("*** PROBLEM HERE: no element for category " + c)
            continue
        indices, c_traffic_snapshots = zip(*found_elements)
        indices = list(indices)
        c_population_names = list(c_traffic_snapshots)
        c_scores = scores[indices]
        
        scores_df = analyze_scores(c_scores, c_population_names, score_factor_names, output_folder, method + "_" + rotation + "/" + c + "/")
        scores_df["category"] = c
        full_scores_df = full_scores_df.append(scores_df, ignore_index=True)
        
        print "Plotting scores (" + method + " and rotation " + rotation + ")...\n"
        plot_factor_scores(factor_names, scores_df, output_folder, method + "_" + rotation + "/" + c + "/", False)
        plot_factor_scores(factor_names, scores_df, output_folder, method + "_" + rotation + "/" + c + "/", True)
        
    #Analyzing loadings
    print "Saving loadings to file (" + method + " and rotation " + rotation + ")...\n"
    loadings_df = analyze_loadings(loadings, factor_names, cell_ids, output_folder, method + "_" + rotation + "/" + c + "/")
    
    #Analyzing uniqueness
    print "\nPrinting the uniqueness matrix with method " + method + " and " + rotation + " rotation...\n" 
    ro.r('print(fit$uniquenesses)')
    print("\n")
    
    uniquenesses = np.asarray(ro.r('fit$uniquenesses'))
    uniquenesses_names = np.asarray(ro.r('names(fit$uniquenesses)'))
    
    print "Saving uniquenesses to file (" + method + " and rotation " + rotation + ")...\n"
    uniquenesses_df = analyze_uniquenesses(uniquenesses, uniquenesses_names, output_folder, method + "_" + rotation + "/" + c + "/")
    
    output_filename = output_folder + "/" + method + "_" + rotation + "/all_loadings" + ("_" + label if label is not None else "") + ".csv"
    loadings_df.to_csv(output_filename, header=True, sep = ",", index=False)
    output_filename = output_folder + "/" + method + "_" + rotation + "/all_scores" + ("_" + label if label is not None else "") + ".csv"
    full_scores_df.to_csv(output_filename, header=True, sep = ",", index=False)
    output_filename = output_folder + "/" + method + "_" + rotation + "/all_uniquenesses" + ("_" + label if label is not None else "") + ".csv"
    uniquenesses_df.to_csv(output_filename, header=True, sep = ",", index=False)
    return loadings_df, full_scores_df, uniquenesses_df
    

def my_transform1(x):
    splitted_x = str(x).split(".")
    return splitted_x[0]

def my_transform2(x):
    splitted_x = str(x).split(".")
    splitted_x = splitted_x[1].split(",")
    return splitted_x[1]

def my_transform3(x):
    x = str(x)
    if x.startswith("X"):
        return x[1:]
    return x

def analyze_loadings(loadings, factor_names, cell_ids, output_folder, folder_prefix):
    #create dataframe reporting for each snapshot the loading associated to each factor
    loadings_df = pd.DataFrame(loadings, columns=factor_names, index=cell_ids)
    loadings_df = loadings_df.reset_index()
    #print loadings_df.head()
    loadings_filename = output_folder + "/" + folder_prefix + "/loadings" + ("_" + label if label is not None else "") + ".csv"
    if not os.path.exists(os.path.dirname(loadings_filename)):
        os.makedirs(os.path.dirname(loadings_filename))
    loadings_df = loadings_df.rename(columns = {'index':'cell_id'})
    loadings_df["cell_id"] = loadings_df["cell_id"].apply(my_transform3)
    loadings_df.to_csv(loadings_filename, header=True, sep = ",", index=False)
    return loadings_df

def analyze_uniquenesses(uniquenesses, cell_ids, output_folder, folder_prefix):
    #create dataframe reporting for each snapshot the loading associated to each factor
    uniquenesses_df = pd.DataFrame(uniquenesses, columns=["uniquenesses"], index=cell_ids)
    uniquenesses_df = uniquenesses_df.reset_index()
    #print uniquenesses_df.head()
    uniquenesses_filename = output_folder + "/" + folder_prefix + "/uniquenesses" + ("_" + label if label is not None else "") + ".csv"
    if not os.path.exists(os.path.dirname(uniquenesses_filename)):
        os.makedirs(os.path.dirname(uniquenesses_filename))
    uniquenesses_df = uniquenesses_df.rename(columns = {'index':'cell_id'})
    uniquenesses_df["cell_id"] = uniquenesses_df["cell_id"].apply(my_transform3)
    uniquenesses_df.to_csv(uniquenesses_filename, header=True, sep = ",", index=False)
    return uniquenesses_df

def analyze_scores(scores, population_names, score_factor_names, output_folder, folder_prefix):
    #create dataframe reporting for each snapshot the loading associated to each factor
    scores_df = pd.DataFrame(scores, columns=score_factor_names, index=population_names)
    scores_df = scores_df.reset_index()
    #print loadings_df.head()
    scores_df["day"] = scores_df["index"].apply(my_transform1)
    scores_df["time"] = scores_df["index"].apply(my_transform2)
    #print loadings_df.head()
    scores_filename = output_folder + "/" + folder_prefix + "/scores" + ("_" + label if label is not None else "") + ".csv"
    if not os.path.exists(os.path.dirname(scores_filename)):
        os.makedirs(os.path.dirname(scores_filename))
    scores_df = scores_df.rename(columns = {'index':'snapshot_name'})
    scores_df.to_csv(scores_filename, header=True, sep = ",", index=False)
    return scores_df

def plot_factor_scores(factor_names, scores_df, output_folder, folder_prefix, norm_colors):
    # plot with various axes scales
    num_subplots = len(factor_names)
    #plot_num_rows = int(math.ceil(num_subplots/3.))
    #plot_num_columns = int(math.ceil(num_subplots / float(plot_num_rows)))
    #fig, axs = plt.subplots(plot_num_rows, plot_num_columns)
    #fig.set_size_inches(30, plot_num_rows * 20)
    page_counter = 1
    max_figs_per_line = 5
    max_rows_per_page = 5
    fig_basename = output_folder + "/" + folder_prefix + "/landuses" + ("_" + label if label is not None else "") \
            + ("_colors_normalized" if norm_colors else "")
    if norm_colors:
        min_score = scores_df[factor_names].min().min()
        max_score = scores_df[factor_names].max().max()
        print "Min score is %f" %min_score
        print "Max score is %f" %max_score
    counter = 0
    f = None
    for factor in factor_names:
        if counter % (max_figs_per_line * max_rows_per_page) == 0:
            if f != None:
                plt.tight_layout()
                plt.savefig('%s_%02d.pdf' % (fig_basename, page_counter))
                plt.close()
                page_counter += 1
                counter = 0
            f, axs = plt.subplots(max_rows_per_page, max_figs_per_line)
            
            if num_subplots == 1:
                axs = np.array([axs])
            axs = axs.ravel()
            # increase the size of the figure such that the x-axis is 4 times bigger; we assume current value of x-axis is 6    
            f.set_size_inches(6 * max_figs_per_line, 4 * max_rows_per_page)
        temp_df = scores_df.pivot(index='day', columns='time', values=factor)
        index = sorted(temp_df.index, reverse=True)
        temp_df = temp_df.reindex(index)
        row_labels =  temp_df.index.values.tolist()
        col_labels = temp_df.columns.values.tolist()
        if norm_colors:
            ms = axs[counter].matshow(temp_df.as_matrix(),vmin=min_score, vmax=max_score, cmap="jet")
        else:
            ms = axs[counter].matshow(temp_df.as_matrix(), cmap="jet")
            divider = make_axes_locatable(axs[counter])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(ms, cax=cax)
        axs[counter].xaxis.set_ticks_position('bottom')
        axs[counter].set_xticks(range(len(col_labels)))
        xtick_labels = [""] * len(col_labels)
        col_labels = [hourtime[0:5] for hourtime in col_labels]
        xtick_labels = [(col_labels[i] if (i % 4 == 0 or i == len(col_labels) - 1) else "") for i in range(len(col_labels)) ]
        axs[counter].set_xticklabels(xtick_labels, rotation=90)
        axs[counter].set_yticks(range(len(row_labels)))
        y_labels = [""] * len(row_labels)
        if is_typical_week:
            row_labels = [datetime.strptime(date_string, '%Y-%m-%d').strftime('%A') for date_string in row_labels]
        else:
            row_labels[0] = row_labels[0] + " (" + (datetime.strptime(row_labels[0], '%Y-%m-%d').strftime('%A'))[0:3] + ")"
            if len(row_labels) > 1:
                row_labels[len(row_labels) - 1] = row_labels[len(row_labels) - 1] + " (" + (datetime.strptime(row_labels[len(row_labels) - 1], '%Y-%m-%d').strftime('%A'))[0:3] + ")"
        y_labels = [(row_labels[len(row_labels) - 1 - i] if (i % 7 == 0 or i == len(row_labels) - 1) else "") for i in range(len(row_labels))]
        y_labels.reverse()
        axs[counter].set_yticklabels(y_labels)
        axs[counter].set_title(factor)
        counter += 1

    i = 0
    while counter % (max_figs_per_line * max_rows_per_page) != 0:
        axs[counter].axis('off')
        counter += 1 

    if norm_colors:
        f.subplots_adjust(right=0.92)
        cbar_ax = f.add_axes([0.95, 0.18, 0.015, 0.7])
        f.colorbar(ms, cax=cbar_ax)
        
    if f != None:
        plt.tight_layout()
        plt.savefig('%s_%02d.pdf' % (fig_basename, page_counter))
        plt.close()
    
'''def z_score_normalize_cell_activity(cells_df):
    temp_df_mean = cells_df[["time", "total_volume_per_day_of_the_week"]].groupby("time").mean()
    temp_df_mean = temp_df_mean.rename(columns = {'total_volume_per_day_of_the_week':'mean'})
    temp_df_mean = temp_df_mean.reset_index()
    cells_df = pd.merge(cells_df, temp_df_mean, on = 'time')
    temp_df_std = cells_df[["time", "total_volume_per_day_of_the_week"]].groupby("time").std()
    temp_df_std = temp_df_std.rename(columns = {'total_volume_per_day_of_the_week':'std'})
    temp_df_std = temp_df_std.reset_index()
    cells_df = pd.merge(cells_df, temp_df_std, on = 'time')
    cells_df["total_volume_per_day_of_the_week"] = (cells_df["total_volume_per_day_of_the_week"] - cells_df["mean"]) / cells_df["std"]
    cells_df.loc[np.isnan(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df.loc[np.isinf(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df = cells_df[["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "total_volume_per_day_of_the_week", "time"]].sort_values(by = ["fake_date", "hour", "cell-id"])
    new_filename = os.path.splitext(input_file)[0] + "_z_score_norm_time" + os.path.splitext(input_file)[1]
    cells_df.to_csv(new_filename, index=False, sep = " ", columns=["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "total_volume_per_day_of_the_week"])
    return cells_df

def min_max_normalize_cell_activity(cells_df):
    temp_df_max = cells_df[["time", "total_volume_per_day_of_the_week"]].groupby("time").max()
    temp_df_max = temp_df_max.rename(columns = {'total_volume_per_day_of_the_week':'max'})
    temp_df_max = temp_df_max.reset_index()
    cells_df = pd.merge(cells_df, temp_df_max, on = 'time')
    temp_df_min = cells_df[["time", "total_volume_per_day_of_the_week"]].groupby("time").min()
    temp_df_min = temp_df_min.rename(columns = {'total_volume_per_day_of_the_week':'min'})
    temp_df_min = temp_df_min.reset_index()
    cells_df = pd.merge(cells_df, temp_df_min, on = 'time')
    cells_df["total_volume_per_day_of_the_week"] = (cells_df["total_volume_per_day_of_the_week"] - cells_df["min"]) / (cells_df["max"] - cells_df["min"])
    cells_df.loc[np.isnan(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df.loc[np.isinf(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df = cells_df[["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "total_volume_per_day_of_the_week", "time"]].sort_values(by = ["fake_date", "hour", "cell-id"])
    new_filename = os.path.splitext(input_file)[0] + "_min_max_norm_time" + os.path.splitext(input_file)[1]
    cells_df.to_csv(new_filename, index=False, sep = " ", columns=["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "total_volume_per_day_of_the_week"])
    return cells_df'''

#read arguments
parser = argparse.ArgumentParser(prog='exploratory_factor_analysis_network_profiles', usage='%(prog)s -r start_date1 end_date1 -r start_date2 end_date2 ... -i input_file -a antennas_file -n True/False')
parser.add_argument ('-i', '--input_file')
parser.add_argument ('-o', '--output_folder')
parser.add_argument ('-l', '--output_label', default=None)
parser.add_argument ('-f', '--num_factors', default="auto")
parser.add_argument ('-p', '--period_min', default=60)
#parser.add_argument ('-zn', '--z_score_normalize_cells', action='store_true', default = False)
#parser.add_argument ('-mmn', '--min_max_normalize_cells', action='store_true', default = False)
parser.add_argument ('-w', '--typical_week', action='store_true', default = False)
parser.add_argument ('-cov', '--covariance', action='store_true', default = False)

cor = not(parser.parse_args().covariance)

input_file = parser.parse_args().input_file
label = parser.parse_args().output_label
period_min = int(parser.parse_args().period_min)
#lower_value = parser.parse_args().lower_value
#norm = parser.parse_args().normalized
is_typical_week= parser.parse_args().typical_week
#nstart = parser.parse_args().start_values
output_folder = parser.parse_args().output_folder
snapshots_filename = output_folder + "/snapshot_matrix_for_landuse_classification" + ("_" + label if label is not None else "") + ".dat"

#read traffic file
NAMES = ["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "category", "total_volume_per_day_of_the_week"]
cells_df = pd.read_csv(input_file, sep=" ", header=0,
                       names=NAMES)
cells_df = cells_df[cells_df["category"] != "other"]
CATEGORIES = cells_df.category.unique()
#cells_df = cells_df.groupby(["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area"]).sum()
cells_df = cells_df.reset_index()
cells_df = cells_df[["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "category", "total_volume_per_day_of_the_week"]]
'''if parser.parse_args().min_max_normalize_cells:
    cells_df = min_max_normalize_cell_activity(cells_df)
elif parser.parse_args().z_score_normalize_cells:
    cells_df = z_score_normalize_cell_activity(cells_df)'''

cells_df["time_service"] = cells_df["fake_date"] + "." + cells_df["#day_of_the_week"].map(str) + "," + cells_df["hour"].map(str) + "," + cells_df["category"].map(str) + ","
'''num_snapshots = len(cells_df["time_service"].unique())
num_cells = len(cells_df["cell-id"].unique())

print "There are %d snapshots and %d cells in the initial dataset" %(num_snapshots, num_cells)
temp_df = cells_df[cells_df["total_volume_per_day_of_the_week"] > 0]
temp_df = temp_df[["cell-id", "total_volume_per_day_of_the_week"]]
temp_df = temp_df.groupby("cell-id",as_index=True).count()
temp_df = temp_df["total_volume_per_day_of_the_week"].reset_index()
cells_with_traffic = temp_df[temp_df["total_volume_per_day_of_the_week"] > 0.1*num_snapshots]["cell-id"].unique()
cells_df = cells_df[cells_df["cell-id"].isin(cells_with_traffic)]
print "There are %d cells after cleaning" %(len(cells_df["cell-id"].unique()))

if parser.parse_args().min_max_normalize_cells:
    cells_df = min_max_normalize_cell_activity(cells_df)
elif parser.parse_args().z_score_normalize_cells:
    cells_df = z_score_normalize_cell_activity(cells_df)'''
    
#print cells_df.head()
cells_df = cells_df.pivot(index='time_service', columns='cell-id', values='total_volume_per_day_of_the_week')
#print cells_df.head()

if not os.path.exists(os.path.dirname(snapshots_filename)):
    os.makedirs(os.path.dirname(snapshots_filename))    
cells_df.to_csv(snapshots_filename, index=True, header=True, sep = " ", index_label=False)

importr('psych')
importr('nFactors')

#read data matrix X
ro.r('x<-rbind(as.matrix(read.table(file="'+ snapshots_filename +'", sep=" ", header=TRUE, row.names=1)))')

if parser.parse_args().num_factors == "auto":
    output_file =  output_folder + "/scree_plot.pdf"
    ro.r('pdf("' + output_file + '")')
    ro.r('fa_score <- fa.parallel(x, fa = "both")')
    ro.r('dev.off()')
    nfact = str(ro.r('fa_score$nfact')[0]) # get eigenvalues
    print "\n\nBest number of factors:\n%s" % nfact
else:
    nfact = parser.parse_args().num_factors
    print "\n\nUsing %s as the number of factors." % nfact

#efa(snapshots_filename, nfact, output_folder, rotation="none", method="ml", cor=cor, period_min = period_min)
efa(snapshots_filename, nfact, output_folder, rotation="varimax", method="ml", cor=cor, period_min = period_min)

try:
    efa(snapshots_filename, nfact, rotation="varimax", method="ml", cor=cor, period_min = period_min)
except rinterface.RRuntimeError:
    print "***ERROR while performing fa with VARIMAX and ml method... Using minres only"
try:
    efa(snapshots_filename, nfact, rotation="varimax", method="minres", cor=cor, period_min = period_min)
except rinterface.RRuntimeError:
    print "***ERROR while performing fa with VARIMAX and ml minres... No other method available"
#efa(snapshots_filename, nfact, output_folder, rotation="promax", method="ml", cor=cor, period_min = period_min)

'''
#efa(snapshots_filename, nfact, rotation="oblimin", method="ml", cor=cor)
pca_princomp(snapshots_filename, cor=cor)

pca_principal(snapshots_filename, nfact, rotation="none", cor=cor)
pca_principal(snapshots_filename, nfact, rotation="varimax", cor=cor)
pca_principal(snapshots_filename, nfact, rotation="promax", cor=cor)
#pca_principal(snapshots_filename, nfact, rotation="oblimin", cor=cor)

efa(snapshots_filename, nfact, rotation="none", method="minres", cor=cor)
efa(snapshots_filename, nfact, rotation="varimax", method="minres", cor=cor)
efa(snapshots_filename, nfact, rotation="promax", method="minres", cor=cor)
#efa(snapshots_filename, nfact, rotation="oblimin", method="minres", cor=cor)

efa(snapshots_filename, nfact, rotation="none", method="pa", cor=cor)
efa(snapshots_filename, nfact, rotation="varimax", method="pa", cor=cor)
efa(snapshots_filename, nfact, rotation="promax", method="pa", cor=cor)
#efa(snapshots_filename, nfact, rotation="oblimin", method="pa", cor=cor)
'''