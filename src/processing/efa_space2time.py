'''
stdbuf -i0 -o0 -e0 python efa_space2time.py -i /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-10-30/med7d.dat -w -l grand_paris_median_2016-09-05_2016-10-30_all_services_1h -o /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-10-30/results_space2time 1> /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-10-30/space2time_out.log 2> /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-10-30/space2time_err.log'''
import argparse
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2 import rinterface
import pandas as pd
import matplotlib

matplotlib.use('Agg')
from processing import utilities
import matplotlib.pyplot as plt
from datetime import datetime
import locale
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

'''
def pca_princomp(snapshot_filename, cor=True):
    print "\n\n\n***Performing factorial analysis with princomp...\n"
    ro.r('fit <- princomp(x, cor= ' + ("T" if cor else "F") + ')')
    print("\n")

    print "\nPrinting a summary of the fitting matrix with princomp...\n"
    ro.r('summary(fit)')  # print variance accounted for
    print("\n")

    print "\nPrinting the fitting matrix with princomp...\n"
    ro.r('summary(fit)')  # print variance accounted for
    print("\n")

    print "\nPrinting the loadings matrix with princomp...\n"
    ro.r('print(loadings(fit), digits=2, cutoff=0.4)')
    print("\n")

    loadings = np.asarray(ro.r('fit$loadings'))
    factor_names = np.asarray(ro.r('colnames(fit$loadings)'))
    snapshot_names = np.asarray(ro.r('rownames(fit$loadings)'))

    scores = np.asarray(ro.r('fit$scores'))
    population_names = np.asarray(ro.r('rownames(fit$scores)'))
    score_factor_names = np.asarray(ro.r('colnames(fit$scores)'))
    analyze_scores(scores, population_names, score_factor_names, "pca_princomp")

    print "Saving loadings to file (princomp)..\n"
    loadings_df = analyze_loadings(loadings, factor_names, snapshot_names, "pca_princomp")

    print "Plotting loadings (princomp)...\n"
    plot_factor_loadings(factor_names, loadings_df, "pca_princomp", False)
    plot_factor_loadings(factor_names, loadings_df, "pca_princomp", True)


def pca_principal(snapshot_filename, nfact, rotation="none", cor=True):
    print "\n\n\n***Performing factorial analysis with principal and " + rotation + " rotation...\n"
    ro.r('fit <- principal(x, nfactors=' + nfact + ', rotate="' + rotation + '", scores=T, covar= ' + (
    "F" if cor else "T") + ')')
    print("\n")

    print "\nPrinting a summary of the fitting matrix with principal and " + rotation + " rotation...\n"
    ro.r('summary(fit)')  # print variance accounted for
    print("\n")

    print "\nPrinting the fitting matrix with principal and " + rotation + " rotation...\n"
    ro.r('print(fit)')
    print("\n")

    print "\nPrinting the loadings matrix with principal and " + rotation + " rotation...\n"
    ro.r('print(loadings(fit), digits=2, cutoff=0.4)')
    print("\n")

    loadings = np.asarray(ro.r('fit$loadings'))
    factor_names = np.asarray(ro.r('colnames(fit$loadings)'))
    snapshot_names = np.asarray(ro.r('rownames(fit$loadings)'))

    scores = np.asarray(ro.r('fit$scores'))
    # print "Dimension of varimax score matrix:\n%s" % str(ro.r('dim(fit$scores)'))
    # print "Head of varimax score matrix:\n%s" % str(ro.r('head(fit$scores)'))
    population_names = np.asarray(ro.r('rownames(fit$scores)'))
    score_factor_names = np.asarray(ro.r('colnames(fit$scores)'))
    # print "Cell names: %s" % str(cell_names)
    # print "Score factor names: %s" % str(score_factor_names)
    analyze_scores(scores, population_names, score_factor_names, "pca_principal_" + rotation)

    print "Saving loadings to file (principal and " + rotation + " rotation)...\n"
    loadings_df = analyze_loadings(loadings, factor_names, snapshot_names, "pca_principal_" + rotation)

    print "Plotting loadings (principal and " + rotation + " rotation)...\n"
    plot_factor_loadings(factor_names, loadings_df, "pca_principal_" + rotation, False)
    plot_factor_loadings(factor_names, loadings_df, "pca_principal_" + rotation, True)
'''

def efa(nfact, rotation="none", method="ml", cor=True, period_min=60):
    # print "Dimension of x matrix:\n%s" % str(ro.r('dim(x)'))
    # print "Head of x matrix:\n%s" % str(ro.r('head(x)'))
    # ro.r('ev <- eigen(cor(x))') # get eigenvalues
    # ro.r('ap <- parallel(subject=nrow(x),var=ncol(x),rep=100,cent=.05)')
    # ro.r('nS <- nScree(x=ev$values, aparallel=ap$eigen$qevpea)')
    # ro.r('plotnScree(nS)')
    # compute num factors

    print "\n\n\n***Performing factorial analysis with " + method + " fitting method, " + rotation + " rotational method...\n"
    ro.r('fit <- fa(r=x,nfactors=' + nfact + ',n.iter=1, min.err = 0.001,  max.iter = 50, \
            rotate="' + rotation + '", scores="regression", \
            residuals=TRUE, SMC=TRUE, missing=FALSE,impute="median",\
            warnings=TRUE, fm="' + method + '",\
            alpha=.1,p=.05,oblique.scores=FALSE,use="pairwise", \
            covar= ' + ("F" if cor else "T") + ', cor = "' + ("cor" if cor else "cov") + '")')

    print "\nPrinting a summary of the fitting matrix with method " + method + " and rotation " + rotation + "...\n"
    ro.r('summary(fit)')  # print variance accounted for
    print("\n")

    print "\nPrinting the fitting matrix with method " + method + " and rotation " + rotation + "...\n"
    ro.r('print(fit)')
    print("\n")

    print "\nPrinting the loadings matrix with method " + method + " and " + rotation + " rotation...\n"
    ro.r('print(loadings(fit), digits=2, cutoff=0.4)')
    print("\n")

    loadings = np.asarray(ro.r('fit$loadings'))
    factor_names = np.asarray(ro.r('colnames(fit$loadings)'))
    print "*** Here follows the factor names:"
    print factor_names
    snapshot_names = np.asarray(ro.r('rownames(fit$loadings)'))
    print "*** Here follows the snapshot names:"
    print snapshot_names

    # Analyzing scores
    scores = np.asarray(ro.r('fit$scores'))
    population_names = np.asarray(ro.r('rownames(fit$scores)'))
    score_factor_names = np.asarray(ro.r('colnames(fit$scores)'))
    scores_df = analyze_scores(scores, population_names, score_factor_names, method + "_" + rotation)
    
    traffic_matrix = np.asarray(ro.r('x'))
    traffic_base_stations = np.asarray(ro.r('rownames(x)'))
    traffic_snapshots = np.asarray(ro.r('colnames(x)'))
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + rotation)
    
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + rotation, threshold=0.4)
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + rotation, threshold=0.6)
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + rotation, threshold=0.8)

    # Analyzing loadings
    print "Saving loadings to file (" + method + " and rotation " + rotation + ")...\n"
    loadings_df = analyze_loadings(loadings, factor_names, snapshot_names, method + "_" + rotation)

    print "Plotting loadings (" + method + " and rotation " + rotation + ")...\n"
    plot_factor_loadings(factor_names, loadings_df, method + "_" + rotation, False)
    plot_factor_loadings(factor_names, loadings_df, method + "_" + rotation, True)
    utilities.plot_time_signatures(loadings_df[factor_names].as_matrix(), loadings_df["snapshot_name"], factor_names, label, output_folder, method + "_" + rotation + "/loadings/", period_min)

    # Analyzing uniqueness
    print "\nPrinting the uniqueness matrix with method " + method + " and " + rotation + " rotation...\n"
    ro.r('print(fit$uniquenesses)')
    print("\n")

    uniquenesses = np.asarray(ro.r('fit$uniquenesses'))
    uniquenesses_names = np.asarray(ro.r('names(fit$uniquenesses)'))

    print "Saving uniquenesses to file (" + method + " and rotation " + rotation + ")...\n"
    uniquenesses_df = analyze_uniquenesses(uniquenesses, uniquenesses_names, method + "_" + rotation)

    print "Plotting uniquenesses (" + method + " and rotation " + rotation + ")...\n"
    plot_factor_loadings(["uniquenesses"], uniquenesses_df, method + "_" + rotation, False, True)
    utilities.plot_time_signatures(uniquenesses_df["uniquenesses"].as_matrix(), uniquenesses_df["snapshot_name"], ["uniquenesses"], label, output_folder, method + "_" + rotation + "/uniquenesses/", period_min)
    
    return loadings_df, scores_df, uniquenesses_df

def nnmf(nfact, method="scd", loss="mse", period_min=60):    
    print "\n\n\n***Performing non-negative-matrix factorization with " + method + " method and " + loss + " loss function...\n"
    ro.r('fit<-nnmf(x, ' + nfact + ', max.iter = 100000, rel.tol = -1, method = "' + method + '",inner.max.iter = 1, loss="' + loss + '", n.threads=20);')
         
    print "\nPrinting a summary of the fitting matrix with method " + method + " and loss " + loss + "...\n" 
    ro.r('summary(fit)')  # print variance accounted for 
    print("\n")
    
    print "\nPrinting the fitting matrix with method " + method + " and loss " + loss + "...\n" 
    ro.r('print(fit)')
    print("\n")
    
    print "\nPrinting the score matrix with method " + method + " and loss " + loss + "...\n" 
    ro.r('print(fit$W, digits=2, cutoff=0.4)')
    print("\n")
    
    print "\nPrinting the factor matrix with method " + method + " and loss " + loss + "...\n" 
    ro.r('print(t(fit$H), digits=2, cutoff=0.4)')
    print("\n")
    
    print "\nPlotting convergence to epoch with method " + method + " and loss " + loss + "...\n" 
    ro.r('jpeg(file = "' + output_folder + "/" + method + "_" + loss + '_convergence.jpeg")')
    ro.r('plot(NULL, xlim = c(1, 3000), ylim = c(0.15, 0.45), xlab = "Epochs", ylab = "' + loss + '");')
    ro.r('lines(cumsum(fit$average.epochs), fit$' + loss + ');')
    ro.r('dev.off()')
    print("\n")
    
    print "\nPrinting the loss and the cumsum with method " + method + " and loss " + loss + "...\n" 
    ro.r('print(fit$loss, digits=2, cutoff=0.4)')
    ro.r('print(cumsum(fit$average.epochs), digits=2, cutoff=0.4)')
    print("\n")
    
    print "\nPlotting the heatmaps with method " + method + " and loss " + loss + "...\n" 
    ro.r('jpeg(file = "' + output_folder + "/" + method + "_" + loss + '_heatmap_W.jpeg")')
    ro.r('heatmap(fit$W, Colv = NA, xlab = "samples", ylab = "factors", margins = c(2,2),'+\
                'labRow = "", labCol = "", scale = "column", col = cm.colors(100));')
    ro.r('dev.off()')
    ro.r('jpeg(file = "' + output_folder + "/" + method + "_" + loss + '_heatmap_H.jpeg")')
    ro.r('heatmap(fit$H, Rowv = NA, ylab = "variables", xlab = "factors", margins = c(2,2),'+\
                'labRow = '', labCol = '', scale = "row", col = cm.colors(100));')
    ro.r('dev.off()')
    print("\n")
    
    loadings = np.asarray(ro.r('t(fit$H)'))
    factor_names = np.asarray([method + str(i) for i in range(loadings.shape[1])])
    print "Factor names are: " + str(factor_names)
    snapshot_names = np.asarray(ro.r('colnames(fit$H)'))
    print "Snapshots are: " + str(snapshot_names)

    # Analyzing scores
    scores = np.asarray(ro.r('fit$W'))
    population_names = np.asarray(ro.r('rownames(fit$W)'))
    print "Population names are: " + str(population_names)
    score_factor_names = np.asarray([method + str(i) for i in range(loadings.shape[1])])
    scores_df = analyze_scores(scores, population_names, score_factor_names, method + "_" + loss)
    
    traffic_matrix = np.asarray(ro.r('x'))
    traffic_base_stations = np.asarray(ro.r('rownames(x)'))
    traffic_snapshots = np.asarray(ro.r('colnames(x)'))
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + loss)
    
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + loss, threshold=0.4, max_factor_value=True)
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + loss, threshold=0.6, max_factor_value=True)
    utilities.generate_signatures(traffic_matrix, traffic_base_stations, \
                                                 traffic_snapshots, loadings, factor_names, label, output_folder, method + "_" + loss, threshold=0.8, max_factor_value=True)

    # Analyzing loadings
    print "Saving loadings to file (" + method + " and loss " + loss + ")...\n"
    loadings_df = analyze_loadings(loadings, factor_names, snapshot_names, method + "_" + loss)

    print "Plotting loadings (" + method + " and loss " + loss + ")...\n"
    plot_factor_loadings(factor_names, loadings_df, method + "_" + loss, False)
    plot_factor_loadings(factor_names, loadings_df, method + "_" + loss, True)
    utilities.plot_time_signatures(loadings_df[factor_names].as_matrix(), loadings_df["snapshot_name"], factor_names, label, output_folder, method + "_" + loss + "/loadings/", period_min)

    return loadings_df, scores_df

def my_transform1(x):
    splitted_x = str(x).split(".")
    return splitted_x[0].strip("X") + "-" + splitted_x[1] + "-" + splitted_x[2]


def my_transform2(x):
    splitted_x = str(x).split(".")
    return splitted_x[4] + ":" + splitted_x[5] + ":" + splitted_x[6]


def analyze_loadings(loadings, factor_names, snapshot_names, folder_prefix):
    # create dataframe reporting for each snapshot the loading associated to each factor
    loadings_df = pd.DataFrame(loadings, columns=factor_names, index=snapshot_names)
    loadings_df = loadings_df.reset_index()
    loadings_df["day"] = loadings_df["index"].apply(my_transform1)
    loadings_df["time"] = loadings_df["index"].apply(my_transform2)
    # print loadings_df.head()
    loadings_filename = output_folder + "/" + folder_prefix + "/loadings" + (
    "_" + label if label is not None else "") + ".csv"
    if not os.path.exists(os.path.dirname(loadings_filename)):
        os.makedirs(os.path.dirname(loadings_filename))
    loadings_df = loadings_df.rename(columns={'index': 'snapshot_name'})
    loadings_df.to_csv(loadings_filename, header=True, sep=",", index=False)
    return loadings_df


def analyze_uniquenesses(uniquenesses, snapshot_names, folder_prefix):
    # create dataframe reporting for each snapshot the loading associated to each factor
    uniquenesses_df = pd.DataFrame(uniquenesses, columns=["uniquenesses"], index=snapshot_names)
    uniquenesses_df = uniquenesses_df.reset_index()
    uniquenesses_df["day"] = uniquenesses_df["index"].apply(my_transform1)
    uniquenesses_df["time"] = uniquenesses_df["index"].apply(my_transform2)
    # print uniquenesses_df.head()
    uniquenesses_filename = output_folder + "/" + folder_prefix + "/uniquenesses" + (
    "_" + label if label is not None else "") + ".csv"
    if not os.path.exists(os.path.dirname(uniquenesses_filename)):
        os.makedirs(os.path.dirname(uniquenesses_filename))
    uniquenesses_df = uniquenesses_df.rename(columns={'index': 'snapshot_name'})
    uniquenesses_df.to_csv(uniquenesses_filename, header=True, sep=",", index=False)
    return uniquenesses_df


def analyze_scores(scores, population_names, score_factor_names, folder_prefix):
    # create dataframe reporting for each snapshot the loading associated to each factor
    scores_df = pd.DataFrame(scores, columns=score_factor_names, index=population_names)
    scores_df = scores_df.reset_index()
    # print loadings_df.head()
    scores_filename = output_folder + "/" + folder_prefix + "/scores" + (
    "_" + label if label is not None else "") + ".csv"
    if not os.path.exists(os.path.dirname(scores_filename)):
        os.makedirs(os.path.dirname(scores_filename))
    scores_df = scores_df.rename(columns={'index': 'cell_id'})
    scores_df.to_csv(scores_filename, header=True, sep=",", index=False)
    return scores_df

def plot_factor_loadings(factor_names, loadings_df, folder_prefix, norm_colors, uniquenesses=False):
    # plot with various axes scales
    num_subplots = len(factor_names)
    page_counter = 1
    max_figs_per_line = 5
    max_rows_per_page = 5
    fig_basename = output_folder + "/" + folder_prefix + ("/loadings/heatmap" if not uniquenesses else "/uniquenesses/heatmap") + ("_" + label if label is not None else "") \
            + ("_colors_normalized" if norm_colors else "")
    
    if not os.path.exists(os.path.dirname(fig_basename)):
        os.makedirs(os.path.dirname(fig_basename))    
    if norm_colors:
        min_loading = loadings_df[factor_names].min().min()
        max_loading = loadings_df[factor_names].max().max()
        print "Min loading is %f" % min_loading
        print "Max loading is %f" % max_loading
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
        temp_df = loadings_df.pivot(index='day', columns='time', values=factor)
        index = sorted(temp_df.index, reverse=True)
        temp_df = temp_df.reindex(index)
        row_labels = temp_df.index.values.tolist()
        col_labels = temp_df.columns.values.tolist()
        if norm_colors:
            ms = axs[counter].matshow(temp_df.as_matrix(), vmin=min_loading, vmax=max_loading, cmap="jet")
        else:
            ms = axs[counter].matshow(temp_df.as_matrix(), cmap="jet")
            divider = make_axes_locatable(axs[counter])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(ms, cax=cax)
        axs[counter].xaxis.set_ticks_position('bottom')
        axs[counter].set_xticks(range(len(col_labels)))
        xtick_labels = [""] * len(col_labels)
        col_labels = [hourtime[0:5] for hourtime in col_labels]
        xtick_labels = [(col_labels[i] if (i % 4 == 0 or i == len(col_labels) - 1) else "") for i in
                        range(len(col_labels))]
        axs[counter].set_xticklabels(xtick_labels, rotation=90)
        axs[counter].set_yticks(range(len(row_labels)))
        y_labels = [""] * len(row_labels)
        if is_typical_week:
            row_labels = [datetime.strptime(date_string, '%Y-%m-%d').strftime('%A') for date_string in row_labels]
            print row_labels
        else:
            row_labels[0] = row_labels[0] + " (" + (datetime.strptime(row_labels[0], '%Y-%m-%d').strftime('%A'))[0:3] + ")"
            if len(row_labels) > 1:
                row_labels[len(row_labels) - 1] = row_labels[len(row_labels) - 1] + " (" + (datetime.strptime(row_labels[len(row_labels) - 1], '%Y-%m-%d').strftime('%A'))[0:3] + ")"
        y_labels = [(row_labels[len(row_labels) - 1 - i] if (i % 7 == 0 or i == len(row_labels) - 1) else "") for i in
                    range(len(row_labels))]
        y_labels.reverse()
        axs[counter].set_yticklabels(y_labels)
        axs[counter].set_title(factor)
        counter += 1

    while counter < len(axs):
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

'''
def z_score_normalize_cell_activity(cells_df):
    temp_df_mean = cells_df[["cell-id", "total_volume_per_day_of_the_week"]].groupby("cell-id").mean()
    temp_df_mean = temp_df_mean.rename(columns={'total_volume_per_day_of_the_week': 'mean'})
    temp_df_mean = temp_df_mean.reset_index()
    cells_df = pd.merge(cells_df, temp_df_mean, on='cell-id')
    temp_df_std = cells_df[["cell-id", "total_volume_per_day_of_the_week"]].groupby("cell-id").std()
    temp_df_std = temp_df_std.rename(columns={'total_volume_per_day_of_the_week': 'std'})
    temp_df_std = temp_df_std.reset_index()
    cells_df = pd.merge(cells_df, temp_df_std, on='cell-id')
    cells_df["total_volume_per_day_of_the_week"] = (cells_df["total_volume_per_day_of_the_week"] - cells_df["mean"]) / \
                                                   cells_df["std"]
    cells_df.loc[np.isnan(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df.loc[np.isinf(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df = cells_df[
        ["#day_of_the_week", "fake_date", "hour", "cell-id", "total_volume_per_day_of_the_week"]].sort_values(
        by=["fake_date", "hour", "cell-id"])
    new_filename = os.path.splitext(input_file)[0] + "_z_score_norm" + os.path.splitext(input_file)[1]
    cells_df.to_csv(new_filename, index=False, sep=" ")
    return cells_df


def min_max_normalize_cell_activity(cells_df):
    temp_df_max = cells_df[["cell-id", "total_volume_per_day_of_the_week"]].groupby("cell-id").max()
    temp_df_max = temp_df_max.rename(columns={'total_volume_per_day_of_the_week': 'max'})
    temp_df_max = temp_df_max.reset_index()
    cells_df = pd.merge(cells_df, temp_df_max, on='cell-id')
    temp_df_min = cells_df[["cell-id", "total_volume_per_day_of_the_week"]].groupby("cell-id").min()
    temp_df_min = temp_df_min.rename(columns={'total_volume_per_day_of_the_week': 'min'})
    temp_df_min = temp_df_min.reset_index()
    cells_df = pd.merge(cells_df, temp_df_min, on='cell-id')
    cells_df["total_volume_per_day_of_the_week"] = (cells_df["total_volume_per_day_of_the_week"] - cells_df["min"]) / (
    cells_df["max"] - cells_df["min"])
    cells_df.loc[np.isnan(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df.loc[np.isinf(cells_df["total_volume_per_day_of_the_week"]), "total_volume_per_day_of_the_week"] = 0
    cells_df = cells_df[
        ["#day_of_the_week", "fake_date", "hour", "cell-id", "total_volume_per_day_of_the_week"]].sort_values(
        by=["fake_date", "hour", "cell-id"])
    new_filename = os.path.splitext(input_file)[0] + "_min_max_norm" + os.path.splitext(input_file)[1]
    cells_df.to_csv(new_filename, index=False, sep=" ")
    return cells_df
'''
    
locale.setlocale(locale.LC_ALL, "C")
# read arguments
parser = argparse.ArgumentParser(prog='exploratory_factor_analysis_network_profiles',
                                 usage='%(prog)s -r start_date1 end_date1 -r start_date2 end_date2 ... -i input_file -a antennas_file -n True/False')
parser.add_argument('-i', '--input_file')
parser.add_argument('-o', '--output_folder')
parser.add_argument('-l', '--output_label', default=None)
parser.add_argument('-f', '--num_factors', default="auto")
parser.add_argument('-w', '--typical_week', action='store_true', default=False)
#parser.add_argument('-zn', '--z_score_normalize_cells', action='store_true', default=False)
#parser.add_argument('-mmn', '--min_max_normalize_cells', action='store_true', default=False)
parser.add_argument('-cov', '--covariance', action='store_true', default=False)
parser.add_argument ('-p', '--period_min', default=60)

cor = not (parser.parse_args().covariance)

input_file = parser.parse_args().input_file
label = parser.parse_args().output_label
period_min = int(parser.parse_args().period_min)
# lower_value = parser.parse_args().lower_value
# norm = parser.parse_args().normalized
is_typical_week = parser.parse_args().typical_week
# nstart = parser.parse_args().start_values
output_folder = parser.parse_args().output_folder
snapshots_filename = output_folder + "/cells_matrix_for_network_profiling" + (
"_" + label if label is not None else "") + ".dat"

# read traffic file#day_of_the_week fake_date hour cell-id area total_volume_per_day_of_the_week
NAMES = ["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "category", "total_volume_per_day_of_the_week"]
cells_df = pd.read_csv(input_file, sep=" ", header=0,
                       names=NAMES)
cells_df = cells_df.groupby(["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area"]).sum()
cells_df = cells_df.reset_index()
cells_df = cells_df[["#day_of_the_week", "fake_date", "hour", "cell-id", "lon", "lat", "area", "total_volume_per_day_of_the_week"]]
print cells_df.head()
'''if parser.parse_args().min_max_normalize_cells:
    cells_df = min_max_normalize_cell_activity(cells_df)
elif parser.parse_args().z_score_normalize_cells:
    cells_df = z_score_normalize_cell_activity(cells_df)'''

cells_df["time"] = cells_df["fake_date"] + "." + cells_df["#day_of_the_week"].map(str) + "," + cells_df["hour"].map(str)
# if norm:
#    temp_df = cells_df[["time", "total_volume_per_day_of_the_week"]].groupby("time").sum()
#    temp_df = temp_df.rename(columns = {'total_volume_per_day_of_the_week':'total_snapshot_volume'})
#    temp_df = temp_df.reset_index()
# print temp_df
#    cells_df = pd.merge(cells_df, temp_df, on = 'time')
#    cells_df['total_volume_per_day_of_the_week'] = cells_df['total_volume_per_day_of_the_week']/cells_df['total_snapshot_volume']
# print cells_df

# print cells_df.head()
cells_df = cells_df.pivot(index='cell-id', columns='time', values='total_volume_per_day_of_the_week')
# print cells_df.head()

if not os.path.exists(os.path.dirname(snapshots_filename)):
    os.makedirs(os.path.dirname(snapshots_filename))
cells_df.to_csv(snapshots_filename, index=True, header=True, sep=" ", index_label=False)

importr('psych')
importr('nFactors')
importr('NNLM')

# read data matrix X
ro.r('x<-rbind(as.matrix(read.table(file="' + snapshots_filename + '", sep=" ", header=TRUE, row.names=1)))')

if parser.parse_args().num_factors == "auto":
    output_file = output_folder + "/scree_plot.pdf"
    ro.r('pdf("' + output_file + '")')
    ro.r('fa_score <- fa.parallel(x, fa = "fa")')
    ro.r('dev.off()')
    nfact = str(ro.r('fa_score$nfact')[0])  # get eigenvalues
    print "\n\nBest number of factors:\n%s" % nfact
else:
    nfact = parser.parse_args().num_factors
    print "\n\nUsing %s as the number of factors." % nfact


LOSS_FUNCTIONS = ["mse", "mkl"]
METHODS = ["lee", "scd"]
for method in METHODS:
    for loss in LOSS_FUNCTIONS:
        try:
            nnmf(nfact, loss=loss, method=method, period_min=period_min)
        except rinterface.RRuntimeError:
            print "***ERROR while performing nnmf with method " + method + " and loss " + loss
            
ROTATIONS = ["varimax", "oblimin", "geominQ"]
METHODS = ["minres", "ml"]
for method in METHODS:
    for rotation in ROTATIONS:
        try:
            efa(nfact, rotation=rotation, method=method, cor=cor, period_min=period_min)
        except rinterface.RRuntimeError:
            print "***ERROR while performing fa with method " + method + " and rotation " + rotation
#efa(snapshots_filename, nfact, rotation="varimax", method="pa", cor=cor)
#efa(snapshots_filename, nfact, rotation="promax", method="ml", cor=cor)

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
#efa(snapshots_filename, nfact, rotation="oblimin", method="ml", cor=cor)
'''