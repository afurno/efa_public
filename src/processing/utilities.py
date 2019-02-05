import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import stats
from datetime import datetime as dt
from matplotlib.dates import  DateFormatter, DayLocator, WeekdayLocator, MONDAY

def plotTimeSeries(the_series, the_series_names, time_slots_per_day, normalized_y=False, show_names=True, ax=None, time_cut_off=None, min_y=None, max_y=None, std_dev_the_weeks=None):
    ''' Plot the activity of the "sample" series.

    Parameters:
        sample -- A list indicating which series we want in the plot
        the_series -- A dict containing all the series
        normalized_y -- A boolean indicatin whether the series shold be normalized or not
        show_names -- A boolean indicating whether there should be a leged with the the names of 
                    the series in the plot
        ax -- A matplotlib.axis 
        time_cut_off -- An integer indicating whether the series should be cut off at some point before the plot
    '''  
    # print "Number of time slots per day: " + str(time_slots_per_day)
    xt = [i for i in range(len(the_series))]
    # print "Here comes the xt values: " + str(xt)
    if not ax:
        fig = plt.figure(); 
        fig.set_size_inches(8, 7) 
        ax = plt.axes();
    
    ax2 = ax.twiny()
    ax2.xaxis.grid(True, which='major', color='gray', linestyle='--', alpha=0.4)
    
    plot_label_each_n_hours = 12
    xt_period = plot_label_each_n_hours * (time_slots_per_day / 24.)
    
    x2_ticks = [xt[i] for i in range(0, len(the_series)) if (i % xt_period) == 0]
    x2_ticks.append(len(the_series))
    ax2.set_xticks(x2_ticks)
    ax2.tick_params(axis='x', which='major', labelsize=11) 
    x2_ticks_labels = ["%.2d" % (x2_tick / (time_slots_per_day / 24.) % 24) for x2_tick in x2_ticks]
    # print "Here comes the x2 ticks: " + str(x2_ticks)
        
    the_tick_labels = ax2.set_xticklabels(x2_ticks_labels) 
    plt.setp(the_tick_labels, rotation=90)
    
    if normalized_y:
        if np.all(the_series[0] == the_series):
            the_series = np.zeros(len(the_series))
        else:
            the_series = stats.zscore(the_series)
    base_line, = ax.plot(xt, the_series, linewidth=2)
    the_series_names = [(datetime.datetime.strptime(the_series_names[i], 'X%Y.%m.%d.%A.%H.%M.%S').strftime('%Y-%m-%d.%A,%H:%M:%S')\
                        if(the_series_names[i].startswith("X")) else the_series_names[i]) for i in range(len(the_series_names))]
        
    pairs = [(xt[i], the_series_names[i].split(".")[0]) for i in range(0, len(the_series)) if the_series_names[i].split(",")[1] == "00:00:00"]
    x1_ticks = [x[0] for x in pairs]
    x1_ticks.append(len(the_series))
    ax.set_xticks(x1_ticks)
    ax.tick_params(axis='x', which='major', labelsize=11) 
    x1_ticks_labels = [x[1] for x in pairs]
    x1_ticks_labels.append("")
    # print "Here comes the pairs: " + str(pairs)
    # print "Here comes the x1 ticks: " + str(x1_ticks)
    the_tick_labels = ax.set_xticklabels(x1_ticks_labels) 
    plt.setp(the_tick_labels, rotation=90)

    ax.set_xlim([0, np.ceil(max(xt))])
    if min_y != None and max_y != None:
        ax.set_ylim([min_y, max_y])
    base_c = base_line.get_color()
    if std_dev_the_weeks is not None:
        ax.fill_between(xt, np.array(the_series) - np.array(std_dev_the_weeks), np.array(the_series) + np.array(std_dev_the_weeks), alpha=0.4, edgecolor=base_c, facecolor=base_c)

def cdfPlot(data, col, lab, fig=None, ax=None):
    ''' plot the cdf of the data.

    Parameters
        data -- a list
        ax -- matplotlib.axis object
    '''

    if not ax:
        fig = plt.figure(); 
        ax = plt.axes();
    pdf, bins = np.histogram(data, bins=30000, normed=True)
    bins = bins[:-1]
    N = sum(pdf)
    cdf = np.cumsum(pdf) / N;
    ax.plot(bins, cdf, col, linewidth=5, markersize=10, label=lab)
    ax.set_ylabel('CDF')
    return fig, ax

import datetime

def is_weekend(date_timestamp):
    days_of_the_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_of_the_week = date_timestamp.strftime("%A")
    return (day_of_the_week == days_of_the_week[5] or day_of_the_week == days_of_the_week[6])

def check_timestamp(date_timestamp, ranges, aggregation_type):
    if aggregation_type == "weekdays" and is_weekend(date_timestamp):
        return False
    elif aggregation_type == "weekends" and not is_weekend(date_timestamp):
        return False
    else:
        for (start_date, end_date) in ranges:
            if date_timestamp >= datetime.datetime.strptime(start_date, "%Y-%m-%d") and date_timestamp <= datetime.datetime.strptime(end_date, "%Y-%m-%d"):
                return True
        return False
    
def is_within(date_string, ranges):    
    date_timestamp = datetime.datetime.strptime(date_string, "%Y-%m-%d")
    return check_timestamp(date_timestamp, ranges, "all")

def generate_signatures(traffic_matrix, traffic_rows_keys, traffic_column_keys, the_loadings, factor_names, label, output_folder, folder_prefix, threshold=None, max_factor_value=None):
    traffic_matrix = np.array(traffic_matrix, copy=True)  
    the_loadings = np.array(the_loadings, copy=True)  
    if threshold is not None:
        low_values_flags = (the_loadings < threshold) if max_factor_value is None else (the_loadings < (threshold*np.max(the_loadings, axis=0,keepdims=True)))
        the_loadings[low_values_flags] = 0
    product_matrix = np.dot(traffic_matrix, the_loadings)
    signatures_df = pd.DataFrame(product_matrix, columns=factor_names, index=traffic_rows_keys)
    signatures_df = signatures_df.reset_index()
    
    output_path = output_folder + "/" + folder_prefix + "/signatures/non_normalized" + ("_threshold_" + str(threshold) if threshold is not None else "") + "/"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    signatures_filename = output_path + ("signatures_" + label if label is not None else "") + ".csv"
    signatures_df.to_csv(signatures_filename, header=True, sep=",", index=False)
    
    norm_signatures_df = signatures_df.copy(deep=True)
    cols = list(norm_signatures_df.columns)
    for col in cols:
        if not col in factor_names:
            continue
        # print norm_signatures_df[col]
        if np.all(signatures_df[col] == 0):
            norm_signatures_df[col] = np.zeros(len(norm_signatures_df[col]))
        else:
            norm_signatures_df[col] = stats.zscore(norm_signatures_df[col])
            
    output_path = output_folder + "/" + folder_prefix + "/signatures/normalized" + ("_threshold_" + str(threshold) if threshold is not None else "") + "/"
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    signatures_filename = output_path + ("norm_signatures_" + label if label is not None else "") + ".csv"
    norm_signatures_df.to_csv(signatures_filename, header=True, sep=",", index=False)
    
    return product_matrix, signatures_df, norm_signatures_df

def plot_factor_signature_on_multiple_pages(the_weeks, the_weeks_dates, basepath, label, TIME_SLOTS_PER_DAY, normalized_y=False, threshold=None, std_dev_the_weeks=None):
    ''' Plot all the cluster signatures in a single plot '''
    max_figs_per_line = 5
    max_rows_per_page = 5
    rows = len(the_weeks) / max_figs_per_line;
    if len(the_weeks) % max_figs_per_line > 0:
        rows += 1;
    page_counter = 1
    week_counter = 0
    f = None
    total_min_y, total_max_y = None, None
    for _, the_week_series in the_weeks.items():
        if normalized_y:
            if np.all(the_week_series[0] == the_week_series):
                the_week_series = np.zeros(len(the_week_series))
            else:
                the_week_series = stats.zscore(the_week_series)
    
        week_min = min(the_week_series)
        week_max = max(the_week_series)
        total_min_y = week_min if (total_min_y is None or total_min_y < week_min) else total_min_y
        total_max_y = week_max if (total_max_y is None or total_max_y < week_max) else total_max_y
    
    for week_number in sorted(the_weeks.keys()):
        if week_counter % (max_figs_per_line * max_rows_per_page) == 0:
            if f != None:
                plt.tight_layout()
                plt.savefig('%s%s_all_weeks_page_%02d.pdf' % (basepath + ('/normalized' if normalized_y else '/non_normalized') + ("_threshold_" + str(threshold) if threshold is not None else "") + "/", label, page_counter))
                plt.close()
                page_counter += 1
            f, ax_array = plt.subplots(max_rows_per_page, max_figs_per_line)
            # increase the size of the figure such that the x-axis is 4 times bigger; we assume current value of x-axis is 6    
            f.set_size_inches(6 * max_figs_per_line, 4 * max_rows_per_page)

        if ax_array.ndim > 1:
            x = (week_counter % (max_figs_per_line * max_rows_per_page)) / max_figs_per_line;
            y = week_counter % max_figs_per_line;
            ax = ax_array[x][y];
        else:
            ax = ax_array[week_counter]

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        # print "Here comes the weeks: " + str(the_weeks)
        # print "Here comes the week names: " + str(the_weeks_dates)
        #print "***\nPlotting series for week " + str(week_number)
        #print "The series is: " + str(the_weeks[week_number].tolist())
        #print "The days are: " + str(the_weeks_dates[week_number].tolist())
        #print "***"
        plotTimeSeries(the_weeks[week_number], the_weeks_dates[week_number], TIME_SLOTS_PER_DAY, normalized_y=normalized_y, \
                        ax=ax, min_y=total_min_y, max_y=total_max_y, std_dev_the_weeks=std_dev_the_weeks[week_number])
        ax.set_xlabel('Week %d' % (week_number))
        ax.xaxis.set_label_coords(0.5, 0.96)
        week_counter += 1
    i = 0
    while week_counter < page_counter * (max_figs_per_line * max_rows_per_page):
        if ax_array.ndim > 1:
            x = (week_counter % (max_figs_per_line * max_rows_per_page)) / max_figs_per_line;
            y = week_counter % max_figs_per_line;
            ax_array[x][y].axis('off');
        else:
            ax_array[i].axis('off')
            i += 1
        week_counter += 1
    if f != None:
        plt.tight_layout()
        plt.savefig('%s%s_all_weeks_page_%02d.pdf' % (basepath + ('/normalized' if normalized_y else '/non_normalized') + ("_threshold_" + str(threshold) if threshold is not None else "") + "/", label, page_counter))
        plt.close()

def break_into_multiple_weeks(factor_series, factor_series_names, TIME_SLOTS_PER_DAY):
    offset = 0
    found = False
    i = 0
    while i < len(factor_series) and not found:
        if factor_series_names[i].startswith("X"):
            factor_series_names[i] = datetime.datetime.strptime(factor_series_names[i], 'X%Y.%m.%d.%A.%H.%M.%S').strftime('%Y-%m-%d.%A,%H:%M:%S')
        
        timestamp = dt.strptime(factor_series_names[i], '%Y-%m-%d.%A,%H:%M:%S')
        if timestamp.strftime("%A") == "Monday":
            found = True
        else:
            offset = 1
            i += 1
    splitted_weeks = [factor_series[i + k : i + k + 7 * TIME_SLOTS_PER_DAY] \
                     for k in range(0, len(factor_series[i:]), 7 * TIME_SLOTS_PER_DAY)]
    splitted_weeks_dates = [factor_series_names[i + k : i + k + 7 * TIME_SLOTS_PER_DAY] \
                     for k in range(0, len(factor_series[i:]), 7 * TIME_SLOTS_PER_DAY)]
    the_weeks = {offset + index : j for index, j in enumerate(splitted_weeks)}
    the_weeks_dates = {offset + index : j for index, j in enumerate(splitted_weeks_dates)}
    if offset > 0:
        the_weeks[0] = factor_series[0:i]
        the_weeks_dates[0] = factor_series_names[0:i]
    return the_weeks, the_weeks_dates

def plotAverageSeries(avg_cell, std_cell, time_slots_per_day, example_name=None, show_names=True, ax=None, time_cut_off=None, min_y=None, max_y=None):
    ''' Plot the activity of the "sample" series.

    Parameters:
        sample -- A list indicating which series we want in the plot
        the_series -- A dict containing all the series
        normalized_y -- A boolean indicatin whether the series shold be normalized or not
        show_names -- A boolean indicating whether there should be a leged with the the names of 
                    the series in the plot
        ax -- A matplotlib.axis 
        time_cut_off -- An integer indicating whether the series should be cut off at some point before the plot
    '''  
    xt = [i / float(time_slots_per_day) for i in range(len(avg_cell))]
    if min_y == None:
        min_y = np.amin(avg_cell) - np.amax(std_cell)
    if max_y == None:
        max_y = np.amax(avg_cell) + np.amax(std_cell)  
    
    plot_second_grid = False
    if not ax:
        fig = plt.figure(); 
        fig.set_size_inches(8, 7) 
        ax = plt.axes();
        # plot_second_grid = True
        
    ax.set_ylim([min_y, max_y])      
    
    ax2 = ax.twiny()
    # ax2.set_xlabel('Hour of the day', labelpad=9, fontsize=14)
    ax2.xaxis.grid(True, which='major', color='gray', linestyle='--', alpha=0.4)
    if plot_second_grid:
        ax2.xaxis.grid(True, which='minor', color='red', linestyle=':', alpha=0.4)
    
    MULTIPLE_LABEL = False
    if time_slots_per_day < 24:
        label_period = 2
        label_base = 24 / time_slots_per_day
        MULTIPLE_LABEL = True
        divisive_factor = 1
    else:
        label_period = (6 * time_slots_per_day / 24.)
        label_base = 1
        divisive_factor = time_slots_per_day / 24.
    
    x2_ticks = [xt[i] for i in range(0, len(avg_cell)) if (i % label_period) == 0]
    x2_ticks.append(len(avg_cell) / float(time_slots_per_day))
    ax2.set_xticks(x2_ticks)
    ax2.tick_params(axis='x', which='major', labelsize=11) 
    if MULTIPLE_LABEL:
        x2_ticks_labels = [("%.2d-%.2d" % ((label_base * i / divisive_factor) % 24, (label_base * i + label_base) % 24) )for i in range(0, len(avg_cell)) if (i % label_period) == 0]
    else:
        x2_ticks_labels = [("%.2d" % ((label_base * i / divisive_factor) % 24)) for i in range(0, len(avg_cell)) if (i % label_period) == 0]
        
    the_tick_labels = ax2.set_xticklabels(x2_ticks_labels) 
    plt.setp(the_tick_labels, rotation=90)
    
    x2_minor_ticks = [xt[i] for i in range(0, len(avg_cell)) if (i % (time_slots_per_day / 24.)) == 0]
    x2_minor_ticks.append(len(avg_cell) / float(time_slots_per_day))
    ax2.set_xticks(x2_minor_ticks, minor=True)
    ax2.tick_params(axis='x', which='minor', labelsize=3)
    
    if example_name and type(example_name) == str:
        base_line, = ax.plot(xt, avg_cell, linewidth=2, label=example_name)
    else:
        base_line, = ax.plot(xt, avg_cell, linewidth=2)
        
    base_c = base_line.get_color()
    # plot standard deviation 
    ax.fill_between(xt, np.array(avg_cell) - np.array(std_cell), np.array(avg_cell) + np.array(std_cell), alpha=0.4, edgecolor=base_c, facecolor=base_c)
    
    # ax.set_xlabel('Day of the week', fontsize=18)
    x_ticks_labels = ['', '', '', '', '', '', '']
    ax.set_xticklabels(x_ticks_labels)
    
    x_minor_ticks = np.array([i for i in range(len(x_ticks_labels))]) + 0.5
    x_minor_ticks_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_xticklabels(x_minor_ticks_labels, minor=True)
    ax.tick_params(axis='x', which='minor', labelsize=16) 
        
    # ax.set_ylabel('Normalized SMS & Call activity', fontsize=18)
    ax.set_xlim([0, np.ceil(max(xt))])
    
    ax.xaxis.grid(True, which='major', color='gray', linestyle='-', alpha=1.0)
    # ax2.xaxis.set_label_coords(0.5, 0.96)
    if show_names and example_name:
        ax.legend(loc=1)
    
def compute_average_series(the_series, normalized_y=True):
    first_element_len = len(the_series[0])
    num_series = len(the_series)
    arr = np.ma.empty((num_series,first_element_len))
    arr.mask = True
    for i in range(len(the_series)):
        y_cell = the_series[i]
        if normalized_y:
            if np.all(y_cell[0] == y_cell):
                y_cell = np.zeros(len(y_cell))
            else:
                y_cell = stats.zscore(y_cell)
        arr[i,:y_cell.shape[0]] = y_cell
    avg_cell = arr.mean(axis=0)
    std_cell = arr.std(axis=0)
    return avg_cell, std_cell

def plot_time_signatures(signatures, signatures_names, factor_names, label, output_folder, folder_prefix, period_min, \
                         normalized_y=False, threshold=None, max_factor_value=None, std_dev_signatures=None):
    TIME_SLOTS_PER_DAY = 24 * 60 / period_min
    the_path = ""
    if normalized_y:
        the_path = output_folder + "/" + folder_prefix + "/normalized" + ("_threshold_" + str(threshold) if threshold is not None else "") + "/"
        if not os.path.exists(os.path.dirname(the_path)):
            os.makedirs(os.path.dirname(the_path))
    else:
        the_path = output_folder + "/" + folder_prefix + "/non_normalized" + ("_threshold_" + str(threshold) if threshold is not None else "") + "/"
        if not os.path.exists(the_path):
            os.makedirs(the_path)
    
    df = pd.DataFrame()
    data_dictionary = {}
    for i in range(len(factor_names)):
        factor_name = factor_names[i]
        if len(signatures.shape) == 1:
            factor_series = signatures
        else:
            factor_series = signatures[:, i]
        #print "Plotting series for " + factor_name
        factor_series_names = signatures_names.values
        the_weeks, the_weeks_dates = break_into_multiple_weeks(factor_series, factor_series_names, TIME_SLOTS_PER_DAY)
        if std_dev_signatures is not None:
            if len(std_dev_signatures.shape) == 1:
                std_dev_factor_series = std_dev_signatures
            else:
                std_dev_factor_series = std_dev_signatures[:, i]
            std_dev_the_weeks, the_weeks_dates = break_into_multiple_weeks(std_dev_factor_series, factor_series_names, TIME_SLOTS_PER_DAY)
        else:
            std_dev_the_weeks = None
            
        plot_factor_signature_on_multiple_pages(the_weeks, the_weeks_dates, output_folder + "/" + folder_prefix, \
                factor_name + "_" + label if label is not None else "", TIME_SLOTS_PER_DAY, normalized_y=normalized_y, threshold=threshold, std_dev_the_weeks = std_dev_the_weeks)
        
        avg_series, dev_series = compute_average_series(the_weeks, normalized_y=normalized_y)
        plotAverageSeries(avg_series, dev_series, TIME_SLOTS_PER_DAY)
        the_whole_series = factor_series if not normalized_y else (np.zeros(len(factor_series)) \
            if np.all(factor_series[0] == factor_series) else stats.zscore(factor_series))
        the_whole_series = the_whole_series.tolist()
        
        if std_dev_signatures is not None:
            the_whole_std_dev_series = std_dev_factor_series if not normalized_y else (np.zeros(len(std_dev_factor_series)) \
                if np.all(std_dev_factor_series[0] == std_dev_factor_series) else stats.zscore(std_dev_factor_series))
            the_whole_std_dev_series = the_whole_std_dev_series.tolist()
            temp_dict = {"factor":factor_name, "whole_series" : the_whole_series, \
                "avg_series":avg_series.tolist(), "std_dev" : dev_series.tolist(), \
                "whole_series_x_axis":np.concatenate([v for (_, v) in sorted(the_weeks_dates.items())]).ravel().tolist(),
                "whole_series_std_dev":the_whole_std_dev_series}
        else:
            temp_dict = {"factor":factor_name, "whole_series" : the_whole_series, \
                "avg_series":avg_series.tolist(), "std_dev" : dev_series.tolist(), \
                "whole_series_x_axis":np.concatenate([v for (_, v) in sorted(the_weeks_dates.items())]).ravel().tolist()}
        
        df = df.append(temp_dict, ignore_index=True)
        data_dictionary[factor_name] = temp_dict
        plt.tight_layout()
        plt.savefig(('%s.pdf') % (the_path + "avg_signature_" + factor_name + "_" + label if label is not None else "" ))
        plt.close()
    df = df.sort_values(by=["factor"])
    df[["factor", "whole_series", "whole_series_x_axis", "avg_series", "std_dev"]].to_csv(the_path + "csv_data_" + (label if label is not None else "") + ".csv", index=False)
    return data_dictionary

def plot_factor_info(factor_name, the_dates, signature, signature_avg, signature_std, \
                                       signature_th04, signature_th04_avg, signature_th04_std,\
                                       signature_th06, signature_th06_avg, signature_th06_std,\
                                       signature_th08, signature_th08_avg, signature_th08_std,\
                                       scores, scores_avg, scores_std, \
                                       basepath, TIME_SLOTS_PER_DAY, plot_scores=True, \
                                       whole_series_std_dev=None, \
                                       whole_series_std_dev_th04=None, \
                                       whole_series_std_dev_th06=None, \
                                       whole_series_std_dev_th08=None):
    ''' Plot all the cluster signatures in a single plot '''
    max_figs_per_line = 4
    if plot_scores:
        max_rows_per_page = 7
    else:       
        max_rows_per_page = 5
    f = plt.figure()
    f.set_size_inches(10 * max_figs_per_line, 4 * max_rows_per_page)
    
    ax = plt.subplot(max_rows_per_page, 1, 1)
    base_line, = ax.plot_date(x=the_dates, y=signature, fmt="-", linewidth=2)  
    ax.xaxis.grid(True, which='major', color='gray', linestyle='-', alpha=1.0)
    ax.xaxis.grid(True, which='minor', color='gray', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d (%a)') )
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MONDAY))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.set_xlim([the_dates[0], the_dates[-1]])
    ax.set_xlabel('Signature')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    if whole_series_std_dev is not None:
        base_c = base_line.get_color()
        ax.fill_between(the_dates, np.array(signature) - np.array(whole_series_std_dev), np.array(signature) + np.array(whole_series_std_dev), alpha=0.4, edgecolor=base_c, facecolor=base_c)
    
    
    ax = plt.subplot(max_rows_per_page, 1, 2)
    base_line, = ax.plot_date(x=the_dates, y=signature_th04, fmt="-", linewidth=2)  
    ax.xaxis.grid(True, which='major', color='gray', linestyle='-', alpha=1.0)
    ax.xaxis.grid(True, which='minor', color='gray', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d (%a)') )
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MONDAY))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.set_xlim([the_dates[0], the_dates[-1]])
    ax.set_xlabel('Signature with th. 0.4')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    if whole_series_std_dev is not None:
        base_c = base_line.get_color()
        ax.fill_between(the_dates, np.array(signature_th04) - np.array(whole_series_std_dev_th04), np.array(signature_th04) + np.array(whole_series_std_dev_th04), alpha=0.4, edgecolor=base_c, facecolor=base_c)

    
    ax = plt.subplot(max_rows_per_page, 1, 3)
    ax.plot_date(x=the_dates, y=signature_th06, fmt="-", linewidth=2)  
    ax.xaxis.grid(True, which='major', color='gray', linestyle='-', alpha=1.0)
    ax.xaxis.grid(True, which='minor', color='gray', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d (%a)') )
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MONDAY))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.set_xlim([the_dates[0], the_dates[-1]])
    ax.set_xlabel('Signature with th. 0.6')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    if whole_series_std_dev is not None:
        base_c = base_line.get_color()
        ax.fill_between(the_dates, np.array(signature_th06) - np.array(whole_series_std_dev_th06), np.array(signature_th06) + np.array(whole_series_std_dev_th06), alpha=0.4, edgecolor=base_c, facecolor=base_c)

    
    ax = plt.subplot(max_rows_per_page, 1, 4)
    ax.plot_date(x=the_dates, y=signature_th08, fmt="-", linewidth=2)  
    ax.xaxis.grid(True, which='major', color='gray', linestyle='-', alpha=1.0)
    ax.xaxis.grid(True, which='minor', color='gray', linestyle='--', alpha=0.4)
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d (%a)') )
    ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MONDAY))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.set_xlim([the_dates[0], the_dates[-1]])
    ax.set_xlabel('Signature with th. 0.8')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    if whole_series_std_dev is not None:
        base_c = base_line.get_color()
        ax.fill_between(the_dates, np.array(signature_th08) - np.array(whole_series_std_dev_th08), np.array(signature_th08) + np.array(whole_series_std_dev_th08), alpha=0.4, edgecolor=base_c, facecolor=base_c)

    
    ax = plt.subplot(max_rows_per_page, max_figs_per_line, 17)
    plotAverageSeries(signature_avg, signature_std, TIME_SLOTS_PER_DAY, ax=ax)
    ax.set_xlabel('Average Signature')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    ax = plt.subplot(max_rows_per_page, max_figs_per_line, 18)
    plotAverageSeries(signature_th04_avg, signature_th04_std, TIME_SLOTS_PER_DAY, ax=ax)
    ax.set_xlabel('Average Signature with th. 0.4')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    ax = plt.subplot(max_rows_per_page, max_figs_per_line, 19)
    plotAverageSeries(signature_th06_avg, signature_th06_std, TIME_SLOTS_PER_DAY, ax=ax)
    ax.set_xlabel('Average Signature with th. 0.6')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    ax = plt.subplot(max_rows_per_page, max_figs_per_line, 20)
    plotAverageSeries(signature_th08_avg, signature_th08_std, TIME_SLOTS_PER_DAY, ax=ax)
    ax.set_xlabel('Average Signature with th. 0.8')
    ax.xaxis.set_label_coords(0.5, 0.96)
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(18)
    
    if plot_scores:
        ax = plt.subplot(max_rows_per_page, 1, 6)
        ax.plot_date(x=the_dates, y=scores, fmt="-", linewidth=2)  
        ax.xaxis.grid(True, which='major', color='gray', linestyle='-', alpha=1.0)
        ax.xaxis.grid(True, which='minor', color='gray', linestyle='--', alpha=0.4)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d (%a)') )
        ax.xaxis.set_major_locator(WeekdayLocator(byweekday=MONDAY))
        ax.xaxis.set_minor_locator(DayLocator())
        ax.set_xlim([the_dates[0], the_dates[-1]])
        ax.set_xlabel('Scores')
        ax.xaxis.set_label_coords(0.5, 0.96)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)
        
        ax = plt.subplot(max_rows_per_page, max_figs_per_line, 25)
        plotAverageSeries(scores_avg, scores_std, TIME_SLOTS_PER_DAY, ax=ax)
        ax.set_xlabel('Average Scores')
        ax.xaxis.set_label_coords(0.5, 0.96)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(18)

    plt.tight_layout()
    if plot_scores:
        plt.savefig('%s%s_signatures_and_scores.pdf' % (basepath, factor_name))
    else:
        plt.savefig('%s%s_signatures.pdf' % (basepath, factor_name))
    plt.close()
        
def plot_data_by_factor(the_dictionary, factor_names, label, output_folder, folder_prefix, period_min, plot_whole_series_std_dev=False):
    TIME_SLOTS_PER_DAY = 24 * 60 / period_min
    plot_scores=True
    the_path = output_folder + "/" + folder_prefix + "/non_normalized/"
    if not os.path.exists(os.path.dirname(the_path)):
        os.makedirs(os.path.dirname(the_path))

    the_normalized_path = output_folder + "/" + folder_prefix + "/normalized/"
    if not os.path.exists(the_normalized_path):
        os.makedirs(the_normalized_path)
    for factor in factor_names:
        #print "Analyzing factor " + factor + "..."
        signature = the_dictionary["signatures"][factor]["whole_series"]
        #print "Analyzing dictionary " + str(the_dictionary) + "..."
        if plot_whole_series_std_dev:
            whole_series_std_dev = the_dictionary["signatures"][factor]["whole_series_std_dev"]
        
        signature_avg = the_dictionary["signatures"][factor]["avg_series"]
        signature_std = the_dictionary["signatures"][factor]["std_dev"]
        whole_series_x_axis = the_dictionary["signatures"][factor]["whole_series_x_axis"]
        whole_series_x_axis = [dt.strptime(x, '%Y-%m-%d.%A,%H:%M:%S') for x in whole_series_x_axis]
        if factor in the_dictionary["signatures_th_0.4"]:
            signature_th04 = the_dictionary["signatures_th_0.4"][factor]["whole_series"]
            if plot_whole_series_std_dev:
                whole_series_std_dev_th04 = the_dictionary["signatures_th_0.4"][factor]["whole_series_std_dev"]
            signature_th04_avg = the_dictionary["signatures_th_0.4"][factor]["avg_series"]
            signature_th04_std = the_dictionary["signatures_th_0.4"][factor]["std_dev"]
        else:
            signature_th04 = np.zeros(len(signature))
            if plot_whole_series_std_dev:
                whole_series_std_dev_th04 = np.zeros(len(signature))
            signature_th04_avg = np.zeros(len(signature_avg))
            signature_th04_std = np.zeros(len(signature_avg))
        
        if factor in the_dictionary["signatures_th_0.6"]:
            signature_th06 = the_dictionary["signatures_th_0.6"][factor]["whole_series"]
            if plot_whole_series_std_dev:
                whole_series_std_dev_th06 = the_dictionary["signatures_th_0.6"][factor]["whole_series_std_dev"]
            signature_th06_avg = the_dictionary["signatures_th_0.6"][factor]["avg_series"]
            signature_th06_std = the_dictionary["signatures_th_0.6"][factor]["std_dev"]
        else:
            signature_th06 = np.zeros(len(signature))
            if plot_whole_series_std_dev:
                whole_series_std_dev_th06 = np.zeros(len(signature))
            signature_th06_avg = np.zeros(len(signature_avg))
            signature_th06_std = np.zeros(len(signature_avg))
        
        if factor in the_dictionary["signatures_th_0.8"]:
            signature_th08 = the_dictionary["signatures_th_0.8"][factor]["whole_series"]
            if plot_whole_series_std_dev:
                whole_series_std_dev_th08 = the_dictionary["signatures_th_0.8"][factor]["whole_series_std_dev"]
            signature_th08_avg = the_dictionary["signatures_th_0.8"][factor]["avg_series"]
            signature_th08_std = the_dictionary["signatures_th_0.8"][factor]["std_dev"]
        else:
            signature_th08 = np.zeros(len(signature))
            if plot_whole_series_std_dev:
                whole_series_std_dev_th08 = np.zeros(len(signature))
            signature_th08_avg = np.zeros(len(signature_avg))
            signature_th08_std = np.zeros(len(signature_avg))
        
        if "scores" in the_dictionary:
            scores = the_dictionary["scores"][factor]["whole_series"]
            scores_avg = the_dictionary["scores"][factor]["avg_series"]
            scores_std = the_dictionary["scores"][factor]["std_dev"]
        else:
            plot_scores = False
            scores = np.zeros(len(signature))
            scores_avg = np.zeros(len(signature_avg))
            scores_std = np.zeros(len(signature_avg))
        if not plot_whole_series_std_dev:
            plot_factor_info(factor, whole_series_x_axis, signature, signature_avg, signature_std, \
                signature_th04, signature_th04_avg, signature_th04_std,\
                signature_th06, signature_th06_avg, signature_th06_std,\
                signature_th08, signature_th08_avg, signature_th08_std,\
                scores, scores_avg, scores_std, \
                the_path, TIME_SLOTS_PER_DAY, plot_scores=plot_scores)
        else:
            plot_factor_info(factor, whole_series_x_axis, signature, signature_avg, signature_std, \
                signature_th04, signature_th04_avg, signature_th04_std,\
                signature_th06, signature_th06_avg, signature_th06_std,\
                signature_th08, signature_th08_avg, signature_th08_std,\
                scores, scores_avg, scores_std, \
                the_path, TIME_SLOTS_PER_DAY, plot_scores=plot_scores, \
                whole_series_std_dev=whole_series_std_dev, \
                whole_series_std_dev_th04=whole_series_std_dev_th04, \
                whole_series_std_dev_th06=whole_series_std_dev_th06, \
                whole_series_std_dev_th08=whole_series_std_dev_th08)
            
        norm_signature = the_dictionary["norm_signatures"][factor]["whole_series"]
        norm_signature_avg = the_dictionary["norm_signatures"][factor]["avg_series"]
        norm_signature_std = the_dictionary["norm_signatures"][factor]["std_dev"]
        whole_series_x_axis = the_dictionary["norm_signatures"][factor]["whole_series_x_axis"]
        whole_series_x_axis = [dt.strptime(x, '%Y-%m-%d.%A,%H:%M:%S') for x in whole_series_x_axis]
        
        if factor in the_dictionary["norm_signatures_th_0.4"]:
            norm_signature_th04 = the_dictionary["norm_signatures_th_0.4"][factor]["whole_series"]
            norm_signature_th04_avg = the_dictionary["norm_signatures_th_0.4"][factor]["avg_series"]
            norm_signature_th04_std = the_dictionary["norm_signatures_th_0.4"][factor]["std_dev"]
        else:
            norm_signature_th04 = np.zeros(len(signature))
            norm_signature_th04_avg = np.zeros(len(norm_signature_avg))
            norm_signature_th04_std = np.zeros(len(norm_signature_avg))
            

        if factor in the_dictionary["norm_signatures_th_0.6"]:
            norm_signature_th06 = the_dictionary["norm_signatures_th_0.6"][factor]["whole_series"]
            norm_signature_th06_avg = the_dictionary["norm_signatures_th_0.6"][factor]["avg_series"]
            norm_signature_th06_std = the_dictionary["norm_signatures_th_0.6"][factor]["std_dev"]
        else:
            norm_signature_th06 = np.zeros(len(signature))
            norm_signature_th06_avg = np.zeros(len(norm_signature_avg))
            norm_signature_th06_std = np.zeros(len(norm_signature_avg))
            

        if factor in the_dictionary["norm_signatures_th_0.8"]:
            norm_signature_th08 = the_dictionary["norm_signatures_th_0.8"][factor]["whole_series"]
            norm_signature_th08_avg = the_dictionary["norm_signatures_th_0.8"][factor]["avg_series"]
            norm_signature_th08_std = the_dictionary["norm_signatures_th_0.8"][factor]["std_dev"]
        else:
            norm_signature_th08 = np.zeros(len(signature))
            norm_signature_th08_avg = np.zeros(len(norm_signature_avg))
            norm_signature_th08_std = np.zeros(len(norm_signature_avg))
        
        plot_factor_info(factor, whole_series_x_axis, norm_signature, norm_signature_avg, norm_signature_std, \
            norm_signature_th04, norm_signature_th04_avg, norm_signature_th04_std,\
            norm_signature_th06, norm_signature_th06_avg, norm_signature_th06_std,\
            norm_signature_th08, norm_signature_th08_avg, norm_signature_th08_std,\
            scores, scores_avg, scores_std, \
            the_normalized_path, TIME_SLOTS_PER_DAY, plot_scores=plot_scores)