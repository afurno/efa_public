'''
Created on Dec 2, 2014

@author: insa-furno
'''
from sys import argv
import pandas as pd
import numpy as np
import argparse
import os
import datetime, pytz
from pytz import timezone

#execute if from shell otherwise bug!

def pre_process_cell_file(cells_filepath, aggregate_cells_with_same_coordinates, output_basepath):
    cells_df = pd.read_csv(cells_filepath, header=0, names= ["id", "X", "Y"], dtype={'id':str, 'X': str, 'Y': str}, sep=";")
    
    cells_df = cells_df.set_index('id')
    
    print "There are %d cells in the urban area dataset" % len(cells_df)
    
    same_position_base_stations = {}
    
    if aggregate_cells_with_same_coordinates:
        cells_df["same_position_base_stations"] = None
        all_cells = cells_df.index
        for cell_id in all_cells:
            # check for cell_id presence in the updated index
            if cell_id not in cells_df.index:
                continue
            row_same_id_df = cells_df[(cells_df['X'] == cells_df.loc[cell_id]['X']) & (cells_df['Y'] == cells_df.loc[cell_id]['Y'])]
            if len(row_same_id_df) > 1:
                same_position_bs_ids = [label for label in row_same_id_df.index]
                new_id = ("multiple_bs_%d_" + cell_id) % (len(same_position_bs_ids))
                print "New cell-id %s cover the following cell ids %s" % (new_id, str(same_position_bs_ids))
                cells_df.loc[new_id] = cells_df.loc[cell_id]
                cells_df.loc[new_id, "same_position_base_stations"] = ' '.join(str(label) for label in  same_position_bs_ids)
                cells_df = cells_df.loc[[cell_id for cell_id in cells_df.index if (cell_id not in same_position_bs_ids)]]
                same_position_base_stations[new_id] = same_position_bs_ids
        print "After filtering cells with same location: %d cells" % len(cells_df)
        print same_position_base_stations

    output_filepath = output_basepath + "/selected_cells.csv"
    if not os.path.exists(output_basepath):
        os.makedirs(output_basepath)
    output_file = open(output_filepath, 'w')
    cells_df.to_csv(output_file, index_label="cell_id")
    
    return cells_df, same_position_base_stations

def pre_process_dataset(cells_df, same_position_base_stations, dataset_basepath, label, \
                        time_period_in_minutes, output_basepath, ranges, aggregation_type, put_zeroes):
    traffic_volume_output_filepath = output_basepath + "/traffic_load.dat"
    print "> Writing file %s" % traffic_volume_output_filepath
    if not os.path.exists(output_basepath):
        os.makedirs(output_basepath)

    files_list = os.listdir(dataset_basepath)
    files_list.sort()
    
    unique_base_stations = [cell_id for cell_id in cells_df.index if not cell_id in same_position_base_stations]
    all_cells = [cell_id for same_position_cells in same_position_base_stations.values() for cell_id in same_position_cells] + unique_base_stations
    '''parser = lambda x: timezone('UTC')\
        .localize(datetime.datetime.utcfromtimestamp(int(x)))\
        .astimezone(timezone('CET')).strftime("%Y-%m-%d %H:%M:%S")'''
    first = True
    for filename in files_list:
        if not filename.startswith("TCPSessCourtParisLyon_"):
            continue
        
        date_format = filename.split("_") #format is TCPSessCourtParisLyon_300_2016_09_nidt
        traffic_volume_filepath = dataset_basepath + '/' + filename
        names = ['datetime', 'cell_id', 'category', 'uplink', 'downlink']
        traffic_df = pd.read_csv(traffic_volume_filepath, sep=';', header=None, names=names)#, parse_dates=[0], date_parser=parser)
        #print "Number of lines in %s: %d" %(filename, len(traffic_df))
        #traffic_df = traffic_df[traffic_df.category.isin(traffic_kind)] #filtering by category
        #print "Number of lines in %s after filtering to categories %s: %d" %(filename, str(traffic_kind), len(traffic_df))
        # filter to relevant cells
        traffic_df = traffic_df[traffic_df['cell_id'].isin(all_cells)]
        print "Number of lines in %s after filtering to selected cells: %d" %(filename, len(traffic_df))
        traffic_df["datetime"] = pd.to_datetime(traffic_df.datetime, unit='s').apply(lambda x: x.tz_localize('UTC'))
        traffic_df = traffic_df.groupby(['datetime', 'cell_id', 'category']).sum()
        traffic_df = traffic_df.reset_index()
        #bad date --->>>>>traffic_df["datetime"] = pd.to_datetime(traffic_df["datetime"])
        time_period_in_minutes = str(time_period_in_minutes)
        
        print "Considering sum of uplink and downlink for selected service categories..."
        if aggregation_type == 'uplink':
            traffic_df['traffic_load'] = traffic_df['uplink']
        elif aggregation_type == 'downlink':
            traffic_df['traffic_load'] = traffic_df['downlink']
        else:
            traffic_df['traffic_load'] = traffic_df['uplink'] + traffic_df['downlink']
        #traffic_df = traffic_df[(traffic_df.datetime >= ranges[0] + " 00:00:00") & (traffic_df.datetime <= ranges[1] + " 23:59:59")]
        if put_zeroes:
            traffic_df = traffic_df[['datetime', 'cell_id', 'category', 'traffic_load']]
            traffic_df = traffic_df.set_index("datetime").groupby(["cell_id", "category"]).resample(time_period_in_minutes + 'Min').sum()
            traffic_df = traffic_df.reset_index()
            # get communication data (sms+call french and non-french)
            year = int(date_format[2])
            month = int(date_format[3])
            day = int(date_format[4])
            next_year = int(date_format[5])
            next_month = int(date_format[6])
            next_day = int(date_format[7])
            timezone = pytz.timezone("Europe/Paris")
            date_from = datetime.datetime(year, month, day, 0, 0, 0)
            date_from = timezone.localize(date_from)
            date_to = datetime.datetime(next_year, next_month, next_day, 0, 0, 0)
            date_to = timezone.localize(date_to)
            timezone = pytz.timezone("UTC")
            date_from = date_from.astimezone(timezone)
            date_to = date_to.astimezone(timezone)
            num_days = (date_to - date_from).days + 1
            num_periods = num_days*round(24.0 * 60.0 / int(time_period_in_minutes))
            print "Checking daylight saving change"
            day_light_savings_change_positive = (next_month == 10 and next_day == 31 and month == 10 and day == 1)
            day_light_savings_change_negative = (next_month == 3 and next_day == 31 and month == 3 and day == 1)
            if day_light_savings_change_positive:
                old_num_periods = num_periods
                num_periods += 1 * 60.0 / int(time_period_in_minutes)
                print "Positive daylight saving change detected. Adding %d slots to num_periods: %d" %(old_num_periods-num_periods, num_periods)
            elif day_light_savings_change_negative:
                old_num_periods = num_periods
                num_periods -= 1 * 60.0 / int(time_period_in_minutes)
                print "Negative daylight saving change detected. Removing %d slots to num_periods: %d" %(old_num_periods-num_periods, num_periods)
            else:
                print "No daylight saving change detected"
            hour_range = pd.date_range(date_from, periods=num_periods, freq=str(time_period_in_minutes) + 'Min', tz="UTC")
            print "There are %d categories in the dataset..." %(len(CATEGORIES))
            print str(CATEGORIES)
            index = [ [d, cell, c] for cell in all_cells for d in hour_range for c in CATEGORIES]
            # this df will be just an index
            temp_df = pd.DataFrame(index)
            temp_df.columns = ['datetime', 'cell_id', 'category']
            temp_df = temp_df.set_index(['datetime', 'cell_id', 'category'])
            traffic_df = traffic_df.set_index(["datetime", "cell_id", 'category'])
            traffic_df = traffic_df.reindex_axis(temp_df.index).fillna(0)
            traffic_df = traffic_df.reset_index()
        else:
            traffic_df = traffic_df.set_index("datetime").groupby(["cell_id", 'category']).resample(time_period_in_minutes + 'Min', how='sum', base=0)
            traffic_df = traffic_df.reset_index()
            traffic_df = traffic_df[['datetime', 'cell_id', 'category', 'traffic_load']]
            traffic_df = traffic_df[np.isfinite(traffic_df['traffic_load'])]
        
        for new_id in same_position_base_stations:
            subset_df = traffic_df[traffic_df["cell_id"].isin(same_position_base_stations[new_id])]
            subset_df = subset_df.groupby(["datetime", 'category']).sum()
            subset_df = subset_df.reset_index()
            traffic_df = traffic_df[~traffic_df["cell_id"].isin(same_position_base_stations[new_id])]
            subset_df["cell_id"] = new_id
            traffic_df = traffic_df.append(subset_df, ignore_index=True)
                
        
        traffic_df["datetime"] = traffic_df["datetime"].apply(lambda x: x.tz_convert('CET'))
        traffic_df['#name_of_the_day'] = traffic_df['datetime'].apply(lambda x: x.strftime('%A'))
        traffic_df['date'] = traffic_df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d'))
        traffic_df['time'] = traffic_df['datetime'].apply(lambda x: x.strftime('%H:%M:%S'))
        traffic_df['aggregation'] = label
    
        traffic_df = pd.merge(traffic_df, cells_df, left_on='cell_id', right_index=True)
        
        traffic_df = traffic_df.rename(columns={'X':'lon', 'Y':'lat'})
        traffic_df = traffic_df[["#name_of_the_day", "date", "time", "cell_id", "lon", "lat", "aggregation", 'category', "traffic_load"]]
        traffic_df = traffic_df.sort_values(by = ["date", "time"])
        #print traffic_df.head()
        grouped_traffic_df = traffic_df.groupby(["#name_of_the_day", "date", "time", "cell_id", "lon", "lat", "aggregation", 'category'])
        #size = grouped_traffic_df.size()
        #print "Should be not empty only with positive daylight traffic change..."
        #print size[size > 1].head()
        #print size[size > 1].tail()
        traffic_df = grouped_traffic_df.sum()
        traffic_df = traffic_df.reset_index()
        traffic_df = traffic_df[["#name_of_the_day", "date", "time", "cell_id", "lon", "lat", "aggregation", 'category', "traffic_load"]]
        traffic_df = traffic_df.sort_values(by=['date', 'time', 'cell_id'])
        # final_df[['traffic_load']] = final_df[['traffic_load']].astype(int)
        
        if first:
            first = False
            traffic_df.to_csv(traffic_volume_output_filepath, sep=" ", index=False)
        else:
            traffic_df.to_csv(traffic_volume_output_filepath, sep=" ", mode="a", header=False, index=False)
    
    #if put_zeroes:
    #    normalize_with_std_scores(output_basepath, "traffic_load", int(time_period_in_minutes), label, CATEGORIES)

def main(dataset_basepath, cells_filepath, time_period, label, output_basepath, ranges, aggregation_type, aggregate_cells_with_same_coordinates, put_zeroes):
    cells_df, same_position_base_stations = pre_process_cell_file(cells_filepath, aggregate_cells_with_same_coordinates, output_basepath)
    pre_process_dataset(cells_df, same_position_base_stations, dataset_basepath, label, time_period, output_basepath, ranges, aggregation_type, put_zeroes)

# -i /Volumes/Data/insa/datasets/tim_big_data_challenge_2015/milan 
# -o /Volumes/Data/ifsttar/activities/spatio_temporal_profiling_via_clustering/milan_internet_2015
# -r 2015-03-01 2015-04-30 
# -t all 
# -c /Volumes/Data/insa/datasets/tim_big_data_challenge_2015/milan/milano-grid/centroids_milan.csv
# -l milan_2015-03-01_2015-04-30 
# -act internet
# -p 60
# -z
CATEGORIES = ['NI', 'c1', 'c2', 'c3', 'c4', 'c5', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'm1', 'm2', 'm3',\
     'm4', 'm5', 'n1', 'n2', 'n3', 'n4', 'o1', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',\
     'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'x3', 'x1', 'x2'] #'other'
if __name__ == '__main__':
    args_number = len(argv)
    if args_number < 8:
        print "USAGE: python preprocessor.py -i dataset_input_path -o output_path -r date1 date2 -r date3 date4 -t aggregation_type -g geographic_aggregation_type"
        exit(1)
    else:
        parser = argparse.ArgumentParser(prog='preprocessor', usage='%(prog)s -r start_date1 end_date1 -r start_date2 end_date2 ... -t aggregation_type -g geographic_aggregation_type')
        parser.add_argument ('-i', '--dataset_basepath')
        parser.add_argument ('-o', '--output_basepath')
        parser.add_argument ('-r', '--ranges', nargs=2, action='append')
        parser.add_argument ('-t', '--type', default="sum") #downlink, uplink, sum
        parser.add_argument ('-p', '--aggregation_period', default=60, type=int)
        parser.add_argument ('-c', '--cells_center_lon_lat_filepath')
        parser.add_argument ('-l', '--label')
        parser.add_argument ('-s', '--same_coordinates_check', action='store_true', default=False)
        parser.add_argument ('-z', '--zeroes', action='store_true', default=False)
        
        ranges = []
        
        for f in parser.parse_args().ranges:
            ranges.append(tuple(f))        
    
        str_ranges = ', '.join(str(a) + "_" + str(b) for a, b in  ranges)
        str_ranges = str_ranges.replace(", ", "_")
        
        output_basepath = parser.parse_args().output_basepath + "_" + str_ranges
        main(parser.parse_args().dataset_basepath, \
             parser.parse_args().cells_center_lon_lat_filepath, \
             parser.parse_args().aggregation_period, parser.parse_args().label, \
             output_basepath, ranges, parser.parse_args().type, parser.parse_args().same_coordinates_check, parser.parse_args().zeroes)
