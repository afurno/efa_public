'''
Created on Dec 2, 2014

@author: insa-furno
'''
from sys import argv
import pandas as pd
import argparse
import os

day_to_date = {"Monday":"1970-01-05", "Tuesday":"1970-01-06", "Wednesday":"1970-01-07", "Thursday":"1970-01-08", "Friday":"1970-01-09"\
               , "Saturday":"1970-01-10", "Sunday":"1970-01-11"}
CATEGORIES = ['NI', 'c1', 'c2', 'c3', 'c4', 'c5', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'm1', 'm2', 'm3',\
     'm4', 'm5', 'n1', 'n2', 'n3', 'n4', 'o1', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8',\
     'w1', 'w2', 'w3', 'w4', 'w5', 'w6', 'x3', 'x1', 'x2', 'other']
def pre_process_dataset(cells_filepath, traffic_volume_filepath, output_basepath, label, ranges, time_period_in_minutes):
    traffic_volume_output_filepath = output_basepath + "/med7d.dat"
    print "> Writing file %s" % traffic_volume_output_filepath
    if not os.path.exists(output_basepath):
        os.makedirs(output_basepath)

    cells_df = pd.read_csv(cells_filepath, dtype={'id':str, 'X': str, 'Y': str, 'same_position_base_stations':str})
    cells_df = cells_df.rename(columns={"id":"cell_id"})
    cells_df = cells_df.set_index('cell_id')
    # dateparse = lambda x, y, z: pd.datetime.strptime(x + " " + y + " " + z, '%A %Y-%m-%d %H:%M:%S')
    traffic_df = pd.read_csv(traffic_volume_filepath, sep=' ')  # , parse_dates=[[0, 1, 2]], date_parser=dateparse
    old_len = len(traffic_df)
    traffic_df["datetime"] = pd.to_datetime(traffic_df.date.map(str) + " " + traffic_df.time.map(str))
    traffic_df = traffic_df[(traffic_df.datetime >= (ranges[0] + " 00:00:00")) & (traffic_df.datetime <= (ranges[1] + " 23:59:59"))]
    new_len = len(traffic_df)
    print "After filtering %d lines were removed... There is a total of %d unique datetimes in the dataset" %(old_len - new_len, len(traffic_df.datetime.unique()))
    traffic_df = traffic_df[['#name_of_the_day', 'time', 'cell_id', 'category', 'traffic_load']]  
    
    traffic_df = traffic_df.groupby(['#name_of_the_day', 'time', 'category', 'cell_id']).median()
    traffic_df = traffic_df.reset_index()
    
    # this df will be just an index
    hour_range = pd.date_range("1970-01-05", periods=round(24.0 * 60.0 / int(time_period_in_minutes)), freq=str(time_period_in_minutes) + 'Min')
    index = [ [d, t._time_repr, cell, category] for cell in cells_df.index for t in hour_range for d in day_to_date.keys() for category in CATEGORIES]
    temp_df = pd.DataFrame(index)
    temp_df.columns = ['#name_of_the_day', 'time', 'cell_id', 'category']
    temp_df = temp_df.set_index(['#name_of_the_day', 'time', 'cell_id', 'category'])
    traffic_df = traffic_df.set_index(['#name_of_the_day', 'time', 'cell_id', 'category'])
    traffic_df = traffic_df.reindex_axis(temp_df.index)
    traffic_df = traffic_df.fillna(0)
    traffic_df = traffic_df.reset_index()
    
    if FILTER_LOW_TRAFFIC_CELLS:
        max_num_samples = len(hour_range) * 7
        temp_df = traffic_df.groupby(["cell_id", "category"])["traffic_load"].apply(lambda x: 
                                                      len([el for el in x if el > 0]))
        typical_week_cells = temp_df.loc[temp_df >= MIN_PERCENTAGE * max_num_samples]
        no_traffic_cells = temp_df.loc[temp_df == 0]
        less_than_percentage_cells = temp_df.loc[(temp_df < MIN_PERCENTAGE * max_num_samples) & (temp_df > 0)]
        print "There are %d (cell, category) pairs with typical week" % (len(typical_week_cells.index.tolist()))
        #cells_without_typical_week = traffic_df[~traffic_df[["cell_id", "category"]].isin(typical_week_cells.index)][["cell_id", "category"]].unique().tolist()
        #print "There are %d cells without typical week: %s" % (len(cells_without_typical_week), cells_without_typical_week)
        print "There are %d (cell, category) pairs with no traffic" % (len(no_traffic_cells.index.tolist()))
        print "%d (cell, category) pairs have instead less than %d%% samples for the median week" % (len(less_than_percentage_cells.index.tolist()), 100 * MIN_PERCENTAGE)
        traffic_df = traffic_df.merge(typical_week_cells.reset_index(),on=["cell_id", "category"], suffixes=["","_temp"])
    
    traffic_df["date"] = None
    for day_of_the_week in day_to_date:
        traffic_df.loc[traffic_df["#name_of_the_day"] == day_of_the_week, "date"] = day_to_date[day_of_the_week]
    traffic_df = pd.merge(traffic_df, cells_df, left_on='cell_id', right_index=True)     
    traffic_df["area"] = label      
    traffic_df = traffic_df[['#name_of_the_day', 'date', 'time', 'cell_id', 'X', 'Y', 'area', 'category', 'traffic_load']]
    traffic_df = traffic_df.sort_values(by=['date', 'time', 'cell_id', 'category'])
    traffic_df = traffic_df.rename(columns={'#name_of_the_day':'#day_of_the_week', 'date':'fake_date', 'time':'hour'\
                                            , "cell_id":"cell-id", "traffic_load":"total_volume_per_day_of_the_week", "X": "lon", "Y":"lat"})
    traffic_df.to_csv(traffic_volume_output_filepath, index=False, sep=" ")
    if FILTER_LOW_TRAFFIC_CELLS:
        cells_df = cells_df.reset_index()
        cells_with_typical_week_filepath = output_basepath + "/cells_with_typical_week.csv"
        cells_df[cells_df.cell_id.isin(typical_week_cells.index)].to_csv(cells_with_typical_week_filepath, index=False)
        cells_without_typical_week_filepath = output_basepath + "/cells_without_typical_week.csv"
        cells_df[cells_df.cell_id.isin(less_than_percentage_cells.index)].to_csv(cells_without_typical_week_filepath, index=False)
        cells_without_traffic_filepath = output_basepath + "/cells_without_traffic.csv"
        cells_df[cells_df.cell_id.isin(no_traffic_cells.index)].to_csv(cells_without_traffic_filepath, index=False)
    '''normalization.pickle_time_series(output_basepath, "/med7d")
    normalization.normalize_with_std_scores(output_basepath, "/med7d", time_period_in_minutes, label)
    normalization.normalize_with_daily_traffic(output_basepath, "/med7d", time_period_in_minutes, label)'''
       
MIN_PERCENTAGE = 0.1# 0.0
FILTER_LOW_TRAFFIC_CELLS = True
def main(dataset_basepath, cells_filepath, output_basepath, label, ranges, period):
    pre_process_dataset(cells_filepath, dataset_basepath, output_basepath, label, ranges, period)
    
if __name__ == '__main__':
    args_number = len(argv)
    if args_number < 5:
        print "USAGE: python preprocessor.py -i dataset_input_path -o output_basepath -c cells_center_lon_lat_filepath"
        exit(1)
    else:
        parser = argparse.ArgumentParser(prog='preprocessor', usage='%(prog)s -r start_date1 end_date1 -r start_date2 end_date2 ... -t aggregation_type -g geographic_aggregation_type')
        parser.add_argument ('-i', '--dataset_basepath')
        parser.add_argument ('-c', '--cells_center_lon_lat_filepath')
        parser.add_argument ('-o', '--output_basepath')
        parser.add_argument ('-l', '--label')
        parser.add_argument ('-r', '--ranges', nargs=2, action='append')
        parser.add_argument ('-p', '--aggregation_period', default=60, type=int)
        
        ranges = []
        
        for f in parser.parse_args().ranges:
            ranges.append(tuple(f))        
        
        print "Filtering to " + ranges[0][0] + " - " + ranges[0][1]
        
        main(parser.parse_args().dataset_basepath, \
             parser.parse_args().cells_center_lon_lat_filepath, \
             parser.parse_args().output_basepath, parser.parse_args().label, parser.parse_args().ranges[0], parser.parse_args().aggregation_period)
