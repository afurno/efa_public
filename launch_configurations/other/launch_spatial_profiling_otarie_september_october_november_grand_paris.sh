cd ~/projects/spatial_profiling_via_clustering/src/ 
export PYTHONPATH="/home/furno/projects/spatial_profiling_via_clustering/:/home/furno/projects/spatial_profiling_via_clustering/src/"

#python preprocessing/traffic_load_preprocessor_otarie_all_services.py -i /home/furno/donnee_otarie/paris-lyon_services-otarie-2016/new_datasets/communication -o /home/furno/efa_otarie/grand_paris_all_services_1h -r 2016-09-05 2016-11-28 -c /home/furno/donnee_otarie/paris-lyon_services-otarie-2016/new_datasets/communication/nidt_lonlat_sept_oct_nov_union_grand_paris.csv -l grand_paris_2016-09-05_2016-11-28_all_services_1h -p 60 -z -s

#python preprocessing/median_week_preprocessor_otarie_all_services.py -i /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-11-28/traffic_load.dat -o /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-10-30/ -c /home/furno/efa_otarie/grand_paris_all_services_1h_2016-09-05_2016-11-28/selected_cells.csv -l grand_paris_median_2016-09-05_2016-10-30_all_services_1h -r 2016-09-05 2016-10-30 -p 60 > med_extraction_grand_paris_all_services_1h_2016-09-05_2016-10-30.log

#python preprocessing/traffic_load_preprocessor_otarie_all_services.py -i /home/furno/donnee_otarie/paris-lyon_services-otarie-2016/new_datasets/communication -o /home/furno/efa_otarie/grand_paris_all_services_30min -r 2016-09-05 2016-11-28 -c /home/furno/donnee_otarie/paris-lyon_services-otarie-2016/new_datasets/communication/nidt_lonlat_sept_oct_nov_union_grand_paris.csv -l grand_paris_2016-09-05_2016-11-28_all_services_30min -p 30 -z -s

#python preprocessing/median_week_preprocessor_otarie_all_services.py -i /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-11-28/traffic_load.dat -o /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-10-30/ -c /home/furno/efa_otarie/grand_paris_all_services_30min_2016-09-05_2016-11-28/selected_cells.csv -l grand_paris_median_2016-09-05_2016-10-30_all_services_30min -r 2016-09-05 2016-10-30 -p 30 > med_extraction_grand_paris_all_services_30min_2016-09-05_2016-10-30.log

#python preprocessing/traffic_load_preprocessor_otarie_all_services.py -i /home/furno/donnee_otarie/paris-lyon_services-otarie-2016/new_datasets/communication -o /home/furno/efa_otarie/grand_paris_all_services_10min -r 2016-09-05 2016-11-28 -c /home/furno/donnee_otarie/paris-lyon_services-otarie-2016/new_datasets/communication/nidt_lonlat_sept_oct_nov_union_grand_paris.csv -l grand_paris_2016-09-05_2016-11-28_all_services_10min -p 10 -z -s

#python preprocessing/median_week_preprocessor_otarie_all_services.py -i /home/furno/efa_otarie/grand_paris_all_services_10min_2016-09-05_2016-11-28/traffic_load.dat -o /home/furno/efa_otarie/grand_paris_all_services_10min_2016-09-05_2016-10-30/ -c /home/furno/efa_otarie/grand_paris_all_services_10min_2016-09-05_2016-11-28/selected_cells.csv -l grand_paris_median_2016-09-05_2016-10-30_all_services_10min -r 2016-09-05 2016-10-30 -p 10 > med_extraction_grand_paris_all_services_10min_2016-09-05_2016-10-30.log