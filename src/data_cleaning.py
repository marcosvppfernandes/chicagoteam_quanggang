import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar


def column_mask(dfs):
    '''
    Takes in list of dfs and returns data frames with only necessary columns
    '''
    cols_crashes = ['crash_record_id', 'crash_date', 'posted_speed_limit',
                    'traffic_control_device', 'weather_condition',
                    'lighting_condition', 'first_crash_type',
                    'trafficway_type',
                    'alignment', 'roadway_surface_cond',
                    'road_defect', 'crash_type', 'damage',
                    'prim_contributory_cause', 'sec_contributory_cause',
                    'street_name', 'num_units', 'most_severe_injury',
                    'injuries_total', 'crash_hour', 'crash_day_of_week',
                    'latitude', 'longitude']
    cols_people = ['crash_record_id', 'person_type', 'sex', 'age',
                   'safety_equipment', 'airbag_deployed',
                   'ejection', 'injury_classification', 'driver_action',
                   'driver_vision', 'physical_condition',
                   'pedpedal_action', 'pedpedal_visibility',
                   'pedpedal_location', 'bac_result', 'bac_result value',
                   'cell_phone_use']
    cols_vehicles = ['crash_record_id', 'unit_type', 'num_passengers',
                     'vehicle_year', 'vehicle_defect', 'vehicle_type',
                     'vehicle_use', 'maneuver', 'occupant_cnt']
    cols_list = [cols_crashes, cols_people, cols_vehicles]
    conv_list = []
    for df, col_mask in list(zip(dfs, cols_list)):
        df.columns = [col.lower() for col in df.columns]
        conv_list.append(df[col_mask])
    return conv_list[0], conv_list[1], conv_list[2]
