import pandas as pd
import numpy as np
from datetime import datetime
from dateutil import parser
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

def column_mask(dfs):
    '''
    Takes in list of dfs and returns data frames with only necessary columns
    '''
    cols_crashes = ['crash_record_id', 'crash_date', 'posted_speed_limit','traffic_control_device', 'weather_condition', 'lighting_condition', 'first_crash_type', 'trafficway_type', 'alignment', 'roadway_surface_cond', 'road_defect', 'crash_type', 'damage', 'prim_contributory_cause', 'sec_contributory_cause', 'street_name', 'num_units', 'most_severe_injury', 'injuries_total', 'crash_hour', 'crash_day_of_week', 'latitude', 'longitude']
    cols_people = ['crash_record_id', 'person_type', 'seat_no', 'city', 'zipcode', 'sex', 'age', 'drivers_license_class', 'safety_equipment', 'airbag_deployed', 'ejection', 'injury_classification', 'driver_action', 'driver_vision', 'physical_condition', 'pedpedal_action', 'pedpedal_visibility', 'pedpedal_location', 'bac_result', 'bac_result value', 'cell_phone_use']
    cols_vehicles = ['crash_record_id', 'unit_type', 'num_passengers', 'make', 'model', 'vehicle_year', 'vehicle_defect', 'vehicle_type', 'vehicle_use', 'maneuver', 'occupant_cnt']
    
    cols_list = [cols_crashes, cols_people, cols_vehicles]
    
    conv_list = []
    
    for df, col_mask in list(zip(dfs, cols_list)):
        df.columns = [col.lower() for col in df.columns]
        conv_list.append(df[col_mask])
        
    return conv_list[0], conv_list[1], conv_list[2]
    
# holidays = pd.tseries.holiday.USFederalHolidayCalendar().holidays(start='2012', end='2022').to_pydatetime()

# final_holidays = pd.merge(dates_holidays, crashes_holiday, on=['DATE_REAL'])

# data['month'] = data['crash_date'].apply(lambda x: int(x[:2]))
# data['day'] = data['crash_date'].apply(lambda x: int(x[3:5]))
# data['year'] = data['crash_date'].apply(lambda x: int(x[6:10]))
# data['time_of_crash'] = data['crash_date'].apply(
#     lambda x: int(x[11:13]+x[14:16]+x[17:19]) if 'AM' in x else int((str(int(x[11:13])+12))+x[14:16]+x[17:19])
# )
# data['time_of_crash'].iloc[111]


# def basic_info(data):
#     print("Dataset shape is: ", data.shape)
#     print("Dataset size is: ", data.size)
#     print("Dataset columns are: ", data.columns)
#     print("Dataset info is: ", data.info())
#     categorical = []
#     numerical = []
#     for i in data.columns:
#         if data[i].dtype == object:
#             categorical.append(i)
#         else:
#             numerical.append(i)
#     print("Categorical variables are:\n ", categorical)
#     print("Numerical variables are:\n ", numerical)
#     return categorical, numerical


