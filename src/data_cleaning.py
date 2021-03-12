import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from dateutil import parser
from sklearn.model_selection import train_test_split
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from sklearn.linear_model import LogisticRegression
%matplotlib inline
sns.set()


holidays = pd.tseries.holiday.USFederalHolidayCalendar().holidays(start='2012', end='2022').to_pydatetime()

final_holidays = pd.merge(dates_holidays, crashes_holiday, on=['DATE_REAL'])

data['month'] = data['crash_date'].apply(lambda x: int(x[:2]))
data['day'] = data['crash_date'].apply(lambda x: int(x[3:5]))
data['year'] = data['crash_date'].apply(lambda x: int(x[6:10]))
data['time_of_crash'] = data['crash_date'].apply(
    lambda x: int(x[11:13]+x[14:16]+x[17:19]) if 'AM' in x else int((str(int(x[11:13])+12))+x[14:16]+x[17:19])
)
data['time_of_crash'].iloc[111]


def basic_info(data):
    print("Dataset shape is: ", data.shape)
    print("Dataset size is: ", data.size)
    print("Dataset columns are: ", data.columns)
    print("Dataset info is: ", data.info())
    categorical = []
    numerical = []
    for i in data.columns:
        if data[i].dtype == object:
            categorical.append(i)
        else:
            numerical.append(i)
    print("Categorical variables are:\n ", categorical)
    print("Numerical variables are:\n ", numerical)
    return categorical, numerical


