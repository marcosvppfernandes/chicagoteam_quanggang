

X_model_features = ['IS_A_HOLIDAY', 'STREET_NO', 'DAMAGE', 'ROADWAY_SURFACE_COND', 'POSTED_SPEED_LIMIT', 
                    'WEATHER_CONDITION', 'LIGHTING_CONDITION']

X = crashes[X_model_features]

class Preprocessor():
    '''
    Takes in Pandas DataFrame (both containing predictor and target); 
    returns DataFrame with NO MISSING VALUES YAY.
    
    Params:
    df - Pandas DataFrame
    focus_list - list of column names (strings)
    target - name of target (str) you want to try - required
    
    '''
    import pandas as pd
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.pipeline import Pipeline
    
    def __init__(self, target_col_name, df=pd.DataFrame([]), thresh=0.4, 
                 train_sizes=[.5, .6, .75, .8], focus_list=None):
        self.df = df
        self.thresh = thresh
        self.target_col_name = target_col_name
        self.train_sizes = train_sizes
        self.focus_list = focus_list
    
    def remove_NaNs(self):
        data_list = []
        self.df = self.df.dropna(thresh=self.thresh*len(self.df), axis=1)
        
        if self.focus_list != None:
            for col in self.focus_list:
                focus_drop_data = self.df.dropna(subset=[col]).fillna(value=np.nan)
                data_list.append(focus_drop_data)
            return data_list
        else:
            data_list.append(self.df.fillna(value=np.nan))
            return data_list
    
    def target_split(self, data_list):
        
        new_data_list = []
        for dataset in data_list:
            X = dataset.drop(labels=self.target_col_name, axis=1)
            y = dataset[self.target_col_name]
            new_data_list.append({'X': X, 'y': y})
        return new_data_list
        
    
    def split_train_test(self, new_data_list):
        from sklearn.model_selection import train_test_split
        tts_data = []
        for xy in new_data_list:
            for size in self.train_sizes:
                X_train, X_test, y_train, y_test = train_test_split(xy['X'], xy['y'],
                                                               stratify=xy['y'], train_size=size)
                
                tts_data.append({'train': {'X': pd.DataFrame(X_train, columns=xy['X'].columns), 
                                           'y': y_train}, 
                                 'test': {'X': pd.DataFrame(X_test, columns=xy['X'].columns), 
                                          'y': y_test}, 
                                 'train_size': size})
            
        return tts_data
    
    
    def scale_and_ohe(self, tts_data, continuous_cols, categorical_cols):
        '''
        Params:
        continuous_cols - list of column names (strings) of all continuous variables
        categorical_cols - list of column names (strings, not df's) of all categorical variables
        '''
        
        prepped_data = []
        
        ss = StandardScaler()
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
        
        for dataset in tts_data:
            # source: https://dunyaoguz.github.io/my-blog/dataframemapper.html
            
            X_tr_cont = ss.fit_transform(dataset['train']['X'][continuous_cols])
            X_tr_cats = ohe.fit_transform(dataset['train']['X'][categorical_cols])
            
            X_train = pd.concat([pd.DataFrame(X_tr_cont, columns=continuous_cols), 
                                 pd.DataFrame(X_tr_cats, columns=ohe.get_feature_names())], axis=1)
            
            X_te_cont = ss.transform(dataset['test']['X'][continuous_cols])
            X_te_cats = ohe.transform(dataset['test']['X'][categorical_cols])
            X_test = pd.concat([pd.DataFrame(X_te_cont, columns=continuous_cols), 
                                 pd.DataFrame(X_te_cats, columns=ohe.get_feature_names())], axis=1)
            
            prepped_data.append(
                {'train': {'X': X_train, 'y': dataset['train']['y']}, 
                'test': {'X': X_test, 'y': dataset['test']['y']},
                'train_size': dataset['train_size']}
            )
        return prepped_data
    
    
    def balance_classes(self, prepped_data, minority_size=0.7, majority_reduce=0.7):
        # source: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
        oversample_even = SMOTE()
        oversample = SMOTE(sampling_strategy=minority_size)
        undersample = RandomUnderSampler(sampling_strategy=majority_reduce)
        smoted_data = []
        for dataset in prepped_data:
            X, y = oversample_even.fit_resample(dataset['train']['X'], dataset['train']['y'])
            
            
            X_part_over, y_part_over = oversample.fit_resample(dataset['train']['X'], dataset['train']['y'])

            X_under, y_under = undersample.fit_resample(X_part_over, y_part_over)
            
            smoted_data.append({'train': {'SMOTE_even_split': {'X': X, 'y': y},
                               'SMOTE_undersampled': {'X': X_under, 'y': y_under}, 'no_SMOTE': {'X': dataset['train']['X'], 'y': dataset['train']['y']}},
                               'test': {'X': dataset['test']['X'], 'y': dataset['test']['y']},
                               'train_size': dataset['train_size']})
        
        
        return smoted_data

######################## CHRISTOS ####################
    
import pandas as pd
df = pd.read_csv('traffic_crashes_chicago.csv')

df = df.drop(['CRASH_RECORD_ID', 'RD_NO', 'CRASH_DATE_EST_I', 'CRASH_DATE', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ALIGNMENT', 'REPORT_TYPE', 'CRASH_TYPE', 'INTERSECTION_RELATED_I', 'NOT_RIGHT_OF_WAY_I', 'HIT_AND_RUN_I', 'DAMAGE', 'DATE_POLICE_NOTIFIED', 'PRIM_CONTRIBUTORY_CAUSE', 'SEC_CONTRIBUTORY_CAUSE', 'STREET_NAME', 'BEAT_OF_OCCURRENCE', 'PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE', 'WORKERS_PRESENT_I', 'MOST_SEVERE_INJURY', 'STREET_NO', 'INJURIES_FATAL', 'INJURIES_INCAPACITATING', 'INJURIES_NON_INCAPACITATING', 'INJURIES_REPORTED_NOT_EVIDENT', 'INJURIES_NO_INDICATION', 'INJURIES_UNKNOWN', 'LATITUDE', 'LONGITUDE',  'LOCATION', 'LANE_CNT'], axis=1)
df['STREET_DIRECTION'].fillna(method='ffill', inplace=True)
df['target-injuries'] = df['INJURIES_TOTAL'] > 0
df.drop('INJURIES_TOTAL', axis=1, inplace=True)

from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
df['WEATHER_CONDITION'] = lbl.fit_transform(df['WEATHER_CONDITION'].astype(str))
df['LIGHTING_CONDITION'] = lbl.fit_transform(df['LIGHTING_CONDITION'].astype(str))
df['ROADWAY_SURFACE_COND'] = lbl.fit_transform(df['ROADWAY_SURFACE_COND'].astype(str))
df['ROAD_DEFECT'] = lbl.fit_transform(df['ROAD_DEFECT'].astype(str))
df['STREET_DIRECTION'] = lbl.fit_transform(df['STREET_DIRECTION'].astype(str))

X = df.drop('target-injuries', axis=1)
y = df['target-injuries']


import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import xgboost as xgb


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
D_train = xgb.DMatrix(X_train, label=Y_train)
D_test  = xgb.DMatrix(X_test, label=Y_test)

params = {
        'min_child_weight': [5,6,7,8],
        'gamma'           : [1.1,1.2,1.3],
        'subsample'       : [.7,.8,.9],
        'max_depth'       : [10,11,12,13],
        'eta'             : [.2,.3,.4],
        'colsample_bytree': [.4,.5,.6]        
        }


xgb = XGBClassifier(learning_rate=0.02,
                    n_estimators=600,
                    objective='binary:logistic',
                    silent=True,
                    nthread=1,
                    tree_method= 'gpu_hist'
#                     verbosity=0,
#                    scale_pos_weight = 7
                   )

print('\n All results:')
print(random_search.cv_results_)
print('\n Best estimator:')
print(random_search.best_estimator_)
print('\n Best normalized gini score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
print(random_search.best_score_)
#       * 2 - 1)
print('\n Best hyperparameters:')
print(random_search.best_params_)
results = pd.DataFrame(random_search.cv_results_)
results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

random_search = RandomizedSearchCV(xgb,
                                   param_distributions=params,
                                   n_iter=param_comb,
                                   scoring='roc_auc',
                                   n_jobs=4,
                                   cv=skf.split(X_train,Y_train),
                                   verbose=3,
                                   random_state=1001 )

start_time = timer(None)
random_search.fit(X_train, Y_train)
timer(start_time)
    
################## MARCOS ##########################  
kf = KFold()
precision_scores = []
for trained_indices, val_indices in kf.split(X, y):
    X_t = X.iloc[trained_indices]
    X_val = X.iloc[val_indices]
    y_t = y.iloc[trained_indices]
    y_val = y.iloc[val_indices]
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    X_t_cat = ohe.fit_transform(X_t.select_dtypes(include='object'))
    X_t_num = ohe.fit_transform(X_t.select_dtypes(exclude='object'))