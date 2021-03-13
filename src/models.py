from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np

class Preprocessor():
    '''
    Takes in Pandas DataFrame (both containing predictor and target); 
    returns DataFrame that's ready to .
    
    Params:
    df - Pandas DataFrame
    focus_list - list of column names (strings)
    target - name of target (str) you want to try - required
    
    '''
    import pandas as pd
    import numpy as np
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    
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
            
            if len(continuous_cols) > 0:
                X_tr_cont = ss.fit_transform(dataset['train']['X'][continuous_cols])
                X_tr_cats = ohe.fit_transform(dataset['train']['X'][categorical_cols])

                X_train = pd.concat([pd.DataFrame(X_tr_cont, columns=continuous_cols), 
                                     pd.DataFrame(X_tr_cats, columns=ohe.get_feature_names())], axis=1)

                X_te_cont = ss.transform(dataset['test']['X'][continuous_cols])
                X_te_cats = ohe.transform(dataset['test']['X'][categorical_cols])
                X_test = pd.concat([pd.DataFrame(X_te_cont, columns=continuous_cols), 
                                     pd.DataFrame(X_te_cats, columns=ohe.get_feature_names())], axis=1)
            else:
                X_train = ohe.fit_transform(dataset['train']['X'][categorical_cols])
                X_test = ohe.transform(dataset['test']['X'][categorical_cols])
            
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

class MasterModeler():
    '''
    Takes in a preprocessed dataset dictionary, places it in a pipeline, and gridsearch through 
    various hyperparameters. Returns diagnostics within the dataset dictionary.
    
    Params:
    classifiers - dictionary of classifier instances (ex: {'logreg' : [LogisticRegression(), grid]})
    '''
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelBinarizer
    from sklearn.model_selection import GridSearchCV, cross_val_score 
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, \
                                mean_squared_error, mean_absolute_error, roc_auc_score, \
                                classification_report
    
    def __init__(self, classifier_dict={'classifier': LogisticRegression(), 'grid': {}}):
        # Inherit stuff from Preprocessor()
        self.classifier_dict = classifier_dict
    
    def model_pipe(self, data_list):
        '''
        Params:
        continuous_cols - pandas DataFrame containing all continuous columns
        categorical cols - pandas DataFrame containing all categorical columns
        '''
        adding_fit_objs = []
        for dataset in data_list:                
            clf = self.classifier_dict['classifier']
            grid = self.classifier_dict['grid']

            gs = GridSearchCV(clf, grid, cv=3)
            fit_obj_even = gs.fit(dataset['train']['SMOTE_even_split']['X'].dropna(), dataset['train']['SMOTE_even_split']['y'].dropna())
            fit_obj_under = gs.fit(dataset['train']['SMOTE_undersampled']['X'].dropna(), dataset['train']['SMOTE_undersampled']['y'].dropna())
            fit_obj_ns = gs.fit(dataset['train']['no_SMOTE']['X'].dropna(), dataset['train']['no_SMOTE']['y'].dropna())
            
            adding_fit_objs.append({'train': {'SMOTE_even_split': {'X': dataset['train']['SMOTE_even_split']['X'], 'y': dataset['train']['SMOTE_even_split']['y'], 'fit_obj': fit_obj_even},
                               'SMOTE_undersampled': {'X': dataset['train']['SMOTE_undersampled']['X'], 'y': dataset['train']['SMOTE_undersampled']['X'], 'fit_obj': fit_obj_under},
                                'no_SMOTE' : {'X': dataset['train']['no_SMOTE']['X'], 'y': dataset['train']['no_SMOTE']['y'], 'fit_obj': fit_obj_ns}},
                               'test': {'X': dataset['test']['X'], 'y': dataset['test']['y']},
                               'train_size': dataset['train_size']})
            
        return adding_fit_objs

    def validate_models(self, fitted_data_list, folds=3):
        from sklearn.model_selection import cross_val_score
        
        with_scores = []
        for dataset in fitted_data_list:
            fit_obj_even = dataset['train']['SMOTE_even_split']['fit_obj']
            fit_obj_under = dataset['train']['SMOTE_undersampled']['fit_obj']
            fit_obj_ns = dataset['train']['no_SMOTE']['fit_obj']
            
            acc_even = cross_val_score(fit_obj_even, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='accuracy')
            
            prec_even = cross_val_score(fit_obj_even, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='precision')
            
            rec_even = cross_val_score(fit_obj_even, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='recall')
            
            acc_under = cross_val_score(fit_obj_under, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='accuracy')
            
            prec_under = cross_val_score(fit_obj_under, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='precision')
            
            rec_under = cross_val_score(fit_obj_under, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='recall')
            
            acc_ns = cross_val_score(fit_obj_ns, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='accuracy')
            
            prec_ns = cross_val_score(fit_obj_ns, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='accuracy')
            
            rec_ns = cross_val_score(fit_obj_ns, dataset['test']['X'], 
                                       dataset['test']['y'], cv=folds, scoring='accuracy')
            
            
            with_scores.append(
                            {'train': {'SMOTE_even_split': {'scores': {'acc': acc_even,'prec': prec_even,'rec': rec_even}},
                               'SMOTE_undersampled': {'scores': {'acc': acc_under,'prec': prec_under,'rec': rec_under}},
                                      'no_SMOTE': {'scores': {'acc': acc_ns,'prec': prec_ns,'rec': rec_ns}}},
                             'train_size': dataset['train_size']}
            )
        return with_scores
            
# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

def kfold_validation(X_train, y_train, classifier, 
                     continuous_cols, categorical_cols, smote=False,
                    minority_size=0.7, majority_reduce=0.7):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import recall_score, precision_score, accuracy_score, roc_auc_score

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1001)

    val_recall = []
    val_prec = []
    val_acc = []
    roc_auc = []

    for train_ind, val_ind in skf.split(X_train, y_train):
        x_t = X_train.iloc[train_ind]
        y_t = y_train.iloc[train_ind]

        ss = StandardScaler()
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        cont = continuous_cols
        cat_cols = categorical_cols

        scaled = ss.fit_transform(x_t[cont])
        dummies = ohe.fit_transform(x_t[cat_cols])

        x_t = pd.concat([pd.DataFrame(scaled, columns=cont), 
                                  pd.DataFrame(dummies, columns=ohe.get_feature_names())], axis=1)


        x_val = X_train.iloc[val_ind]
        y_val = y_train.iloc[val_ind]
        
        
        sc = ss.transform(x_val[cont])
        dums = ohe.transform(x_val[cat_cols])

        x_val = pd.concat([pd.DataFrame(sc, columns=cont), 
                                  pd.DataFrame(dums, columns=ohe.get_feature_names())], axis=1)
        
        if smote==True:
            oversample = SMOTE(sampling_strategy=minority_size)
            undersample = RandomUnderSampler(sampling_strategy=majority_reduce)
            
            X_part_over, y_part_over = oversample.fit_resample(x_t, y_t)

            X_under, y_under = undersample.fit_resample(X_part_over, y_part_over)
            
            clf = classifier

            clf.fit(X_under, y_under)

            val_recall.append(recall_score(y_val, clf.predict(x_val)))
            val_prec.append(precision_score(y_val, clf.predict(x_val)))
            val_acc.append(accuracy_score(y_val, clf.predict(x_val)))
            roc_auc.append(roc_auc_score(y_val, clf.predict(x_val)))
            
        else:
            clf = classifier

            clf.fit(x_t, y_t)

            val_recall.append(recall_score(y_val, clf.predict(x_val)))
            val_prec.append(precision_score(y_val, clf.predict(x_val)))
            val_acc.append(accuracy_score(y_val, clf.predict(x_val)))
            roc_auc.append(roc_auc_score(y_val, clf.predict(x_val)))
            
    return val_recall, val_prec, val_acc, roc_auc, clf

def model_mask_binary(crashes):
    
    crashes['injured'] = crashes['injuries_total'] > 0
    
    conds = [(crashes['traffic_control_device']== 'NO CONTROLS'), (crashes['traffic_control_device']!= 'NO CONTROLS')]
    choices = ['No_device', 'device_present']

    crashes.traffic_control_device = np.select(conds, choices)
        
    conds = [((crashes['roadway_surface_cond']== 'DRY')|(crashes['roadway_surface_cond']== 'SAND, MUD, DIRT')), 
         ((crashes['roadway_surface_cond']!= 'DRY')&(crashes['roadway_surface_cond']!= 'SAND, MUD, DIRT'))]
    choices = ['Dry', 'Not_Dry']

    crashes.roadway_surface_cond = np.select(conds, choices)
    
    conds = [(crashes['weather_condition']== 'CLEAR'), 
         (crashes['weather_condition']!= 'CLEAR')]
    choices = ['Clear', 'Not_Clear']

    crashes.weather_condition = np.select(conds, choices)
    
    conds = [(crashes['num_units']<= 2), 
         (crashes['num_units']>= 3)]
    choices = ['2orless', '3+']

    crashes.num_units = np.select(conds, choices)
    
    mod_mask = ['traffic_control_device', 'weather_condition', 'first_crash_type',
           'roadway_surface_cond', 'road_defect', 'damage', 'prim_contributory_cause', 'num_units', 
            'crash_hour', 'injured']

    crash_mod = crashes[mod_mask]
    crash_mod = crash_mod[(crash_mod['road_defect']!='UNKNOWN')&(crash_mod['prim_contributory_cause']!='UNABLE TO DETERMINE')&(crash_mod['prim_contributory_cause']!='NOT APPLICABLE')]
    
    return crash_mod

def model_mask_ternary(crashes, people, vehicles):
    import pandas as pd
    import numpy as np
    from datetime import datetime
    from dateutil import parser
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    
    '''must read in dataframes again'''
    cr_cols_drop = ['CRASH_DATE_EST_I', 'LANE_CNT', 'INTERSECTION_RELATED_I', 'NOT_RIGHT_OF_WAY_I','HIT_AND_RUN_I','PHOTOS_TAKEN_I', 'STATEMENTS_TAKEN_I', 'DOORING_I', 'WORK_ZONE_I', 'WORK_ZONE_TYPE','WORKERS_PRESENT_I', 'LOCATION', 'RD_NO']
    
    crashes = crashes.drop(labels=cr_cols_drop, axis=1)
    
    crashes.dropna(inplace=True)
    
    crashes['INJURIES_FATAL'] = np.where(crashes['INJURIES_FATAL']>0, 1, 0)
    
    crashes.WEATHER_CONDITION = np.where(crashes.WEATHER_CONDITION=='BLOWING SNOW', 'SNOW', crashes.WEATHER_CONDITION)
    
    crashes.WEATHER_CONDITION = np.where(crashes.WEATHER_CONDITION=='FREEZING RAIN/DRIZZLE', 'RAIN', crashes.WEATHER_CONDITION)
    
    crashes.WEATHER_CONDITION = np.where(crashes.WEATHER_CONDITION=='FOG/SMOKE/HAZE', 'OTHER', crashes.WEATHER_CONDITION)
    
    crashes.WEATHER_CONDITION = np.where(crashes.WEATHER_CONDITION=='SLEET/HAIL', 'OTHER', crashes.WEATHER_CONDITION)
    
    crashes.WEATHER_CONDITION = np.where(crashes.WEATHER_CONDITION=='BLOWING SAND, SOIL, DIRT', 'OTHER', crashes.WEATHER_CONDITION)
    
    crashes.WEATHER_CONDITION = np.where(crashes.WEATHER_CONDITION=='SEVERE CROSS WIND GATE', 'OTHER', crashes.WEATHER_CONDITION)
    
    # Let's bin the speed limit in 9 groups, the last one being 45 miles/hour or above
    crashes.POSTED_SPEED_LIMIT = pd.cut(crashes.POSTED_SPEED_LIMIT,[0, 5, 10, 15, 20, 25, 30, 35, 40, 45], precision=0, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    crashes['MOST_SEVERE_INJURY'] = np.where(crashes['MOST_SEVERE_INJURY']=='REPORTED, NOT EVIDENT',
                                         'NONINCAPACITATING INJURY', crashes['MOST_SEVERE_INJURY'])
    
    crashes['MOST_SEVERE_INJURY'] = np.where(crashes['MOST_SEVERE_INJURY']=='FATAL',
                                         'INCAPACITATING INJURY', crashes['MOST_SEVERE_INJURY'])
    
    crashes['MOST_SEVERE_INJURY'] = np.where(crashes['MOST_SEVERE_INJURY']=='INCAPACITATING INJURY','INCAPACITATING INJURY/FATAL', crashes['MOST_SEVERE_INJURY'])
    
    crashes['DATE_ACCIDENT']= pd.to_datetime(crashes['CRASH_DATE'], format='%m/%d/%Y %I:%M:%S %p')
    
    holidays = pd.tseries.holiday.USFederalHolidayCalendar().holidays(start='2012', end='2022').to_pydatetime()
    
    holidays_date = [holiday.date() for holiday in holidays]
    
    def isitaholiday(date):
        ''' super useful function'''
        if date.date() in holidays_date:
            return 1
        else: 
            return 0
    
    crashes['IS_A_HOLIDAY'] = crashes['DATE_ACCIDENT'].apply(isitaholiday)
    
    crashes['HOLIDAY_NAME'] = crashes['DATE_ACCIDENT'].apply(isitaholiday)
    
    crashes.drop(['CRASH_DATE'], axis = 1, inplace = True)
    
    people = people.drop(columns=['CELL_PHONE_USE', 'BAC_RESULT VALUE', 'PEDPEDAL_LOCATION', 'PEDPEDAL_VISIBILITY',
                              'EMS_RUN_NO', 'EMS_AGENCY', 'HOSPITAL', 'DRIVERS_LICENSE_CLASS', 
                              'DRIVERS_LICENSE_STATE', 'ZIPCODE', 'SEAT_NO', 'PEDPEDAL_ACTION'])
    
    people.dropna(inplace=True)
    
    vehicles = vehicles.drop(columns=['NUM_PASSENGERS', 'CMRC_VEH_I', 'TOWED_I', 'FIRE_I', 'EXCEED_SPEED_LIMIT_I', 
                                  'TOWED_BY', 'TOWED_TO', 'AREA_00_I', 'AREA_01_I', 'AREA_02_I', 'AREA_03_I', 
                                  'AREA_04_I', 'AREA_05_I', 'AREA_06_I', 'AREA_07_I', 'AREA_08_I', 'AREA_09_I', 
                                  'AREA_10_I', 'AREA_11_I', 'AREA_12_I', 'AREA_99_I', 'CMV_ID', 'USDOT_NO', 'CCMC_NO', 
                                  'ILCC_NO', 'COMMERCIAL_SRC', 'GVWR', 'CARRIER_NAME', 'CARRIER_STATE', 'CARRIER_CITY',
                                  'HAZMAT_PLACARDS_I', 'HAZMAT_NAME', 'UN_NO', 'HAZMAT_PRESENT_I', 'HAZMAT_REPORT_I',
                                  'HAZMAT_REPORT_NO', 'MCS_REPORT_I', 'MCS_REPORT_NO', 'HAZMAT_VIO_CAUSE_CRASH_I',
                                  'MCS_VIO_CAUSE_CRASH_I', 'IDOT_PERMIT_NO', 'WIDE_LOAD_I', 'TRAILER1_WIDTH',
                                  'TRAILER2_WIDTH', 'TRAILER1_LENGTH', 'TRAILER2_LENGTH', 'TOTAL_VEHICLE_LENGTH',
                                  'AXLE_CNT', 'VEHICLE_CONFIG', 'CARGO_BODY_TYPE', 'LOAD_TYPE', 'HAZMAT_OUT_OF_SERVICE_I',
                                  'MCS_OUT_OF_SERVICE_I', 'HAZMAT_CLASS', 'LIC_PLATE_STATE'])
    
    vehicles.dropna(inplace=True)
    
    vehicles = vehicles.drop(columns=['RD_NO', 'VEHICLE_ID', 'CRASH_DATE'])
    
    inner_merged_total = pd.merge(vehicles, crashes, on=['CRASH_RECORD_ID'])
    
    inner_merged_total = pd.merge(inner_merged_total, people, on=['CRASH_RECORD_ID'])
    
    df = inner_merged_total.sort_values(by=['MOST_SEVERE_INJURY'], ascending=False)
    
    df = df[118131:]
    
    df['MOST_SEVERE_INJURY'] = np.where(df['MOST_SEVERE_INJURY']=='NO INDICATION OF INJURY',
                                         'aNO INDICATION OF INJURY', df['MOST_SEVERE_INJURY'])
    
    df = df.sort_values(by=['MOST_SEVERE_INJURY'], ascending=False)
    
    df = df[845359:]
    
    df['MOST_SEVERE_INJURY'] = np.where(df['MOST_SEVERE_INJURY']=='aNO INDICATION OF INJURY',
                                         'NO INDICATION OF INJURY', df['MOST_SEVERE_INJURY'])
    
    df_a = df
    
    def basic_info(data):
        categorical = []
        numerical = []
        for i in data.columns:
            if data[i].dtype == object:
                categorical.append(i)
            else:
                numerical.append(i)
        return categorical, numerical

    categorical, numerical = basic_info(df_a)
    
    df_a2 = df_a.copy(deep = True)
    
    df1 = df.drop(columns=['MOST_SEVERE_INJURY'])
    
    categoricalx, numericalx = basic_info(df1)
    
    df3 = df1[categoricalx]
            
    df4 = df3.drop(columns=['CRASH_RECORD_ID','MAKE','MODEL','DATE_POLICE_NOTIFIED','STREET_NAME','PERSON_ID','RD_NO','CRASH_DATE','CITY'])
    
    categoricalx, numericalx = basic_info(df4)
    
    df_a2 = df_a2.drop(columns=['CRASH_RECORD_ID', 'MODEL', 'DATE_POLICE_NOTIFIED', 'STREET_NAME', 'PERSON_ID', 'RD_NO', 'CRASH_DATE', 'CITY', 'STATE'])
    
    df_a2 = df_a2.drop(columns=['CRASH_UNIT_ID','MAKE','VEHICLE_YEAR','STREET_NO','BEAT_OF_OCCURRENCE','LATITUDE','LONGITUDE','DATE_ACCIDENT','VEHICLE_ID'])
    
    categorical2 = df_a2.columns
    
    # may need to fix
    df_a4 = df_a2.drop(columns=['UNIT_TYPE','VEHICLE_DEFECT','VEHICLE_TYPE','VEHICLE_USE','TRAVEL_DIRECTION','MANEUVER', 'FIRST_CONTACT_POINT', 'TRAFFIC_CONTROL_DEVICE','DEVICE_CONDITION','WEATHER_CONDITION','LIGHTING_CONDITION','FIRST_CRASH_TYPE','TRAFFICWAY_TYPE','ALIGNMENT','ROADWAY_SURFACE_COND','ROAD_DEFECT','REPORT_TYPE','CRASH_TYPE','DAMAGE','PRIM_CONTRIBUTORY_CAUSE','SEC_CONTRIBUTORY_CAUSE','STREET_DIRECTION', 'MOST_SEVERE_INJURY'])
    
    ds = df.drop(columns=['MOST_SEVERE_INJURY'])
    
    deletar = []
    for x in ds.columns:
        if len(ds[x].value_counts()) > 50:
                deletar.append(x)
                
    ds1 = ds.drop(columns=['CRASH_UNIT_ID','CRASH_RECORD_ID','MAKE','MODEL','VEHICLE_YEAR','DATE_POLICE_NOTIFIED','STREET_NO','STREET_NAME','BEAT_OF_OCCURRENCE','LATITUDE','LONGITUDE','DATE_ACCIDENT','PERSON_ID', 'RD_NO', 'VEHICLE_ID','CRASH_DATE','CITY','AGE'])
    
    ds1 = ds1.drop(columns=['INJURIES_TOTAL', 'INJURIES_FATAL', 'INJURIES_INCAPACITATING','INJURIES_NON_INCAPACITATING','INJURIES_REPORTED_NOT_EVIDENT','INJURIES_NO_INDICATION', 'INJURIES_UNKNOWN'])
    
    ds1 = ds1.drop(columns=['INJURY_CLASSIFICATION'])
    
    dfs = df
    
    dfs['MOST_SEVERE_INJURY'] = np.where(dfs['MOST_SEVERE_INJURY']=='NONINCAPACITATING INJURY', 0, dfs['MOST_SEVERE_INJURY'])
    
    dfs['MOST_SEVERE_INJURY'] = np.where(dfs['MOST_SEVERE_INJURY']=='NO INDICATION OF INJURY', 1, dfs['MOST_SEVERE_INJURY'])
    
    dfs['MOST_SEVERE_INJURY'] = np.where(dfs['MOST_SEVERE_INJURY']=='INCAPACITATING INJURY/FATAL', 2, dfs['MOST_SEVERE_INJURY'])
    
    col_mask = ['UNIT_TYPE', 'VEHICLE_DEFECT', 'VEHICLE_TYPE', 'VEHICLE_USE', 'TRAVEL_DIRECTION', 'MANEUVER', 'FIRST_CONTACT_POINT', 'TRAFFIC_CONTROL_DEVICE', 'DEVICE_CONDITION', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'FIRST_CRASH_TYPE', 'TRAFFICWAY_TYPE', 'ALIGNMENT', 'ROADWAY_SURFACE_COND', 'ROAD_DEFECT', 'REPORT_TYPE', 'CRASH_TYPE', 'DAMAGE', 'PRIM_CONTRIBUTORY_CAUSE', 'SEC_CONTRIBUTORY_CAUSE', 'STREET_DIRECTION', 'PERSON_TYPE', 'STATE', 'SEX', 'SAFETY_EQUIPMENT', 'AIRBAG_DEPLOYED', 'EJECTION', 'DRIVER_ACTION', 'DRIVER_VISION', 'PHYSICAL_CONDITION', 'BAC_RESULT', 'MOST_SEVERE_INJURY']
    
    model_data = dfs[col_mask]
    categoricalz, numericalz = basic_info(model_data)
    
    ############### FUNC RETURN
    return model_data, categoricalz, numericalz

def show_feature_importances(clf):
    '''Takes in XGBoost clf only'''
    listy=clf.get_booster().feature_names
    listy=(X.columns)
    listp=clf.get_booster().get_score(importance_type = 'gain')


    counter = 0
    for k,v in listp.items():
        listp[k] = [v, listy[counter]]
        counter += 1
        
    return listp
    
