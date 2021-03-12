from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report
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

def kfold_validation(X_train, y_train):
    from sklearn.model_selection import KFold
    from sklearn.metrics import recall_score

    kf = KFold()

    val_recall = []

    for train_ind, val_ind in kf.split(X_train, y_train):
        x_t = X_train.iloc[train_ind]
        y_t = y_train.iloc[train_ind]

        ss = StandardScaler()
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

        cont = ['TIME']
        cat_cols = list(tts_data[0]['train']['X'].drop(labels=['VEHICLE_ID', 'TIME'], axis=1).columns)

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

        lr = LogisticRegression()

        lr.fit(x_t, y_t)

        val_recall.append(recall_score(y_val, lr.predict(x_val)))


######################## CHRISTOS ####################

# from sklearn import preprocessing
# lbl = preprocessing.LabelEncoder()
# df['WEATHER_CONDITION'] = lbl.fit_transform(df['WEATHER_CONDITION'].astype(str))
# df['LIGHTING_CONDITION'] = lbl.fit_transform(df['LIGHTING_CONDITION'].astype(str))
# df['ROADWAY_SURFACE_COND'] = lbl.fit_transform(df['ROADWAY_SURFACE_COND'].astype(str))
# df['ROAD_DEFECT'] = lbl.fit_transform(df['ROAD_DEFECT'].astype(str))
# df['STREET_DIRECTION'] = lbl.fit_transform(df['STREET_DIRECTION'].astype(str))

# X = df.drop('target-injuries', axis=1)
# y = df['target-injuries']


# import warnings
# warnings.filterwarnings('ignore')
# import numpy as np
# import pandas as pd
# from datetime import datetime
# from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import StratifiedKFold
# from xgboost import XGBClassifier
# import xgboost as xgb

# print('\n All results:')
# print(random_search.cv_results_)
# print('\n Best estimator:')
# print(random_search.best_estimator_)
# print(random_search.best_score_)
# print('\n Best hyperparameters:')
# print(random_search.best_params_)
# results = pd.DataFrame(random_search.cv_results_)
# results.to_csv('xgb-random-grid-search-results-01.csv', index=False)


# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 1001)

# random_search = RandomizedSearchCV(xgb,
#                                    param_distributions=params,
#                                    n_iter=param_comb,
#                                    scoring='roc_auc',
#                                    n_jobs=4,
#                                    cv=skf.split(X_train,Y_train),
#                                    verbose=3,
#                                    random_state=1001 )

# random_search.fit(X_train, Y_train)

# from sklearn.model_selection import KFold
# from sklearn.metrics import recall_score

# kf = KFold()

# val_recall = []

# for train_ind, val_ind in kf.split(X_train, y_train):
#     x_t = X_train.iloc[train_ind]
#     y_t = y_train.iloc[train_ind]
    
#     ss = StandardScaler()
#     ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    
#     cont = ['TIME']
#     cat_cols = list(tts_data[0]['train']['X'].drop(labels=['VEHICLE_ID', 'TIME'], axis=1).columns)
    
#     scaled = ss.fit_transform(x_t[cont])
#     dummies = ohe.fit_transform(x_t[cat_cols])
    
#     x_t = pd.concat([pd.DataFrame(scaled, columns=cont), 
#                               pd.DataFrame(dummies, columns=ohe.get_feature_names())], axis=1)
    
    
#     x_val = X_train.iloc[val_ind]
#     y_val = y_train.iloc[val_ind]
    
#     sc = ss.transform(x_val[cont])
#     dums = ohe.transform(x_val[cat_cols])
    
#     x_val = pd.concat([pd.DataFrame(sc, columns=cont), 
#                               pd.DataFrame(dums, columns=ohe.get_feature_names())], axis=1)
    
#     lr = LogisticRegression()
    
#     lr.fit(x_t, y_t)
    
#     val_recall.append(recall_score(y_val, lr.predict(x_val)))
    
