import pandas as pd
import numpy as np
from timeit import default_timer as timer
import datetime
import re
import itertools
import functools
import time
import os
import copy
import matplotlib.pyplot as plt
from scipy.stats.mstats import winsorize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from lightgbm import LGBMClassifier
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
import warnings
from optuna.exceptions import ExperimentalWarning
import logging
import sys
import joblib
from multiprocess import Pool
from functools import partial
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.feature_selection import mutual_info_classif
import shap



def summary_stats(df=None, date_format='D', n_digits=2):

    '''
      date_format: show dates up to days https://numpy.org/doc/stable/reference/arrays.datetime.html#arrays-dtypes-dateunits
      n_digits: rounding digits for min, max, mean, ...
    '''

    import pandas as pd
    import numpy as np
    import warnings
    
    NUMERICS = ['number', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    DATES = ['datetimetz', 'datetime64', 'datetime']
    CATEGORICALS = ['object']
    BOOLS = ['bool']
    all_cols = df.columns

    tot_rows, tot_cols = df.shape
    pd.set_option('display.max_rows', tot_cols)

    num_cols = df.select_dtypes(include=NUMERICS).columns
    date_cols = df.select_dtypes(include=DATES).columns
    cat_cols = df.select_dtypes(include=CATEGORICALS).columns
    bool_cols = df.select_dtypes(include=BOOLS).columns

    # convert int64 to float (so as to evaluate mean, etc)
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(float)    
    
    # Numerical stats
    num_stats = pd.DataFrame(columns=['VARIABLE', 'TYPE', 'OBS', 'UNIQUE', 'NANs', 'INFs', 'ZEROs', 'MIN', 'MAX', 'MEAN',
                                      'STDDEV', 'MEDIAN', 'PERC1', 'PERC5', 'PERC95', 'PERC99', 'SUM'])
    for var in num_cols:

        val = df[var].values
        val = val[~np.isnan(val)]
        val = val[~np.isinf(val)]

        warnings.filterwarnings("ignore")
        perc = np.quantile(val, [0.01, 0.05, 0.95, 0.99])
        median = np.round(np.median(val), n_digits)
        warnings.filterwarnings("default")

        add_row = pd.DataFrame({'VARIABLE': var,
                                'TYPE': str(df[var].dtypes),
                                'OBS': tot_rows,
                                'UNIQUE': df[var].nunique(),
                                'NANs': df[var].isna().sum(),
                                'INFs': np.isinf(df[var].values).sum(),
                                'ZEROs': sum(df[var] == 0),
                                'MIN': np.round(val.min(), n_digits),
                                'MAX': np.round(val.max(), n_digits),
                                'MEAN': np.round(val.mean(), n_digits),
                                'STDDEV': np.round(val.std(), n_digits),
                                'MEDIAN': median,
                                'PERC1': np.round(perc[0], n_digits),
                                'PERC5': np.round(perc[1], n_digits),
                                'PERC95': np.round(perc[2], n_digits),
                                'PERC99': np.round(perc[3], n_digits),
                                'SUM': np.round(val.sum(), n_digits)
                               }, index = [0])

        num_stats = pd.concat([num_stats, add_row])


    # Categorical stats
    cat_stats = pd.DataFrame(columns=['VARIABLE', 'TYPE', 'OBS', 'UNIQUE', 'NANs', 'BLANKs', 'VALUES_BY_FREQ'])

    for var in cat_cols:

        add_row = pd.DataFrame({'VARIABLE': var,
                                'TYPE': str(df[var].dtypes),
                                'OBS': tot_rows,
                                'UNIQUE': df[var].nunique(),
                                'NANs': df[var].isna().sum(),
                                'BLANKs': sum(df[var] == ''),
                                'VALUES_BY_FREQ': '|'.join(df[var].value_counts().index[:40].astype(str))
                               }, index = [0])

        cat_stats = pd.concat([cat_stats, add_row])


    # Boolean stats
    bool_stats = pd.DataFrame(columns=['VARIABLE', 'TYPE', 'OBS', 'NANs', 'MEAN', 'STDDEV', 'MEDIAN', 'PERC1',
                                       'PERC5', 'PERC95', 'PERC99', 'SUM', 'VALUES_BY_FREQ'])
    for var in bool_cols:

        val = df[var].dropna().values

        warnings.filterwarnings("ignore")
        perc = np.quantile(val.astype(int), [0.01, 0.05, 0.95, 0.99])
        median = np.round(np.median(val), n_digits)
        warnings.filterwarnings("default")        
                                                   
        add_row = pd.DataFrame({'VARIABLE': var,
                                'TYPE': str(df[var].dtypes),
                                'OBS': tot_rows,
                                'NANs': df[var].isna().sum(),
                                'MEAN': np.round(val.mean(), n_digits),
                                'STDDEV': np.round(val.std(), n_digits),
                                'MEDIAN': median,
                                'PERC1': perc[0],
                                'PERC5': perc[1],
                                'PERC95': perc[2],
                                'PERC99': perc[3],
                                'SUM': val.sum(),
                                'VALUES_BY_FREQ': ', '.join([str(k) + ': ' + str(v)
                                                             for k, v in df[var].dropna().value_counts().to_dict().items()])
                               }, index = [0])

        bool_stats = pd.concat([bool_stats, add_row])

    # Date stats
    date_stats = pd.DataFrame(columns=['VARIABLE', 'TYPE', 'OBS', 'UNIQUE', 'NANs', 'MIN', 'MAX', 'MEDIAN', 'PERC1',
                                       'PERC5', 'PERC95', 'PERC99'])
    for var in date_cols:

        val = df[var].dropna().values
        val_str = np.datetime_as_string(val, unit=date_format)

        # calculation for median and quantile
        val_cnt = sorted(pd.Series(val_str).unique())
        mapping = dict(zip(val_cnt, range(len(val_cnt))))
        mapped = [mapping.get(v, v) for v in val_str]
        warnings.filterwarnings("ignore")
        med_ind = np.median(mapped).astype(int)
        warnings.filterwarnings("default")  

        if len(val) > 0:
        
            warnings.filterwarnings("ignore")
            perc = np.quantile(mapped, [0.01, 0.05, 0.95, 0.99]).astype(int)
            median = [k for k, v in mapping.items() if v == med_ind][0]
            warnings.filterwarnings("default")  
            add_row = pd.DataFrame({'VARIABLE': var,
                                    'TYPE': str(df[var].dtypes),
                                    'OBS': tot_rows,
                                    'UNIQUE': df[var].nunique(),
                                    'NANs': df[var].isna().sum(),
                                    'MIN': np.datetime_as_string(val.min(), unit=date_format),
                                    'MAX': np.datetime_as_string(val.max(), unit=date_format),
                                    'MEDIAN': median,
                                    'PERC1': [k for k, v in mapping.items() if v == perc[0]][0],
                                    'PERC5': [k for k, v in mapping.items() if v == perc[1]][0],
                                    'PERC95': [k for k, v in mapping.items() if v == perc[2]][0],
                                    'PERC99': [k for k, v in mapping.items() if v == perc[3]][0]
                                   }, index = [0])
        else:
            add_row = pd.DataFrame({'VARIABLE': var,
                            'TYPE': str(df[var].dtypes),
                            'OBS': tot_rows,
                            'UNIQUE': df[var].nunique(),
                            'NANs': df[var].isna().sum()
                       }, index = [0])

        date_stats = pd.concat([date_stats, add_row])

    # final stats
    all_col_set = ['VARIABLE', 'TYPE', 'OBS', 'UNIQUE', 'NANs', 'INFs', 'ZEROs', 'BLANKs', 'MEAN', 'STDDEV', 'MIN', 
                                       'PERC1', 'PERC5', 'MEDIAN', 'PERC95', 'PERC99', 'MAX', 'SUM', 'VALUES_BY_FREQ']
    used_col_set = []
    final_stats = pd.DataFrame(columns=all_col_set)
    if num_stats.shape[0] > 0:
        final_stats = pd.concat([final_stats, num_stats])
        used_col_set.extend(num_stats.columns)
    if cat_stats.shape[0] > 0:
        final_stats = pd.concat([final_stats, cat_stats])
        used_col_set.extend(cat_stats.columns)
    if bool_stats.shape[0] > 0:
        final_stats = pd.concat([final_stats, bool_stats])
        used_col_set.extend(bool_stats.columns)
    if date_stats.shape[0] > 0:
        final_stats = pd.concat([final_stats, date_stats])
        used_col_set.extend(date_stats.columns)

    final_stats = final_stats[[x for x in all_col_set if x in np.unique(used_col_set)]]

    if final_stats['VARIABLE'].nunique() != final_stats.shape[0]:
        print('-- Duplicated variables found!')

    if final_stats['VARIABLE'].nunique() != tot_cols:
        print('-- Missing variables found:\n    ', '\n     '.join(set(all_cols) - set(final_stats['VARIABLE'].values)))

    final_stats['order'] = pd.Categorical(final_stats['VARIABLE'], categories = all_cols, ordered = True)
    final_stats = final_stats.sort_values(by='order').drop(columns=['order'])

    # add percentage to missing, inf, zeros, blank
    if 'NANs' in final_stats.columns:
        final_stats['NANs'] = [str(x) + ' (' +  str(np.round(x / tot_rows * 100, 1)) + '%)'
                               if not np.isnan(x) else '' for x in final_stats['NANs'].values]
    if 'INFs' in final_stats.columns:
        final_stats['INFs'] = [str(x) + ' (' +  str(np.round(x / tot_rows * 100, 1)) + '%)'
                               if not np.isnan(x) else '' for x in final_stats['INFs'].values]
    if 'ZEROs' in final_stats.columns:
        final_stats['ZEROs'] = [str(x) + ' (' +  str(np.round(x / tot_rows * 100, 1)) + '%)'
                               if not np.isnan(x) else '' for x in final_stats['ZEROs'].values]
    if 'BLANKs' in final_stats.columns:
        final_stats['BLANKs'] = [str(x) + ' (' +  str(np.round(x / tot_rows * 100, 1)) + '%)'
                               if not np.isnan(x) else '' for x in final_stats['BLANKs'].values]
    final_stats.fillna('', inplace=True)
    
    return final_stats


def stats_with_description(df, df_vardescr_path, col_to_lowercase=True, lag_label=None, pc_label=None):
    
    df_stats = summary_stats(df)
    df_vardescr = pd.read_csv(df_vardescr_path, sep=';').drop(columns=['Type'])
    if col_to_lowercase:
        df_vardescr['Variable Name'] = df_vardescr['Variable Name'].str.lower()
    if lag_label is not None:
        dd=df_vardescr.copy()
        dd['Variable Name'] = dd['Variable Name'] + lag_label
        dd['Description'] = 'Lag of ' + dd['Description']
        df_vardescr = pd.concat([df_vardescr, dd])
    if pc_label is not None:
        dd=df_vardescr.copy()
        dd['Variable Name'] = dd['Variable Name'] + pc_label
        dd['Description'] = 'Percentage change of ' + dd['Description']
        ddf_vardescr = pd.concat([df_vardescr, dd])
    df_stats = df_stats.merge(df_vardescr, how='left', left_on='VARIABLE', right_on='Variable Name').drop(columns=['Variable Name'])
    move_col = df_stats.pop('Description')
    df_stats.insert(1, 'Description', move_col)
    df_stats.fillna('', inplace=True)
    
    return df_stats


def safe_last(x, missing_char = ''):
    
    x = x[x != missing_char]
    if len(x) > 0:
        return x.values[-1]
    else:
        return missing_char
    
    
def match_gvkey(df, df_link_final, col_to_check = ['cusip', 'ticker']):
    
    mapping = {'cusip': 'cusip',       # map to df_link_final column names  {'current_table': df_link_final}
          'ticker': 'tic',
          'permno': 'LPERMNO'}
        
    n_row = df.shape[0]
    df['xxx_ID'] = np.arange(len(df))
    start = timer()
    for col in col_to_check:
        t_df = df.copy().merge(df_link_final[['gvkey', mapping[col]]], left_on=col, right_on=mapping[col], how='left').groupby('xxx_ID')['gvkey'].agg(xxx=(lambda x: np.unique(x.dropna()).tolist())).reset_index()
        t_df['type_'+col] = t_df.apply(lambda x: [col] if len(x['xxx']) > 0 else [], axis=1)
        t_df.rename(columns={'xxx': 'gvkey_' + col}, inplace=True)
        df=df.merge(t_df, on='xxx_ID', how='left')

    def f(x):
        x = [i for i in x if len(i) > 0]
        if len(x) > 0:
            return x[0]
        else:
            return []   
        
#     df['gvkey'] = df[['gvkey_' + x for x in col_to_check]].apply(lambda x: np.unique(x.sum()).tolist(), axis = 1)
#     df['match_type'] = df[['type_' + x for x in col_to_check]].apply(lambda x: np.unique(x.sum()).tolist(), axis = 1)
    df['gvkey'] = df[['gvkey_' + x for x in col_to_check]].apply(f, axis = 1)
    df['match_type'] = df[['type_' + x for x in col_to_check]].apply(f, axis = 1)
    df['gvkey_tot'] = df['gvkey'].str.len()
    df.drop(columns=['xxx_ID'] + ['gvkey_' + x for x in col_to_check] + ['type_' + x for x in col_to_check], inplace=True)
    first_cols = ['gvkey', 'match_type', 'gvkey_tot']
    df = df[first_cols + [x for x in df.columns if x not in first_cols]]
    
    if n_row != df.shape[0]:
        print('#### wrong expected number of rows')
    
    # stats
    print('Total elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))
    v, c = np.unique(df.match_type, return_counts=True)
    print('\n- Matching procedure:')
    display(pd.DataFrame({'match_type': v, 'count': c}))
    
    print('\n- Multiple gvkey count:')
    display(df.gvkey_tot.value_counts().to_frame().reset_index().rename(columns={'index': 'tot_gvkey', 'gvkey_tot': 'count'}))
    
    multiple_gvkey = np.unique(df[df['gvkey_tot'] > 1]['gvkey'])
    multiple_gvkey_flat = list(itertools.chain(*multiple_gvkey))
    v,c=np.unique(multiple_gvkey_flat, return_counts=True)
    multiple_gvkey_check = pd.DataFrame({'val': v, 'count': c}).sort_values(by='count', ascending=False).query('count > 1')
    if multiple_gvkey_check.shape[0] > 0:
        print('\n- Multiple gvkey with shared values:', multiple_gvkey_check.shape[0])
    
    return df


def merge_df(df_left, df_right, left_on_key = 'year', right_on_key = 'year', right_columns = ['year', 'value'],
            duplicated_groupby_check = 'year', check_duplicates_right=True, warn=True):
    
    '''
        right_columns: columns to keep when merging df_right
        duplicated_groupby_check: column to groupby multiple gvkey so as to select most populated one and remove duplicates
    '''

    cc = 0
    ref_tab = pd.DataFrame({'gvkey': [], 'ID': []})

    # assign unique ID to multiple gvkey istances
    df_mul = df_left[df_left.gvkey_tot > 1]
    if df_mul.shape[0] > 0:
        mul_list = np.unique(df_mul['gvkey'])

        for el in mul_list:
            ref_tab = pd.concat([ref_tab, pd.DataFrame({'gvkey': el, 'ID': str(cc)})])
            cc += 1

        check_multiple_assignment = ref_tab.groupby('gvkey').agg(count=('ID', lambda x: len(x)),
                                        uniqueID=('ID', lambda x: list(x))).reset_index().query('count > 1')
        if check_multiple_assignment.shape[0] > 0:
            for index, row in check_multiple_assignment.iterrows():
                ref_tab['ID'][ref_tab['ID'].isin(row['uniqueID'])] = min(row['uniqueID'])

        ref_tab.drop_duplicates(inplace=True)

    # append unique ID for single gvkey istances
    sin_list = np.unique(df_left[df_left.gvkey_tot == 1]['gvkey'])
    sin_list = list(itertools.chain(*sin_list))
    t_sin = pd.DataFrame({'gvkey': sin_list, 'ID': ''})
    t_sin = t_sin[~t_sin['gvkey'].isin(ref_tab['gvkey'])]
    t_sin['ID'] = [str(x) for x in np.arange(cc, cc + len(t_sin))]
    ref_tab=pd.concat([ref_tab, t_sin])
    ref_tab = dict(zip(ref_tab['gvkey'], ref_tab['ID']))
    
    # map ID on df_left
    df_left.insert(0, 'xxx_ID', df_left['gvkey'].apply(lambda x: np.unique([y for y in [ref_tab.get(i) for i in x] if y is not None])))
    if np.unique([len(x) for x in df_left['xxx_ID']]).max() > 1:
        print('##### warning: multiple unique ID mapped to gvkey - taken only first value on df_left')
    df_left['xxx_ID'] = df_left['xxx_ID'].str[0]
    
    # remove duplicated unique ID xxx_ID for single gvkey_original
    check_ID_left = df_left[['xxx_ID', 'gvkey_original']].drop_duplicates()
    check_ID_left_dupl_gvkey = check_ID_left['gvkey_original'].value_counts().to_frame().reset_index().query('gvkey_original > 1')['index'].values
    if len(check_ID_left_dupl_gvkey) > 0:
        check_ID_left_dupl_ID = check_ID_left[check_ID_left['gvkey_original'].isin(check_ID_left_dupl_gvkey)]['xxx_ID'].values
        row_counts=df_left.shape[0]
        df_left = df_left.copy()[~df_left['xxx_ID'].isin(check_ID_left_dupl_ID)]
        ref_tab = {k: v for k, v in ref_tab.items() if v not in check_ID_left_dupl_ID}
        print('- Removed rows from df_left because of multiple unique ID on same gvkey_original:', row_counts - df_left.shape[0])
    
    # map ID on df_right
    df_right = df_right[df_right['gvkey_tot'] > 0]
    df_right.insert(0, 'xxx_ID', df_right['gvkey'].apply(lambda x: np.unique([y for y in [ref_tab.get(i) for i in x] if y is not None])))
    if np.unique([len(x) for x in df_right['xxx_ID']]).max() > 1:
        print('##### warning: multiple unique ID mapped to gvkey - taken only first value on df_right')
    pd.options.mode.chained_assignment = None  # default='warn'
    df_right.loc[:, ['xxx_ID']] = df_right['xxx_ID'].str[0]
    pd.options.mode.chained_assignment = 'warn'  # default='warn'
    df_right = df_right[~df_right['xxx_ID'].isnull()]

    # solve duplicated rows (some company may have been split during the same year) in df_left
    check_duplicates=df_left[['xxx_ID']+[duplicated_groupby_check]].groupby(['xxx_ID', duplicated_groupby_check]).size().reset_index().rename(columns={0:'val'})
    check_duplicates=check_duplicates[check_duplicates['val'] > 1]
    if check_duplicates.shape[0] > 0:
        print('\n-- dropping duplicated values in df_left for "' + duplicated_groupby_check + '" - duplicates are evaluated according to "gvkey_original"')
        index_to_remove = []
        for id in check_duplicates['xxx_ID'].unique():
            df_t=df_left.copy()[df_left['xxx_ID'] == id].copy()
    #         df_t['gvkey']=df_t['gvkey'].apply(lambda x: '_'.join(x))
            majority_gvkey=df_t['gvkey_original'].value_counts().to_frame().sort_values(by='gvkey_original', ascending=False).reset_index()
            gvkey_to_keep=majority_gvkey.iloc[0]['index']
            duplicated_val=df_t.groupby(duplicated_groupby_check).size().to_frame().reset_index().reset_index().rename(columns={0:'val'})
            duplicated_val=duplicated_val[duplicated_val['val'] > 1][duplicated_groupby_check].values
    #         allowed_gvkey=np.unique(df_t[~df_t[duplicated_groupby_check].isin(duplicated_val)]['gvkey_original'].values)
    #         mask = (~df_t[duplicated_groupby_check].isin(duplicated_val)) | ((df_t['gvkey_original'].isin(allowed_gvkey)) & (df_t[duplicated_groupby_check].isin(duplicated_val)))
            mask = (~df_t[duplicated_groupby_check].isin(duplicated_val)) | ((df_t['gvkey_original'] == gvkey_to_keep) & (df_t[duplicated_groupby_check].isin(duplicated_val)))
            index_to_remove.extend(mask[mask==False].index)
        df_left=df_left.drop(index_to_remove, axis=0)

        check_duplicates=df_left[['xxx_ID']+[duplicated_groupby_check]].groupby(['xxx_ID', duplicated_groupby_check]).size().reset_index().rename(columns={0:'val'})
        if check_duplicates[check_duplicates['val'] > 1].shape[0] > 0:
            print('    #### duplicates still present')

    # solve duplicated rows (some company may have been split during the same year) in df_right
    if check_duplicates_right:
        check_duplicates=df_right[['xxx_ID']+right_columns].groupby(['xxx_ID',duplicated_groupby_check]).size().reset_index().rename(columns={0:'val'})
        check_duplicates=check_duplicates[check_duplicates['val'] > 1]
        if check_duplicates.shape[0] > 0:
            print('\n-- dropping duplicated values in df_right for "' + duplicated_groupby_check + '" - duplicates are evaluated according to "gvkey"')
            index_to_remove = []
            for id in check_duplicates['xxx_ID'].unique():
                df_t=df_right.copy()[df_right['xxx_ID'] == id].copy()
                df_t['gvkey']=df_t['gvkey'].str[0]
                majority_gvkey=df_t['gvkey'].value_counts().to_frame().sort_values(by='gvkey', ascending=False).reset_index()
                gvkey_to_keep=majority_gvkey.iloc[0]['index']
                duplicated_val=df_t.groupby(duplicated_groupby_check).size().to_frame().reset_index().reset_index().rename(columns={0:'val'})
                duplicated_val=duplicated_val[duplicated_val['val'] > 1][duplicated_groupby_check].values
    #             allowed_gvkey=np.unique(df_t[~df_t[duplicated_groupby_check].isin(duplicated_val)]['gvkey'].values)
    #             mask = (~df_t[duplicated_groupby_check].isin(duplicated_val)) | ((df_t['gvkey'].isin(allowed_gvkey)) & (df_t[duplicated_groupby_check].isin(duplicated_val)))
                mask = (~df_t[duplicated_groupby_check].isin(duplicated_val)) | ((df_t['gvkey'] == gvkey_to_keep) & (df_t[duplicated_groupby_check].isin(duplicated_val)))
                index_to_remove.extend(mask[mask==False].index)
            df_right=df_right.drop(index_to_remove, axis=0)

        check_duplicates=df_right[['xxx_ID']+[duplicated_groupby_check]].groupby(['xxx_ID', duplicated_groupby_check]).size().reset_index().rename(columns={0:'val'})
        if check_duplicates[check_duplicates['val'] > 1].shape[0] > 0:
            print('    #### duplicates still present')
            
    # merge dataset by "xxx_ID" and left_on_key / right_on_key
    if type(left_on_key) != list:
        left_on_key = [left_on_key]
    if type(right_on_key) != list:
        right_on_key = [right_on_key]
    df_merge = df_left.copy()
    df_merge = df_merge.merge(df_right[['xxx_ID']+right_columns], left_on=['xxx_ID']+left_on_key, right_on=['xxx_ID']+right_on_key, how='left')

    if df_left.shape[0] != df_merge.shape[0] and warn:
        print('##### warning: merged dataset has different number of rows (' + str(df_merge.shape[0]) +
              ') with respect to df_left (' + str(df_left.shape[0]) + ')')

#     df_merge.drop(columns=['xxx_ID'], inplace=True)
    
    return df_merge, ref_tab


def find_consec_year(df, min_consecutive_years = 2, year_col = 'year', col_na_check = 'EPS_next_year'):
    
    lis = df.copy()[~df[col_na_check].isna()][year_col].values
    df['consec_year'] = False
    df['consec_year_group'] = -1
    if len(lis) > 0:
        consec_group = functools.reduce(lambda x,y : x[:-1]+[x[-1]+[y]] if (x[-1][-1]+1==y) else [*x,[y]], lis[1:] , [[lis[0]]] )
        year_to_keep = [x for x in consec_group if len(x) >= min_consecutive_years]
        year_groups = [[i] * len(year_to_keep[i]) for i in range(len(year_to_keep))]
        year_to_keep = [x for y in year_to_keep for x in y]
        year_groups = [x for y in year_groups for x in y]
        df.loc[df[year_col].isin(year_to_keep), 'consec_year'] = True
        df.loc[df[year_col].isin(year_to_keep), 'consec_year_group'] = year_groups

    return df


def safe_div(x, y):
    
    # 0/0 -> 0
    if x == 0 and y == 0:
        return 0
    # num/0 -> sign(num)
    elif y == 0:
        return (x > 0) - (x < 0)
    else:
        return x / y    

    
def plot_time_range(df, gr_by=['gvkey', 'consec_year_group'], time_var='fyear', label_space=10, fig_size=(17, 10),
                   min_obs=5, min_count=50, df_with_lagged_var=False):

    plot_data=(df.sort_values(time_var)
     .groupby(gr_by)
     .agg(count = (time_var, lambda x: list(x.astype(int).astype(str))[0 if df_with_lagged_var else 1] + '-' +
                   list(x.astype(int).astype(str))[-1]))
               ['count'].value_counts().to_frame().reset_index().rename(columns={'index': 'range'})
               .assign(size = lambda x: x['count'] / x['count'].sum(),
                      min = lambda x: x['range'].str[:4].astype(int),
                      max = lambda x: x['range'].str[5:].astype(int),
                      obs = lambda x: x['max'] - x['min'])
               .sort_values('obs', ascending = False).reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=fig_size)

    tot_perc = plot_data[(plot_data['obs'] >= min_obs) | (plot_data['count'] >= min_count)]['size'].sum()
    year_val = sorted(pd.concat([plot_data['min'], plot_data['max']]).unique())
    cc=0
    for index, row in plot_data.iterrows():
        if row['obs'] >= min_obs or row['count'] >= min_count:
            y_val = - label_space * cc
            min_val = row['min']
            max_val = row['max']
            if min_val == max_val:
                ax.scatter(min_val, y_val, linewidth = row['size'] * 150, color = 'blue')
            else:
                ax.plot([min_val, max_val], [y_val, y_val], linewidth = row['size'] * 100, color = 'blue',
                        alpha = (0.5 if row['size'] > 0.1 else 1))
            ax.text(year_val[0]-1, y_val, str(np.round(row['size'] * 100, 1)) + '% ', size = 10,
                    verticalalignment= ('top' if cc % 2 == 0 else 'bottom'),
                   horizontalalignment= ('left' if cc % 2 == 0 else 'right'))
            cc += 1
    ax.set_yticks([])
    ax.set_xlim(year_val[0]-2, year_val[-1]+1)
    ax.set_xticks(ticks=year_val, labels=year_val)
    plt.grid(True, axis='x')
    plt.title('Showing ' + str(np.round(tot_perc*100,1)) + '% of data')
    plt.show()
    
    
def make_features(df_work, df_original, thrsh_zeros = 0.3, file_name = '', col_keep = [], time_var = '',
                 CODING_TABLES_FOLDER = '', STATS_FOLDER = ''):
    
    '''
    Create lagged and year percentage change after filling missings with zeros and normalizing by Total Asset.
    Fundamentals variables are automatically read from "fundamentals_variables_list.csv".
    
    Args:
        df_work: dataset to used for features
        df_original: dataset used to subset df_work. Needed for lagged variables
        thrsh_zeros: columns with percentage of zeros >= thrsh_zeros will be removed
        file_name: name to be used when saving missing values percentage stats
        col_keep: columns to be kept aside of fundamentals variables
        time_var: variable to be used as time index
    '''

    # remove rows with Total Assets = 0
    row_count = df_work.shape[0]
    df_work = df_work[df_work['at'] > 0]
    if df_work.shape[0] != row_count:
        print('\n- Rows dropped because of zero Total Assets:', row_count - df_work.shape[0])

    # replace missing values with zeros
    fund_vars = pd.read_csv(os.path.join(CODING_TABLES_FOLDER, 'fundamentals_variables_list.csv'), sep=';')['Variable'].to_list()
    common_fund_vars = list(set(df_work.columns) & set(fund_vars))
    df_work.loc[:, common_fund_vars] = df_work.loc[:, common_fund_vars].fillna(0)
    print('\n- Replacing missing values with zeros')

    # remove empty columns and columns with % of zeros above thrsh_zeros
    zero_perc = pd.DataFrame(columns=['Variable', 'perc_zeros'])
    for var in common_fund_vars:
        zero_perc = pd.concat([zero_perc,pd.DataFrame({'Variable': var,
                                                  'perc_zeros': np.round(sum(df_work[var] == 0) / df_work.shape[0] * 100, 1)},
                                                 index=[0])])
    zero_perc = zero_perc.sort_values('perc_zeros').reset_index(drop=True)
    zero_perc['keep'] = np.where(zero_perc['perc_zeros'] >= thrsh_zeros * 100, '', 'x')
    col_remove = zero_perc[zero_perc['keep'] == '']['Variable'].to_list()
    keep_fund_vars = zero_perc[zero_perc['keep'] == 'x']['Variable'].to_list()
    df_work = df_work.drop(columns=col_remove)
    print('\n- Removed columns because percentage of zeros is above', thrsh_zeros*100, '%:', len(col_remove))
    (zero_perc.merge(pd.read_csv(os.path.join(CODING_TABLES_FOLDER, 'fundamentals_variables_list.csv'), sep=';'), on='Variable', how='left')
     .to_csv(os.path.join(STATS_FOLDER, '01_' + file_name + '_removed_col.csv'), index=False, sep=';'))
    print('\n- List of variables and percentages saved to', os.path.join(STATS_FOLDER, '01_' + file_name + '_removed_col.csv'))
    print('- Remaining columns:', len(common_fund_vars) - len(col_remove))

    # evaluate lag
    print('\n- Evaluating lagged variables')
    lag_df = (df_original[['main_index'] + keep_fund_vars].copy()
              .rename(columns=dict(zip(keep_fund_vars, [x + '_lag1' for x in keep_fund_vars])))
              .rename(columns={'main_index': 'LAG_index'}).fillna(0))

    df_work = df_work.merge(lag_df, on='LAG_index', how='left')[col_keep + keep_fund_vars + [x + '_lag1' for x in keep_fund_vars]]
    if df_work.drop(columns=col_keep).isna().any().sum() > 0:
        print('   ##### unmatched lagged variables found')

    # evaluate percentage change over consecutive years
    print('\n- Evaluating percentage change')
    for var in keep_fund_vars:
        df_work = df_work.assign(**{var + '_pc' : lambda x: (x[var] - x[var + '_lag1']) / x[var + '_lag1']})
    print('- Replacing NaN and Inf with zeros')
    df_work.loc[:, [x + '_pc' for x in keep_fund_vars]] = df_work.loc[:, [x + '_pc' for x in keep_fund_vars]].fillna(0)
    df_work.loc[:, [x + '_pc' for x in keep_fund_vars]] = df_work.loc[:, [x + '_pc' for x in keep_fund_vars]].replace([np.inf, -np.inf], 0)

    # normalize by total asset and replace total asset with log()
    print('\n- Normalizing by Total Asset')
    norm_var = [x for x in keep_fund_vars if x != 'at']
    df_work[norm_var] = df_work[norm_var].div(df_work['at'], axis=0)
    norm_var = [x for x in [x + '_lag1' for x in keep_fund_vars] if x != 'at_lag1']
    df_work[norm_var] = df_work[norm_var].div(df_work['at_lag1'], axis=0)
    np.seterr(divide = 'ignore') 
    df_work['at'] = np.log10(df_work['at'])
    df_work['at_lag1'] = np.log10(df_work['at_lag1'])
    np.seterr(divide = 'warn') 
    print('- Replacing Inf with zeros (some lagged total assets may be zero - only for oldest year available)')
    df_work.loc[:, [x + '_lag1' for x in keep_fund_vars]] = df_work.loc[:, [x + '_lag1' for x in keep_fund_vars]].fillna(0)
    df_work.loc[:, [x + '_lag1' for x in keep_fund_vars]] = df_work.loc[:, [x + '_lag1' for x in keep_fund_vars]].replace([np.inf, -np.inf], 0)

    # final check
    if df_work.drop(columns=col_keep).isna().any().sum() > 0:
        print('\n##### missing values still present:')
        print('\n   - '.join(df_work.drop(columns=col_keep).columns.values[df_work.drop(columns=col_keep).isna().sum() > 0]))

    if df_work.drop(columns=col_keep).isin([np.inf, -np.inf]).any().sum() > 0:
        print('\n##### Infinity values still present:')
        print('\n   - '.join(df_work.drop(columns=col_keep).columns.values[df_work.drop(columns=col_keep).isin([np.inf, -np.inf]).sum() > 0]))
        
    if df_work[col_keep].isna().any().sum() > 0:
        print('\n##### missing values found in:')
        miss = df_work[col_keep].isna().sum().loc[lambda x : x > 0]
        print('\n'.join(['   -' + a + ': ' + str(b) for a, b in zip(miss.index.values, miss.values)]))

    # time stats
    time_stat = df_work.groupby(time_var).agg(matched = (time_var, lambda x: len(x)))
    time_stat = pd.concat([time_stat, pd.DataFrame({col: time_stat[col].sum() for col in time_stat}, index=['Total'], columns=time_stat.columns)])
    display(time_stat)
    
    # reset index
    df_work = df_work.reset_index(drop=True)
    print('\n- Indices have been resetted')
    
    return df_work


def winsorize_and_scale(df, winsorize_perc=0.01, skip_cols=[]):
    
    '''
    Winsorize data and scale by magnitude
    '''
    
    for col in df.columns:
        if col not in skip_cols:
            # winsorize
            df[col] = winsorize(df[col], limits=[winsorize_perc, winsorize_perc])

            magnitude = 10 ** np.ceil(np.log10(max(abs(df[col]))))
            df[col] = df[col] / magnitude

    return df


class customCV:
    def __init__(self, n_train_years=2, n_test_years=1, rolling_wind_step=1, train_test_offset=0,
                 year_variable='fyear', back_ward=True, show_info=True):
        '''
        back_ward: if True backward evaluates test-train from last available year,
                    otherwise forward evaluates from oldest year
        '''
        
        self.n_train_years=n_train_years
        self.n_test_years=n_test_years 
        self.rolling_wind_step=rolling_wind_step
        self.train_test_offset=train_test_offset
        self.year_variable = year_variable
        self.back_ward=back_ward     
        self.show_info=show_info

    def determine_n_split(self, X):
        # determine number of splits
        
        avail_year = np.sort(X[self.year_variable].unique())
        cont=True
        split_batch = []
        if self.back_ward:
            end=len(avail_year)
            while cont:
                test_yr = range(end - self.n_test_years , end)
                train_yr = range(end - self.n_test_years - self.train_test_offset - self.n_train_years,
                                 end - self.n_test_years - self.train_test_offset)
                if min(train_yr) < 0:
                    cont = False
                else:
                    end -= self.rolling_wind_step
                    split_batch.append([avail_year[train_yr], avail_year[test_yr]])
            split_batch.reverse()

        else:
            start=0
            while cont:
                train_yr = range(start, start + self.n_train_years)
                test_yr = range(start + self.n_train_years + self.train_test_offset,
                                start + self.n_train_years + self.train_test_offset + self.n_test_years)
                if min(test_yr) >= len(avail_year):
                    cont = False
                else:
                    start += self.rolling_wind_step
                    split_batch.append([avail_year[train_yr], avail_year[test_yr]])
        
        return split_batch

        
    def split(self, X, y=None, groups=None):
        
        split_batch = self.determine_n_split(X)
        for i in range(len(split_batch)):
            train_idx = np.where(X[self.year_variable].isin(split_batch[i][0]))[0].astype(int)
            test_idx = np.where(X[self.year_variable].isin(split_batch[i][1]))[0].astype(int)
            if self.show_info:
                print('Split', i+1, 'Train:', split_batch[i][0], '('+str(len(train_idx))+' obs)',
                      '   Test:', split_batch[i][1], '('+str(len(test_idx))+' obs)')
            yield train_idx, test_idx

    def get_n_splits(self, X, y=None, groups=None):
        return len(self.determine_n_split(X))
    
   
def to_labels(pos_probs, threshold):
    # apply threshold to positive probabilities to create labels
    return (pos_probs >= threshold).astype('int')
            
            
def eval_score(measure, true_lab, pred_lab=None, pred_proba=None, threshold=None):
            
            '''
            Evaluates all metrics with safe mode for zero _division.
            If "threshold" is provided, pred_proba are converted to labels.
            '''
      
            if threshold is not None:
                if pred_proba is None:
                    raise ValueError('If "threshold" is provided, "pred_proba" is expected.')
                pred_lab = to_labels(pred_proba, threshold)
            
            if threshold is None and pred_lab is None and pred_proba is None:
                raise ValueError('If "threshold" is not provided, "pred_lab" or "pred_proba" is expected.')
            
            if measure.__name__ in ['f1_score', 'precision_score', 'recall_score']:
                
                out = measure(true_lab, pred_lab, zero_division=0)
                
            if measure.__name__ in ['roc_auc_score']:
        
                out = measure(true_lab, pred_proba)
            
            if measure.__name__ in ['accuracy_score']:
        
                out = measure(true_lab, pred_lab)
            
            return out
        

def evaluate_score_df(measure, true_lab_train, true_lab_valid, true_lab_test,
                      pred_lab_train=None, pred_lab_valid=None, pred_lab_test=None,
                      pred_proba_train=None, pred_proba_valid=None, pred_proba_test=None,
                      threshold=None, split_i=None):

    '''
    Creates DataFrame row with optimal and 0.5 threshold, if provided, on train, validation and test.
    '''

    # without threshold and only probabilities
    if measure.__name__ in ['roc_auc_score']:
        add_row = pd.DataFrame({'split': split_i,
                                'train_best': eval_score(measure, true_lab=true_lab_train, pred_proba=pred_proba_train),
                                'valid_best': eval_score(measure, true_lab=true_lab_valid, pred_proba=pred_proba_valid),
                                'test_best': eval_score(measure, true_lab=true_lab_test, pred_proba=pred_proba_test)}, index=[split_i])

    # with threshold
    if measure.__name__ in ['f1_score', 'accuracy_score', 'precision_score', 'recall_score']:
        add_row = pd.DataFrame({'split': split_i,
                                'train_best': eval_score(measure, true_lab=true_lab_train, pred_proba=pred_proba_train, threshold=threshold),
                                'valid_best': eval_score(measure, true_lab=true_lab_valid, pred_proba=pred_proba_valid, threshold=threshold),
                                'test_best': eval_score(measure, true_lab=true_lab_test, pred_proba=pred_proba_test, threshold=threshold),
                                'best_thresh': threshold,
                                'train_05': eval_score(measure, true_lab=true_lab_train, pred_proba=pred_proba_train, threshold=0.5),
                                'valid_05': eval_score(measure, true_lab=true_lab_valid, pred_proba=pred_proba_valid, threshold=0.5),
                                'test_05': eval_score(measure, true_lab=true_lab_test, pred_proba=pred_proba_test, threshold=0.5)}, index=[split_i])                    
                                    
    return add_row 


def fit_predict_cv_classifier(df, model=None, measure=None, cv_iterator=None, out_of_sample_years=1,
                              time_var='fyear', add_measure=[], return_model=False, show_split=False):

    '''
    Evaluates performance for each fold for a binary classifier. Cross validation requires "time_var" variable
    to split according to years. Threshold optimization is performed on the validation set using "measure".
    
    Args:
        - df: dataframe with target 'y', "time_var" for years and other input variables
        - model: classification model
        - measure: performance measure (function). Used to evaluate optimal threshold.
        - cv_iterator: iterator for cross-validation
        - out_of_sample_years: number of years to evaluate out-of-sample performance
        - time_var: string for year variable
        - add_measure: list of additional performance measures (function) to be evaluated withptimal threshold.
        - return_model: if True returns dictionary of fitted models
        - show_split: if True only prints splits and skip everything else
        
    Return:
        - df_perf: dataframe with ['split', 'train_best', 'valid_best', 'test_baset', 'best_thresh',
                    'train_05', 'valid_05', 'test_05'], where _best refers to performance with threshold optimization, _05 without
        - fitted_models: dictionary of models fitted on every split
        - df_perf_add: dictionary of dataframe of same shape of df_perf evaluated on each measure in add_measure
    '''
    
    # evaluate out-of-sample
    avail_year = np.sort(df[time_var].unique())
    year_to_remove = avail_year[-out_of_sample_years:]    
    
    df_perf = pd.DataFrame()
    df_perf_add = {m.__name__: pd.DataFrame() for m in add_measure}
    cv_iterator.show_info=False
    fitted_models = {}
    df_fit = df.copy()[~df[time_var].isin(year_to_remove)]  # skip year_to_remove, keep in mind .iloc is used below
    thresholds = np.arange(0, 1, 0.001)
    scores_valid = pd.DataFrame(index=thresholds)
    pred_list={}
    for split_i, (train_idx, valid_idx) in enumerate(cv_iterator.split(df_fit)):

        # create train, validation and test (out-of-sample)
        X_train = df_fit.drop(columns=['y', time_var]).iloc[train_idx]
        y_train = df_fit['y'].iloc[train_idx]
        X_valid = df_fit.drop(columns=['y', time_var]).iloc[valid_idx]
        y_valid = df_fit['y'].iloc[valid_idx]
        year_test_start = np.where(avail_year == df_fit[time_var].iloc[valid_idx].max())[0][0] + 1
        test_idx = np.where(df[time_var].isin(avail_year[year_test_start:(year_test_start+out_of_sample_years)]))[0]
        X_test = df.drop(columns=['y', time_var]).iloc[test_idx]    # test_idx is extracted from df, wih .iloc
        y_test = df['y'].iloc[test_idx]

        if show_split:
            print('Split', split_i+1, 'Train:', np.sort(df_fit[time_var].iloc[train_idx].unique()), '('+str(len(train_idx))+' obs)',
                 '   Validation:', np.sort(df_fit[time_var].iloc[valid_idx].unique()), '('+str(len(valid_idx))+' obs)',
                 '   Test:', np.sort(df[time_var].iloc[test_idx].unique()), '('+str(len(test_idx))+' obs)')
            continue

        # fit model
        fit_model = model.fit(X_train,y=y_train)
        if return_model:
            fitted_models['split_'+str(split_i)] = fit_model
        y_pred_train = model.predict_proba(X_train)[:, 1]
        y_pred_valid = model.predict_proba(X_valid)[:, 1]
        y_pred_test = model.predict_proba(X_test)[:, 1]

        pred_list['split_'+str(split_i)] = {'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test,
                                            'y_pred_train': y_pred_train, 'y_pred_valid': y_pred_valid,
                                            'y_pred_test': y_pred_test}
        # evaluate threshold scores
        scores_valid = scores_valid.merge(pd.DataFrame({'split_'+str(split_i):
                                                        [measure(y_valid, to_labels(y_pred_valid, t)) for t in thresholds]},
                                                       index=thresholds), left_index=True, right_index=True, how='left')

    # evaluate best threshold over all splits
    best_thresh = thresholds[np.argmax(scores_valid.mean(axis=1))]

    # save results and additional measures
    for split_i in range(len(pred_list)):

        y_train = pred_list['split_'+str(split_i)]['y_train']
        y_valid = pred_list['split_'+str(split_i)]['y_valid']
        y_test = pred_list['split_'+str(split_i)]['y_test']
        y_pred_train = pred_list['split_'+str(split_i)]['y_pred_train']
        y_pred_valid = pred_list['split_'+str(split_i)]['y_pred_valid']
        y_pred_test = pred_list['split_'+str(split_i)]['y_pred_test']

        df_perf = pd.concat([df_perf, evaluate_score_df(measure, true_lab_train=y_train, true_lab_valid=y_valid,
                                                   true_lab_test=y_test, pred_proba_train=y_pred_train,
                                                   pred_proba_valid=y_pred_valid, pred_proba_test=y_pred_test,
                                                   threshold=best_thresh, split_i=split_i)])

        if len(add_measure) > 0:
            for m in add_measure:

                df_perf_add[m.__name__] = pd.concat([df_perf_add[m.__name__],
                                                evaluate_score_df(m, true_lab_train=y_train, true_lab_valid=y_valid,
                                                                  true_lab_test=y_test, pred_proba_train=y_pred_train,
                                                                  pred_proba_valid=y_pred_valid, pred_proba_test=y_pred_test,
                                                                  threshold=best_thresh, split_i=split_i)])
    
    return df_perf, fitted_models, pred_list, df_perf_add


def make_model_name(tune_params, round_float=False, round_float_digits=5):
    
    if round_float:
        for k, v in tune_params.items():
            if type(v) == float:
                tune_params[k] = np.round(tune_params[k], round_float_digits)
    
    return '-'.join([k+str(tune_params[k]) for k in sorted(tune_params.keys())])


class Objective:
    def __init__(self, df, measure, cv_iterator, model_type, out_of_sample_years, time_var='fyear',
                 model_save_path=None, add_measure=[], optim_measure='valid_best'):
        # Hold this implementation specific arguments as the fields of the class.
        self.df = df
        self.measure = measure
        self.cv_iterator = cv_iterator
        self.model_type = model_type
        self.out_of_sample_years = out_of_sample_years
        self.time_var = time_var
        self.model_save_path = model_save_path
        self.add_measure = add_measure
        self.optim_measure = optim_measure

    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        
        # print and update iter count
        try:
            iters=joblib.load('iter.pkl')
            iters['iter'] += 1
            iters['avg_time'] = (iters['avg_time'] + (timer()-iters['time'])) / 2
            iters['time'] = timer()
            print('Trial', iters['iter'], '/', iters['tot_iter'], '    avg elapsed time: ',
                  str(datetime.timedelta(seconds=round(iters['avg_time']))),
                  '  current optimal value:', iters['best_val'], ' '*30, end='\r')
            joblib.dump(iters, 'iter.pkl')
        except:
            pass
        
        # settings
        if self.model_type == 'RandomForest':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier
            
            # define parameters
            tune_params = {'n_estimators': trial.suggest_int('n_estimators', 10, 2000, log=True),
                           'min_samples_split': trial.suggest_int('min_samples_split', 2, 300, log=True),
                           'max_features': trial.suggest_int('max_features', 10, self.df.shape[1] - 3)}
        
            # define model
            model = RandomForestClassifier(bootstrap = False, class_weight = 'balanced',
                                           random_state = 666, n_jobs = -1,
                                           **tune_params)
            
        if self.model_type == 'GradientBoost':
            # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier
            
            # define parameters
            tune_params = {'n_estimators': trial.suggest_int('n_estimators', 10, 2000, log=True),
                           'min_samples_split': trial.suggest_int('min_samples_split', 2, 300, log=True),
                           'max_features': trial.suggest_int('max_features', 10, self.df.shape[1] - 3),
                           'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1, log=True),
                           'subsample': trial.suggest_float('subsample', 0.01, 1)}
        
            # define model
            model = GradientBoostingClassifier(loss = 'deviance', criterion = 'friedman_mse', random_state = 666,
                                               **tune_params)
            
        if self.model_type == 'LightGBM':
            # https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html
            # https://lightgbm.readthedocs.io/en/latest/Parameters.html
            
            # define parameters
            tune_params = {'n_estimators': trial.suggest_int('n_estimators', 10, 2000, log=True),
                           'max_depth': trial.suggest_int('max_depth', 2, 100, log=True),
                           'min_child_samples': trial.suggest_int('min_child_samples', 2, 300, log=True),
                           'reg_alpha': trial.suggest_categorical('reg_alpha',
                                    [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]),
                           'reg_lambda': trial.suggest_categorical('reg_lambda',
                                    [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]),
                           'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 1),
                           'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1, log=True),
                           'subsample': trial.suggest_float('subsample', 0.01, 1)}
            
            # define model
            model = LGBMClassifier(boosting_type='gbdt', num_leaves=np.min([2**tune_params['max_depth'], 100]), objective='binary',
                                   class_weight = 'balanced', random_state = 666, n_jobs=-1,
                                   device_type='cpu', deterministic=True, tree_learner='serial',                                   
                                   **tune_params)
            
        if self.model_type == 'ElasticNet':
            # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
            
            # define parameters
            tune_params = {'alpha': trial.suggest_categorical('alpha',
                                    [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0, 1.0, 10.0, 100.0]),
                           'l1_ratio': trial.suggest_float('l1_ratio', 0, 1),
                           'eta0': trial.suggest_float('eta0', 1e-6, 1, log=True)}
        
            # define model
            model = SGDClassifier(loss = 'log_loss', penalty = 'elasticnet', shuffle = True,
                                  random_state = 666, learning_rate = 'adaptive',
                                  early_stopping = True, class_weight = 'balanced', n_jobs = -1,
                                  **tune_params)
         
        # evaluate or reload performance
        mod_name = make_model_name(tune_params, round_float=(True if self.model_type == 'LightGBM' else False))
        pkl_path = os.path.join(self.model_save_path, mod_name + '.pkl')
        if os.path.exists(pkl_path):
            pkl_reload=joblib.load(pkl_path)
            perf = pkl_reload['perf']
            fitted_model = pkl_reload['fitted_model']
            pred_list = pkl_reload['pred_list']
            perf_add = pkl_reload['perf_add']
        else:
            perf, fitted_model, pred_list, perf_add = fit_predict_cv_classifier(df=self.df, model=model, measure=self.measure,
                                                  cv_iterator=self.cv_iterator, out_of_sample_years=self.out_of_sample_years,
                                                  time_var=self.time_var, add_measure=self.add_measure,
                                                  return_model=(True if self.model_save_path is not None else False))
        
            # save fitted models
            if self.model_save_path is not None:
                joblib.dump({'measure': self.measure,
                             'perf': perf,
                             'fitted_model': fitted_model,
                             'pred_list': pred_list,
                             'perf_add': perf_add}, pkl_path, compress=('lzma', 3))

        # evaluate optimization value
        avg_perf = perf[self.optim_measure].mean()
                
        return avg_perf
    

def update_current_optimal_val(study, frozen_trial):
    iters=joblib.load('iter.pkl')
    iters['best_val'] = study.best_value
    joblib.dump(iters, 'iter.pkl')
                
                
def tune_hyperparameters(df, tot_trials=100, model_type='', measure=None, cv_iterator=None, time_var='',
                         out_of_sample_years=1, add_measure=[], optim_measure='valid_best',
                         file_name='', tuning_folder='', tuning_checkpoint_folder='', reload=False):
    
    '''
    Tune hyperparameters with Optuna
    
    Args:
        - df: dataframe with target 'y', "time_var" for years and other input variables
        - tot_trials: number of trials for optimization
        - model_type: which model to tune. 'RandomForest', 'GradientBoost', 'LightGBM', 'ElasticNet'
        - measure: performance measure (function). Used to evaluate optimal threshold.
        - cv_iterator: iterator for cross-validation
        - out_of_sample_years: number of years to evaluate out-of-sample performance
        - time_var: string for year variable
        - add_measure: list of additional performance measures (function) to be evaluated with optimal threshold
        - optim_measure: optimization measure, one of columns of "df_pred" in fit_predict_cv_classifier()
        - reload: if True, reload previous results
    '''

    if model_type not in ['RandomForest', 'GradientBoost', 'ElasticNet', 'LightGBM']:
        raise ValueError('\''+model_type+'\' not supported. See docs for implemented learners.')
    
    # show splits
    _, _, _, _ = fit_predict_cv_classifier(df=df, cv_iterator=cv_iterator, out_of_sample_years=out_of_sample_years,
                                        time_var=time_var, show_split=True)
    print('\n')
    
    study_name = '_'.join([file_name, model_type, measure.__name__])
    tuning_checkpoint = os.path.join(tuning_checkpoint_folder, study_name)
    
    if not reload:
    
        # optimization settings and folder
        if os.path.exists(os.path.join(tuning_folder, study_name + '.db')):
            print('###### Reloading study:', os.path.join(tuning_folder, study_name + '.db'), '\n')
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        storage = optuna.storages.RDBStorage(url='sqlite:///' + os.path.join(tuning_folder, study_name + '.db'),
                                             heartbeat_interval=60,
                                             grace_period=600)
        
        if not os.path.exists(tuning_checkpoint):
            os.makedirs(tuning_checkpoint)
        joblib.dump({'iter': 0, 'tot_iter': tot_trials, 'time': timer(), 'avg_time': 0, 'best_val': ''}, 'iter.pkl') # create iter count

        # create study and optimize
        np.random.seed(66)
        obj = Objective(df=df, measure=measure, cv_iterator=cv_iterator, model_type=model_type,
                        out_of_sample_years=out_of_sample_years, time_var=time_var,
                        model_save_path=tuning_checkpoint, add_measure=add_measure)

        np.random.seed(66)
        study = optuna.study.create_study(storage=storage,
                                          sampler=optuna.samplers.TPESampler(seed=666),
                                          study_name=study_name,
                                          direction='maximize',
                                          load_if_exists=True)

        start=timer()
        print('- Started at:', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"), '\n')
        study.optimize(obj, n_trials=tot_trials, n_jobs=1, gc_after_trial=True, callbacks=[update_current_optimal_val])
        eval_time=datetime.timedelta(seconds=round(timer()-start))
        print('\nTotal elapsed time:', str(eval_time))
        
        # save study
        joblib.dump({'study': study, 'eval_time': eval_time, 'study_name': study_name, 'measure': measure,
                     'cv_iterator': cv_iterator, 'out_of_sample_years': out_of_sample_years, 'add_measure': add_measure,
                     'optim_measure': optim_measure},os.path.join(tuning_folder, study_name + '.pkl'), compress=('lzma', 3))
        print('\n\n- Pickle saved to', os.path.join(tuning_folder, study_name + '.pkl'))
        os.remove('iter.pkl')

    else:
        print('\n- Session reloaded')
        pkl_reload=joblib.load(os.path.join(tuning_folder, study_name + '.pkl'))
        study=pkl_reload['study']
        eval_time=pkl_reload['eval_time']
        print('\nTotal elapsed time:', str(eval_time))
    
    # print results
    print('\n\nOptimization metric:', measure.__name__, 'on', optim_measure)
    print('Optimal score:', study.best_value, '  (Trial', str(study.best_trial.number)+')')
    print('Best params:\n', study.best_params)

    # save logs and optimization figures
    study_log = study.trials_dataframe(multi_index=False)
    study_log.columns = study_log.columns.str.replace('params_', '')
    study_log['pkl'] = study_log.apply(lambda x: os.path.join(tuning_checkpoint,
                                        make_model_name(x[x.index.isin(study.best_params.keys())].to_dict(),
                                                       round_float=(True if model_type=='LightGBM' else False)) + '.pkl'), axis=1)
    study_log['best_par'] = np.where(study_log['number'] == study.best_trial.number, 'x', '')
    plot_optimization_history(study).write_image(os.path.join(tuning_folder, study_name + '_optim_hist.png'), scale = 2)
    plot_contour(study).write_image(os.path.join(tuning_folder, study_name + '_contour.png'), scale = 2)
    plot_param_importances(study).write_image(os.path.join(tuning_folder, study_name + '_params_importance.png'), scale = 2)

    # append all performance to study_log
    df_add = pd.DataFrame()
    for pkl_path in study_log.pkl.unique():
        row_add = pd.DataFrame()
        try:
            pkl_reload=joblib.load(pkl_path)
            perf = pkl_reload['perf']
            perf_add = pkl_reload['perf_add']

            row_add = pd.concat([row_add,
                                 (perf.groupby('best_thresh').agg('mean').reset_index().drop(columns='split')
                                  .add_prefix(measure.__name__.replace('_score', '').upper()+'.')
                                 )], axis=1) 
            for k, v in perf_add.items():
                v['gby']=0
                row_add = pd.concat([row_add,
                                     (v.groupby('gby').agg('mean').reset_index().drop(columns=['gby', 'split'])
                                      .add_prefix(k.replace('_score', '').upper()+'.')
                                     )], axis=1) 
            row_add.insert(0, 'pkl', pkl_path)
            df_add = pd.concat([df_add, row_add])
        except:
            pass
    
    study_log = study_log.merge(df_add, on='pkl', how='left')
    
    # print best parameters performance
    avail_measure = study_log.columns[study_log.columns.str.endswith(optim_measure)].str.replace('.'+optim_measure, '', regex=True).tolist()
    avail_set = study_log.columns[study_log.columns.str.startswith(avail_measure[0])].str.replace(avail_measure[0]+'.', '', regex=True).tolist()
    best_row = study_log[study_log['number'] == study.best_trial.number]

    df_best = pd.DataFrame()
    for meas in avail_measure:
        common_cols = list(set([meas+'.'+x for x in avail_set]) & set(best_row.columns))
        add_df = best_row.loc[:, common_cols]
        add_df.columns = add_df.columns.str.replace(meas+'.', '', regex=True)
        add_df.insert(0, 'Performance', meas)
        df_best = pd.concat([df_best, add_df])
    df_best=df_best.fillna('')#.fillna(method='bfill').fillna(method='ffill')
    # df_best=df_best[[df_best.columns[df_best.columns.str.contains(x)][0] for x in ['Performance', 'thresh', 'train', 'valid', 'test']]]
    df_best=df_best[['Performance', 'best_thresh', 'train_best', 'valid_best', 'test_best', 'train_05', 'valid_05', 'test_05']]
    print('Best params performance:')
    display(df_best)
    
    study_log.to_csv(os.path.join(tuning_folder, study_name + '.csv'), index=False, sep=';')
    print('\n- Tuning log saved to', os.path.join(tuning_folder, study_name + '.csv'))
    
    
    return study, study_log


def cv_performance_summary(**kwargs):

    '''
    - **kwargs: any dictionary that contains "measure", "perf" and "perf_add" returned by fit_predict_cv_classifier()
    '''
    
    measure = kwargs.get('measure')
    perf = kwargs.get('perf')
    perf_add = kwargs.get('perf_add')
    
    all_measure={measure.__name__: perf}
    all_measure.update(perf_add)
    col_order=['best_thresh', 'train_best', 'valid_best', 'test_best', 'train_05', 'valid_05', 'test_05']
    
    df_out = pd.DataFrame()
    for meas, tab in all_measure.items():
        common_cols = [x for x in tab.columns if x in col_order]
        if 'best_thresh' not in common_cols:
            tab['best_thresh'] = 0
        add_row=tab.groupby('best_thresh').agg('mean').reset_index().drop(columns='split')
        if 'best_thresh' not in common_cols:
            add_row['best_thresh'] = None
        add_row.insert(0, 'Performance', meas.replace('_score', '').upper())
        df_out=pd.concat([df_out, add_row])

    df_out['best_thresh']=df_out['best_thresh'].fillna(method='bfill').fillna(method='ffill')    
    df_out=df_out.fillna('')
    df_out=df_out[['Performance'] + col_order]

    return df_out


class GroupVariables:
    def __init__(self, df, time_var='fyear', n_jobs=-1, file_name='', var_description='',
                 stats_folder='', checkpoint_folder='', tuning_folder='', groupvar_folder='', featimp_folder=''):
        
        '''
        Evaluate average distance correlation over years and group variables.

        Args:
            - df: dataframe with [time_var, 'y', ....]
            - time_var: string for year variable
            - n_jobs: number of processes. -1 all available cores, 0 serial
        '''
        
        self.df=df
        self.time_var=time_var
        self.n_jobs=n_jobs
        self.file_name=file_name
        self.var_description=var_description
        self.stats_folder=stats_folder
        self.checkpoint_folder=checkpoint_folder
        self.tuning_folder=tuning_folder
        self.groupvar_folder=groupvar_folder
        self.featimp_folder=featimp_folder

    def evaluate_distance(self, reload=False):
        
        pkl_path = os.path.join(self.groupvar_folder, self.file_name+'_dist_mat.pkl')
        
        if reload:
            pkl_reload=joblib.load(pkl_path)
            self.avg_dist=pkl_reload['avg_dist']
            self.var_names=pkl_reload['var_names']
            print('- Distance matrix reloaded from', pkl_path)
        else:
            years = self.df[self.time_var].unique()
            start = timer()
            print('- Evaluating distance matrix', end='')
            
            # multiprocess
            if self.n_jobs !=0:
                print(' with Multiprocess...')
                n_processes = None if self.n_jobs == -1 else self.n_jobs
                p=Pool(n_processes)
                results=p.imap_unordered(partial(GroupVariables._distance_mp, df=self.df, time_var=self.time_var), years)                
            # serial
            else:
                print(' without Multiprocess...')
                mat_list=[]
                msg_list=[]
                for year in years:
                    dist, out_message = GroupVariables._distance_mp(year, df=self.df, time_var=self.time_var)
                    mat_list.append(dist)
                    msg_list.append(out_message)
                results=zip(mat_list, msg_list)
                del mat_list, msg_list
            
            # average over years
            matrix_list=[]    
            for dist, out_message in results:
                if out_message:
                    print('\n'.join(out_message))
                matrix_list.append(squareform(dist))
            avg_dist=np.stack(matrix_list, axis=0).mean(axis=0)
            del matrix_list
            print('Elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))
            
            var_names = self.df.drop(columns=[self.time_var, 'y']).columns.tolist()
            
            # save pkl
            joblib.dump({'avg_dist': avg_dist, 'var_names': var_names}, pkl_path, compress=('lzma', 3))
            print('\n- Distance matrix pickle saved to', pkl_path)

            self.avg_dist=avg_dist
            self.var_names=var_names            
    
    def plot_dendro(self, fig_size=(8, 12)):
        
        self.dist_linkage=GroupVariables._plot_dendro(dist_mat=self.avg_dist, var_names=self.var_names, fig_size=fig_size)
    
    def test_variables_groups(self, model_type='LightGBM', find_cluster_range=[2,3], reload=False):
    
        print('\n- Testing different number of groups for all variables - Started at:', datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        # reload best model and configuration
        pkl_path = [x for x in os.listdir(self.tuning_folder) if x.startswith(f'{self.file_name}_{model_type}') and x.endswith('.pkl')][0]
        study_path = [x for x in os.listdir(self.tuning_folder) if x.startswith(f'{self.file_name}_{model_type}') and x.endswith('.csv')][0]
        pkl_reload = joblib.load(os.path.join(self.tuning_folder, pkl_path))
        study_name=pkl_reload['study_name']
        measure=pkl_reload['measure']
        cv_iterator=pkl_reload['cv_iterator']
        out_of_sample_years=pkl_reload['out_of_sample_years']
        add_measure=pkl_reload['add_measure']
        optim_measure=pkl_reload['optim_measure']
        study_log=pd.read_csv(os.path.join(self.tuning_folder, study_path), sep=';')
        study_log=study_log[study_log['best_par'] == 'x']
        best_mod_reload=joblib.load(study_log['pkl'].values[0])
        best_model=best_mod_reload['fitted_model']['split_0']

        # evaluate model performance for each number of clusters (variables)
        df_perf_cluster=pd.DataFrame()
        best_variables_log={}
        start=timer()
        for i, num_clust in enumerate(range(find_cluster_range[0], find_cluster_range[1]+1)):

            print(f'   Grouping in {num_clust} clusters ({i+1}/{find_cluster_range[1] - find_cluster_range[0] + 1})',
                  '   Last update at:', datetime.datetime.now().strftime("%H:%M:%S"), end='\r')

            # evaluate clusters
            cluster_ids = hierarchy.fcluster(self.dist_linkage, num_clust, criterion="maxclust")
            cluster_dict={}
            for cl in np.unique(cluster_ids):
                cluster_dict[str(cl)]=[self.var_names[i] for i in np.where(cluster_ids == cl)[0]]

            # select best variable in each cluster according to mutual information gain with target variable
            # https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html
            mi_path=os.path.join(self.groupvar_folder,
                                    self.file_name+'_'+model_type+'_'+measure.__name__+'_'+str(num_clust)+'_mutualInfo.pkl')
            if not reload or not os.path.exists(mi_path):
                best_variables=[]
                mi_log={}
                for cl, var in cluster_dict.items():
                    df_t = self.df.copy()[[self.time_var, 'y'] + var]
                    mi = mutual_info_classif(X=df_t.drop(columns=[self.time_var, 'y']), y=df_t['y'],
                                             discrete_features='auto', n_neighbors=3, copy=False, random_state=666)
                    best_variables.append(var[np.argmax(mi)])
                    mi_log[str(cl)] = dict(zip(var, mi))
                    del df_t
                save_dict={'best_variables': best_variables, 'cluster_dict': cluster_dict, 'mutual_information': mi_log}
                joblib.dump(save_dict, mi_path)
            else:
                save_dict=joblib.load(mi_path)
                best_variables=save_dict['best_variables']
            best_variables_log[str(num_clust)+' cluster']=save_dict

            # evaluate model performance on dataset with best variables only
            df_featimp = self.df.copy()[[self.time_var, 'y'] + best_variables]    

            if model_type == 'RandomForest':   # adapt max_features to new set of variables
                best_model.set_params(**{'max_features': int(best_model.max_features / self.df.shape[1] * df_featimp.shape[1])})

            clust_path=os.path.join(self.groupvar_folder,
                                    self.file_name+'_'+model_type+'_'+measure.__name__+'_'+str(num_clust)+'.pkl')
            if not reload or not os.path.exists(clust_path):
                perf, fitted_model, pred_list, perf_add = fit_predict_cv_classifier(df=df_featimp, model=best_model, measure=measure,
                                                              cv_iterator=cv_iterator, out_of_sample_years=out_of_sample_years,
                                                              time_var=self.time_var, add_measure=add_measure, return_model=False)
                perf_t=cv_performance_summary(measure=measure, perf=perf, perf_add=perf_add)
                perf_t=perf_t[perf_t['Performance']==measure.__name__.replace('_score', '').upper()][['best_thresh', optim_measure]]
                perf_t.insert(0, 'Model', str(num_clust)+' cluster')
                joblib.dump(perf_t, clust_path)
            else:
                perf_t=joblib.load(clust_path)

            df_perf_cluster=pd.concat([df_perf_cluster, perf_t])

        print('\n   Elapsed time:', str(datetime.timedelta(seconds=round(timer()-start))))

        # compare performance on "optim_measure"
        perf_best=cv_performance_summary(**best_mod_reload)
        perf_best=perf_best[perf_best['Performance']==measure.__name__.replace('_score', '').upper()][['best_thresh', optim_measure]]
        perf_best.insert(0, 'Model', 'All variables')
        df_perf_cluster=pd.concat([perf_best, df_perf_cluster])
        df_perf_cluster.insert(1, 'Performance', measure.__name__.replace('_score', '').upper())
        df_perf_cluster['diff']=df_perf_cluster[optim_measure]-df_perf_cluster[df_perf_cluster['Model']=='All variables'][optim_measure]
        df_perf_cluster.loc[df_perf_cluster['Model']=='All variables', 'diff']=9999
        df_perf_cluster.sort_values(by='diff', ascending=False, inplace=True)
        df_perf_cluster.loc[df_perf_cluster['Model']=='All variables', 'diff']=''

        # save results
        save_path=os.path.join(self.groupvar_folder, self.file_name+'_dist_mat_cluster_variables.pkl')
        joblib.dump({'df_perf_cluster': df_perf_cluster, 'cluster_dict': cluster_dict}, save_path)
        print('\n- Cluster pickle saved to', save_path)

        display(df_perf_cluster)

        self.df_perf_cluster=df_perf_cluster
        self.best_variables_log=best_variables_log
        self.model_type=model_type
        self.optim_measure=optim_measure
    
    def select_groups(self, num_cluster=5, save_prefix='01c_'):

        # recall performance
        perf=self.df_perf_cluster['Performance'].values[0]
        perf_pre=np.round(self.df_perf_cluster[self.df_perf_cluster['Model'] == 'All variables'][self.optim_measure].values[0], 3)
        perf_post=np.round(self.df_perf_cluster[self.df_perf_cluster['Model'] == str(num_cluster)+' cluster'][self.optim_measure].values[0], 3)

        best_variables=self.best_variables_log[str(num_cluster)+' cluster']['best_variables']
        cluster_dict=self.best_variables_log[str(num_cluster)+' cluster']['cluster_dict']
        df_group=pd.DataFrame()
        for cl, var in cluster_dict.items():
            df_t=pd.DataFrame({'Variable': var, 'Group': cl})
            mi_dict=self.best_variables_log[str(num_cluster)+' cluster']['mutual_information'][cl]
            df_t=df_t.merge(pd.DataFrame(mi_dict.items(), columns=['Variable', 'Mutual Information']), on='Variable', how='left')
            df_t['Selected']=np.where(df_t['Variable'].isin(best_variables), 'x', '')
            df_t.sort_values(by='Mutual Information', ascending=False, inplace=True)
            df_group=pd.concat([df_group, df_t])
        
        # add variables description
        var_descr=pd.read_csv(os.path.join(self.stats_folder, self.var_description), sep=';')[['VARIABLE', 'Description']]
        df_group=df_group.merge(var_descr, how='left', left_on='Variable', right_on='VARIABLE').drop(columns='VARIABLE')

        # print info
        print(f'- {num_cluster} groups of variables selected:\n')
        for index, row in df_group[df_group.Selected == 'x'].iterrows():
            print(f"   '{row['Variable']}': {row['Description']}")
        print(f'\n- {perf} changed from {perf_pre} (all variables) to {perf_post}')
        
        # save csv with groups
        save_path=os.path.join(self.stats_folder, f'{save_prefix}{self.file_name}_variables_groups.csv')
        df_group.to_csv(save_path, index=False, sep=';')
        print('\n- List of groups of variables saved to', save_path)

        self.best_variables=best_variables

        return best_variables
    
    def train_model(self, reload=False, select_variables=[]):
        
        # reload best model and configuration
        pkl_path = [x for x in os.listdir(self.tuning_folder) if x.startswith(f'{self.file_name}_{self.model_type}') and x.endswith('.pkl')][0]
        study_path = [x for x in os.listdir(self.tuning_folder) if x.startswith(f'{self.file_name}_{self.model_type}') and x.endswith('.csv')][0]
        pkl_reload = joblib.load(os.path.join(self.tuning_folder, pkl_path))
        study_name=pkl_reload['study_name']
        measure=pkl_reload['measure']
        cv_iterator=pkl_reload['cv_iterator']
        out_of_sample_years=pkl_reload['out_of_sample_years']
        add_measure=pkl_reload['add_measure']
        optim_measure=pkl_reload['optim_measure']
        study_log=pd.read_csv(os.path.join(self.tuning_folder, study_path), sep=';')
        study_log=study_log[study_log['best_par'] == 'x']
        best_mod_reload=joblib.load(study_log['pkl'].values[0])
        best_model=best_mod_reload['fitted_model']['split_0']

        # select best variables
        if not select_variables:
            best_variables=self.best_variables
        else:
            best_variables=select_variables
        
        # evaluate model performance on dataset with best variables only
        df_featimp = self.df.copy()[[self.time_var, 'y'] + best_variables]    

        if self.model_type == 'RandomForest':   # adapt max_features to new set of variables
            best_model.set_params(**{'max_features': int(best_model.max_features / self.df.shape[1] * df_featimp.shape[1])})

        model_path=os.path.join(self.featimp_folder,
                                self.file_name+'_'+self.model_type+'_model_for_explainer.pkl')
        if not reload or not os.path.exists(model_path):
            perf, fitted_model, pred_list, perf_add = fit_predict_cv_classifier(df=df_featimp, model=best_model, measure=measure,
                                                          cv_iterator=cv_iterator, out_of_sample_years=out_of_sample_years,
                                                          time_var=self.time_var, add_measure=add_measure, return_model=True)
            perf_summ=cv_performance_summary(measure=measure, perf=perf, perf_add=perf_add)
            perf_summ=perf_summ[perf_summ['Performance']==measure.__name__.replace('_score', '').upper()][['best_thresh', optim_measure]]
            joblib.dump({'fitted_model': fitted_model, 'perf_summ': perf_summ, 'out_of_sample_years': out_of_sample_years,
                         'time_var': self.time_var, 'cv_iterator': cv_iterator, 'best_variables': best_variables}, model_path)
        else:
            load_t=joblib.load(model_path)
            fitted_model=load_t['fitted_model']
            perf_summ=load_t['perf_summ']

        perf=self.df_perf_cluster['Performance'].values[0]
        perf_post=np.round(perf_summ[self.optim_measure].values[0], 3)
        print(f'- Fitted model {perf}: {perf_post}')
        print('\n- Fitted model pickle saved to', model_path)
        
    @staticmethod
    def _distance_mp(year, df, time_var='fyear'):
    
        '''
        Evaluates distance correlation for multiprocessing. Returns dist in vector format, to be passed to squareform()
        https://dcor.readthedocs.io/en/latest/theory.html
        https://en.wikipedia.org/wiki/Distance_correlation
        '''    
    
        from scipy.spatial.distance import pdist
        from dcor import u_distance_correlation_sqr
        from scipy.spatial.distance import squareform
        import numpy as np

        out_message=[]

        # evaluate unbiased SQUARED distance correlation
        dist=pdist(df[df[time_var] == year].drop(columns=[time_var, 'y']).values.T, lambda x,y: u_distance_correlation_sqr(x, y))
        # replace distance with constant variable with 1 (max distance)
        dist=np.nan_to_num(dist, nan=1)
        # replace negative distance close to 0 with 0
        if dist.min() < 0:
            out_message.append(f'Negative value found in {year}: {dist.min()} - Replacing with 0')
            dist[dist < 0] = 0
        # square root
        dist=np.sqrt(dist)
        # check nan
        if np.isnan(dist).sum() > 0:
            out_message.append(f'NaN values still present in {year}')
        # check range
        if dist.min() < 0 or dist.max() > 1:
            out_message.append(f'Error in range for {year}: min={dist.min()}  max={dist.max()}')

        return dist, out_message
    
    @staticmethod
    def _plot_dendro(dist_mat, var_names, fig_size=(8, 12)):
    
        from scipy.cluster.hierarchy import ClusterWarning
        from warnings import simplefilter

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size)

        simplefilter("ignore", ClusterWarning)
        dist_linkage = hierarchy.ward(dist_mat)
        dendro = hierarchy.dendrogram(
            dist_linkage, labels=var_names, ax=ax1, leaf_rotation=90
        )
        dendro_idx = np.arange(0, len(dendro["ivl"]))

        cmap=plt.cm.viridis
        im2=ax2.imshow(dist_mat[dendro["leaves"], :][:, dendro["leaves"]], cmap=cmap)
        ax2.set_xticks(dendro_idx)
        ax2.set_yticks(dendro_idx)
        ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
        ax2.set_yticklabels(dendro["ivl"])
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.08)
        fig.colorbar(im2, cax=cax, orientation='vertical')
        fig.tight_layout()
        plt.show()

        return dist_linkage
    

class FeatureImportance:
    def __init__(self, df, df_orig=None, var_display_name=None, file_name='', featimp_folder=''):
        
        '''
        Evaluate feature importance using SHAP.

        Args:
            - df: dataframe with [time_var, 'y', ....]
            - df_orig: dataframe of variables to be used for x-axis distribution. If df has been scaled, takes original values
                        for the variables. If None is set equal to df
            - var_display_name: dictionary of {'original_variable_name': 'name to show on plot'} 
        '''
        
        self.df=df
        self.df_orig=df_orig
        self.var_display_name=var_display_name
        self.file_name=file_name
        self.featimp_folder=featimp_folder

    def eval_shap_values(self, model_type='LightGBM', reload=False):

        pkl_path=os.path.join(self.featimp_folder, self.file_name+'_'+model_type+'_shap_values.pkl')
        
        if not reload or not os.path.exists(pkl_path):
        
            # reload model
            model_path=os.path.join(self.featimp_folder, self.file_name+'_'+model_type+'_model_for_explainer.pkl')
            t_mod = joblib.load(model_path)
            cv_iterator=t_mod['cv_iterator']
            out_of_sample_years=t_mod['out_of_sample_years']
            time_var=t_mod['time_var']
            best_variables=t_mod['best_variables']

            # evaluate out-of-sample
            avail_year = np.sort(self.df[time_var].unique())
            year_to_remove = avail_year[-out_of_sample_years:] 

            df_fit = self.df.copy()[~self.df[time_var].isin(year_to_remove)]  # skip year_to_remove, keep in mind .iloc is used below. Used to isolate test set

            shap_log={}
            for split_i, (train_idx, valid_idx) in enumerate(cv_iterator.split(df_fit)):

                # create test set (out-of-sample)
                year_test_start = np.where(avail_year == df_fit[time_var].iloc[valid_idx].max())[0][0] + 1
                test_idx = np.where(self.df[time_var].isin(avail_year[year_test_start:(year_test_start+out_of_sample_years)]))[0]
                X_test = self.df.drop(columns=['y', time_var]).iloc[test_idx][best_variables]    # test_idx is extracted from df, wih .iloc
                y_test = self.df['y'].iloc[test_idx]
                year_test_lab = ','.join(np.sort(self.df[time_var].iloc[test_idx].unique()).astype(str))
                print(f'- Evaluating SHAP on: {year_test_lab} ({len(test_idx)} obs)')

                # select fitted model and run explainer
                model=t_mod['fitted_model']['split_'+str(split_i)]
                explainer = shap.explainers.Tree(model, X_test, model_output='probability')
                shap_values = explainer(X_test)
                
                # recall original values of input variables (used for interaction plot)
                if self.df_orig is not None:
                    data_orig=self.df_orig[self.df_orig['main_index'].isin(y_test.index.values)][shap_values.feature_names]
                    for col in data_orig.columns: # winsorize
                        data_orig[col] = winsorize(data_orig[col], limits=[0.01, 0.01])
                    data_orig=data_orig.to_numpy()
                else:
                    data_orig=shap_values.data
            
                shap_log['split_'+str(split_i)]={'year_lab': year_test_lab, 'shap_values': shap_values,
                                                 'y': y_test, 'data_orig': data_orig}

            # stack shap values for all test set together
            all_splits_lab=','.join([v['year_lab'] for v in shap_log.values()])
            shap_values_all=copy.copy(shap_log['split_0']['shap_values'])
            expected_obs=sum([v['shap_values'].values.shape[0] for v in shap_log.values()])
            y_all=copy.copy(shap_log['split_0']['y'])
            data_orig_all=copy.copy(shap_log['split_0']['data_orig'])
            for i in range(1, split_i+1):
                shap_values_all.values = np.vstack([shap_values_all.values, shap_log['split_'+str(i)]['shap_values'].values])
                shap_values_all.base_values = np.hstack([shap_values_all.base_values, shap_log['split_'+str(i)]['shap_values'].base_values])
                shap_values_all.data = np.vstack([shap_values_all.data, shap_log['split_'+str(i)]['shap_values'].data])
                y_all=pd.concat([y_all, shap_log['split_'+str(i)]['y']], axis=0)
                data_orig_all = np.vstack([data_orig_all, shap_log['split_'+str(i)]['data_orig']])
            shap_log['split_all']={'year_lab': all_splits_lab, 'shap_values': shap_values_all, 'y': y_all, 'data_orig': data_orig_all}

            if shap_values_all.values.shape[0] + shap_values_all.base_values.shape[0] + shap_values_all.data.shape[0] != expected_obs*3:
                print('\n ######## stacked "shap_values_all" rows do not match expected number')

            # save pickle
            joblib.dump(shap_log, pkl_path, compress=('lzma', 3))
            
        else:
            shap_log=joblib.load(pkl_path)

        print('\n- SHAP values pickle saved to', pkl_path)
        
        self.model_type=model_type
        self.shap_log=shap_log
        
    def plot_bar(self):

        '''
        Plot average absolute SHAP importance
        '''

        for split in self.shap_log.keys():

            year_lab = self.shap_log[split]['year_lab']
            shap_values = copy.copy(self.shap_log[split]['shap_values'])
            y = self.shap_log[split]['y']

            if self.var_display_name is not None:
                shap_values.feature_names=[self.var_display_name[var] for var in shap_values.feature_names]
            cohort_var = np.where(y == 1, "EPS Increase", "EPS Decrease").tolist()
            fig=plt.figure()
            ax = fig.add_subplot(111)
            shap.plots.bar(shap_values.cohorts(cohort_var).abs.mean(0), show=False, max_display=shap_values.shape[1])
            ax.set_title(f'SHAP absolute importance for year {year_lab}', fontsize=20)
            ax.set_xlabel("Average absolute impact on predicted probability", fontsize=15)
            save_lab=year_lab if split != 'split_all' else 'All'
            fig.savefig(os.path.join(self.featimp_folder, '00_'+self.file_name+'_'+self.model_type+'SHAP_importance_'+save_lab+'.png'),
                        bbox_inches='tight', dpi=300)
            
            if split != 'split_all':
                plt.close()
        
        print(f'- Plot saved in {self.featimp_folder}\n')
        plt.show()

    def plot_swarm(self):

        '''
        Plot SHAP importance swarmplot
        '''

        for split in self.shap_log.keys():

            year_lab = self.shap_log[split]['year_lab']
            shap_values = copy.copy(self.shap_log[split]['shap_values'])

            if self.var_display_name is not None:
                shap_values.feature_names=[self.var_display_name[var].replace('\n', ' ') for var in shap_values.feature_names]
            fig=plt.figure()
            ax = fig.add_subplot(111)
            shap.plots.beeswarm(shap_values, show=False, max_display=shap_values.shape[1])
            ax.set_title(f'SHAP contribution for year {year_lab}\n', fontsize=20)
            ax.set_xlabel("Impact on predicted probability", fontsize=15)
            save_lab=year_lab if split != 'split_all' else 'All'
            fig.savefig(os.path.join(self.featimp_folder, '01_'+self.file_name+'_'+self.model_type+'SHAP_swarm_'+save_lab+'.png'),
                        bbox_inches='tight', dpi=300)
            
            if split != 'split_all':
                plt.close()
        
        print(f'- Plot saved in {self.featimp_folder}\n')

        plt.show()
        
    def plot_interactions(self, top_features=5, custom_features=[], multirow_label=True, alpha=0.7):

        '''
        Args:
            - top_features: (int) number of features to use, sorted by average absolute SHAP
            - custom_features: (list) if not [], list of variables to be used for the interactions. Overrides top_features
            - multirow_label: if False, '\n' in features names (if any) will be replaced with ' '
        '''

        for split in self.shap_log.keys():

            year_lab = self.shap_log[split]['year_lab']
            shap_values = copy.copy(self.shap_log[split]['shap_values'])
            y = self.shap_log[split]['y']
            data_orig=self.shap_log[split]['data_orig']

            # set original values of input variables
            shap_values.data=data_orig

            # select variables to compare
            if len(custom_features) > 0:
                work_features=custom_features
            else:
                var_idx=shap_values.abs.mean(0).argsort[-top_features:].values.tolist()[::-1]
                work_features=[shap_values.feature_names[i] for i in var_idx]

            # adjust names
            if self.var_display_name is not None:
                var_display_name_t=copy.copy(self.var_display_name)
                if not multirow_label:
                    var_display_name_t={k: v.replace('\n', ' ') for k, v in var_display_name_t.items()}
                shap_values.feature_names=[var_display_name_t[var] for var in shap_values.feature_names]
            else:
                var_display_name_t=dict(zip(work_features, work_features))

            cc=1
            for i in range(len(work_features)):
                for j in range(len(work_features)):
                    
                    if i != j:

                        i_var_lab=work_features[i]
                        j_var_lab=work_features[j]
                        i_var_new=[v for k, v in var_display_name_t.items() if k == i_var_lab][0]
                        j_var_new=[v for k, v in var_display_name_t.items() if k == j_var_lab][0]

                        shap.plots.scatter(shap_values[:, i_var_new], color=shap_values[:, j_var_new], alpha=alpha, show=False)
                        ax = plt.gca()
                        ax.set_title(f'SHAP interaction for year {year_lab}\n', fontsize=20)
                        ax.set_ylabel(f'SHAP value for\n{i_var_new}')
                        save_lab=year_lab if split != 'split_all' else 'All'
                        plt.savefig(os.path.join(self.featimp_folder,'02_'+self.file_name+'_'+self.model_type+
                                                 'SHAP_interaction_'+str(cc).zfill(2)+'_'+i_var_lab+'_'+j_var_lab+'_'+save_lab+'.png'),
                                    bbox_inches='tight', dpi=300)
                        cc+=1

                        if i != 0 or split != 'split_all':
                            plt.close()                    

        print(f'- Plot saved in {self.featimp_folder}\n')

        plt.show()

        