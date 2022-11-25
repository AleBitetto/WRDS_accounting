import pandas as pd
import numpy as np
from timeit import default_timer as timer
import datetime
import re
import itertools
import functools
import time


def summary_stats(df=None, date_format='D', n_digits=2):

    '''
      date_format: show dates up to days https://numpy.org/doc/stable/reference/arrays.datetime.html#arrays-dtypes-dateunits
      n_digits: rounding digits for min, max, mean, ...
    '''

    import pandas as pd
    import numpy as np
    
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

        perc = np.quantile(val, [0.01, 0.05, 0.95, 0.99])

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
                                'MEDIAN': np.round(np.median(val), n_digits),
                                'PERC1': np.round(perc[0], n_digits),
                                'PERC5': np.round(perc[1], n_digits),
                                'PERC95': np.round(perc[2], n_digits),
                                'PERC99': np.round(perc[3], n_digits),
                                'SUM': np.round(val.sum(), n_digits)
                               }, index = [0])

        num_stats = num_stats.append(add_row)


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

        cat_stats = cat_stats.append(add_row)


    # Boolean stats
    bool_stats = pd.DataFrame(columns=['VARIABLE', 'TYPE', 'OBS', 'NANs', 'MEAN', 'STDDEV', 'MEDIAN', 'PERC1',
                                       'PERC5', 'PERC95', 'PERC99', 'SUM', 'VALUES_BY_FREQ'])
    for var in bool_cols:

        val = df[var].dropna().values

        perc = np.quantile(val.astype(int), [0.01, 0.05, 0.95, 0.99])
                                                   
        add_row = pd.DataFrame({'VARIABLE': var,
                                'TYPE': str(df[var].dtypes),
                                'OBS': tot_rows,
                                'NANs': df[var].isna().sum(),
                                'MEAN': np.round(val.mean(), n_digits),
                                'STDDEV': np.round(val.std(), n_digits),
                                'MEDIAN': np.round(np.median(val), n_digits),
                                'PERC1': perc[0],
                                'PERC5': perc[1],
                                'PERC95': perc[2],
                                'PERC99': perc[3],
                                'SUM': val.sum(),
                                'VALUES_BY_FREQ': ', '.join([str(k) + ': ' + str(v)
                                                             for k, v in df[var].dropna().value_counts().to_dict().items()])
                               }, index = [0])

        bool_stats = bool_stats.append(add_row)

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
        med_ind = np.median(mapped).astype(int)

        if len(val) > 0:
        
            perc = np.quantile(mapped, [0.01, 0.05, 0.95, 0.99]).astype(int)
            add_row = pd.DataFrame({'VARIABLE': var,
                                    'TYPE': str(df[var].dtypes),
                                    'OBS': tot_rows,
                                    'UNIQUE': df[var].nunique(),
                                    'NANs': df[var].isna().sum(),
                                    'MIN': np.datetime_as_string(val.min(), unit=date_format),
                                    'MAX': np.datetime_as_string(val.max(), unit=date_format),
                                    'MEDIAN': [k for k, v in mapping.items() if v == med_ind][0],
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

        date_stats = date_stats.append(add_row)

    # final stats
    all_col_set = ['VARIABLE', 'TYPE', 'OBS', 'UNIQUE', 'NANs', 'INFs', 'ZEROs', 'BLANKs', 'MEAN', 'STDDEV', 'MIN', 
                                       'PERC1', 'PERC5', 'MEDIAN', 'PERC95', 'PERC99', 'MAX', 'SUM', 'VALUES_BY_FREQ']
    used_col_set = []
    final_stats = pd.DataFrame(columns=all_col_set)
    if num_stats.shape[0] > 0:
        final_stats = final_stats.append(num_stats)
        used_col_set.extend(num_stats.columns)
    if cat_stats.shape[0] > 0:
        final_stats = final_stats.append(cat_stats)
        used_col_set.extend(cat_stats.columns)
    if bool_stats.shape[0] > 0:
        final_stats = final_stats.append(bool_stats)
        used_col_set.extend(bool_stats.columns)
    if date_stats.shape[0] > 0:
        final_stats = final_stats.append(date_stats)
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


def stats_with_description(df, df_vardescr_path, col_to_lowercase=True):
    
    df_stats = summary_stats(df)
    df_vardescr = pd.read_csv(df_vardescr_path, sep=';').drop(columns=['Type'])
    if col_to_lowercase:
        df_vardescr['Variable Name'] = df_vardescr['Variable Name'].str.lower()
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
            ref_tab = ref_tab.append(pd.DataFrame({'gvkey': el, 'ID': str(cc)}))
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
    ref_tab=ref_tab.append(t_sin)
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