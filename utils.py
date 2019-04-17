import numpy as np
import pandas as pd
import datetime

filename = 'data/MFI_with_RS_new_filtering.csv'
df_filtered = pd.read_csv(filename, encoding = "latin").set_index('RSNR', drop=False)
ALL_ABS_TYP1 = np.asarray(['B78', 'B53', 'B35', 'B75', 'B51', 'B62', 'B7', 'B18', 'B67', 'B8',
       'B27', 'B64', 'B56', 'B42', 'B81', 'B71', 'B82', 'B39', 'B50',
       'B65', 'B55', 'B54', 'B76', 'B61', 'B77', 'B49', 'B73', 'B45',
       'B60', 'B37', 'B52', 'B47', 'B72', 'Cw10', 'B41', 'B38', 'B13',
       'B48', 'Cw9', 'B46', 'A2', 'Cw8', 'A68', 'Cw1', 'B59', 'A69',
       'Cw14', 'A66', 'Cw17', 'Cw2', 'A33', 'A34', 'Cw12', 'Cw5', 'A11',
       'A24', 'Cw16', 'Cw18', 'A23', 'Cw6', 'Cw15', 'Cw4', 'Cw7', 'A30',
       'B57', 'B63', 'B44', 'A32', 'A80', 'B58', 'A29', 'A36', 'A74',
       'A25', 'A43', 'A3', 'A31', 'A1', 'A26', 'Bw6'])
ALL_ABS_TYP2 = np.asarray(['DP11', 'DR13', 'DR4', 'DQ4', 'DR51', 'DP6', 'DR52', 'DQ6', 'DQ7',
       'DQ2', 'DQ8', 'DQ9', 'DP19', 'DQ5', 'DR7', 'DR1', 'DR53', 'DP4',
       'DP5', 'DR12', 'DR10', 'DP2', 'DP1', 'DP20', 'DR8', 'DR9', 'DP23',
       'DR15', 'DP28', 'DP14', 'DP10', 'DR16', 'DR11', 'DR18', 'DR17',
       'DR14', 'DR103', 'DP17', 'DP3', 'DP13', 'DP9', 'DP18', 'DP15'])
ALL_ABS = np.asarray(['B78', 'B53', 'B35', 'B75', 'B51', 'B62', 'B7', 'B18', 'B67', 'B8',
       'B27', 'B64', 'B56', 'B42', 'B81', 'B71', 'B82', 'B39', 'B50',
       'B65', 'B55', 'B54', 'B76', 'B61', 'B77', 'B49', 'B73', 'B45',
       'B60', 'B37', 'B52', 'B47', 'B72', 'Cw10', 'B41', 'B38', 'B13',
       'B48', 'Cw9', 'B46', 'A2', 'Cw8', 'A68', 'Cw1', 'B59', 'A69',
       'Cw14', 'A66', 'Cw17', 'Cw2', 'A33', 'A34', 'Cw12', 'Cw5', 'A11',
       'A24', 'Cw16', 'Cw18', 'DP11', 'DR13', 'DR4', 'DQ4', 'A23', 'DR51',
       'DP6', 'DR52', 'Cw6', 'Cw15', 'Cw4', 'Cw7', 'A30', 'B57', 'B63',
       'DQ6', 'DQ7', 'DQ2', 'DQ8', 'DQ9', 'DP19', 'DQ5', 'DR7', 'DR1',
       'DR53', 'DP4', 'DP5', 'DR12', 'B44', 'A32', 'A80', 'B58', 'A29',
       'DR10', 'DP2', 'DP1', 'DP20', 'DR8', 'A36', 'A74', 'DR9', 'DP23',
       'DR15', 'DP28', 'DP14', 'DP10', 'DR16', 'A25', 'A43', 'A3', 'DR11',
       'DR18', 'DR17', 'DR14', 'DR103', 'DP17', 'A31', 'DP3', 'A1',
       'A26', 'DP13', 'DP9', 'DP18', 'DP15', 'Bw6'])

def filter_patients(df_in, analysis_type="1"):
    df_filtered_typ = df_in[df_in['TYP']=='Lumi-Single-Klasse'+ analysis_type]
    # Let's only keep patients with at least 2 measurements
    df_n = df_filtered_typ.groupby('RSNR')['IL_DAT'].nunique()
    print('{} patients need to be excluded as they only have one date of measurements.'.format((df_n==1).sum()))
    df_filtered_typ = df_filtered_typ.set_index('RSNR', drop=False)[df_n>1]
    return df_filtered_typ

def get_time_series(df, Ab_id, dummy_value_inter=499, dummy_value_whole=np.nan):    
    # Transform into datetime objects for sorting
    dates = [datetime.datetime.strptime(x, '%d.%m.%Y') for x in list(df['IL_DAT'].unique())]
    dates = sorted(dates)
    
    # Check if there is any value matching the ab id
    values = []
    if Ab_id in df['I_NAM'].values:
        for date in dates:
            str_date = datetime.datetime.strftime(date, '%d.%m.%Y')
            df_date = df[df['IL_DAT']==str_date]
            if Ab_id in df_date['I_NAM'].values:
                val = df_date[df_date['I_NAM'] == Ab_id]['I_VAL'].values
                values.append(float(val[0]))
            else:
                values.append(dummy_value_inter)
        return np.asarray(values), dates
    else:
        return dummy_value_whole*np.ones(len(dates)), dates

def get_evolution_df(df, complete=False, analysis_type='1'):
    evo_df = pd.DataFrame()
    if complete:
        # Need to check if we only consider type1 or type2 or both...
        if analysis_type == '1':
            # CAREFUL: There is no automatic check to verify...
            list_abs = ALL_ABS_TYP1 # Generate a ts for every ab
        elif analysis_type == '2':
            list_abs = ALL_ABS_TYP2
        else:
            list_abs = ALL_ABS
    else:
        list_abs = list(df['I_NAM'].unique()) 
    for ab_id in list_abs:
        ts, dates = get_time_series(df[df['TYP']=='Lumi-Single-Klasse'+analysis_type], ab_id, dummy_value_inter=499, dummy_value_whole=499)
        evo_df[ab_id] = ts
    dates = pd.DatetimeIndex(dates)
    evo_df = evo_df.set_index(dates)
    return evo_df
