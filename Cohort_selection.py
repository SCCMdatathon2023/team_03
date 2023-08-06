import pandas as pd
import numpy as np
import os
from .utils import get_demography, print_demo

# get parent directory of this file
script_dir = os.path.dirname(__file__)

# get root directory of this project (two levels up from this file)
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

# MIMIC
df0 = pd.read_csv(os.path.join(root_dir, 'data', 'MIMIC_data.csv'))

print(len(df0), "Initial rows in extracted MIMIC\n")
demo0 = print_demo(get_demography(df0))
print(f"{len(df0)} sepsis stays \n({demo0})\n")

df1 = df0[df0.sepsis3 == 1]
print(f"Removed {len(df0) - len(df1)} stays without sepsis")
demo1 = print_demo(get_demography(df1))
print(f"{len(df1)} sepsis stays \n({demo1})\n")

df1['mv_24hr'] = np.where((df1['mech_vent_overall']== 1) & (df1['MV_init_offset_abs']<=1), 1, 0)
df1['vp_24hr'] = np.where((df1['vasopressor_overall']== 1) & (df1['VP_init_offset_abs']<=1), 1, 0)
df1['rrt_72hr'] = np.where((df1['rrt_overall']== 1) & (df1['RRT_init_offset_abs']<=3), 1, 0)

df2 = df1[df1.los_icu >= 1]
print(f"Removed {len(df1) - len(df2)} stays with less than 24 hours")
demo2 = print_demo(get_demography(df2))
print(f"{len(df2)} stays with sepsis and LoS > 24h \n({demo2})\n")

df3 = df2[df2.admission_age >= 18]
print(f"Removed {len(df2) - len(df3)} stays with non-adult patient")
demo3 = print_demo(get_demography(df3))
print(f"{len(df3)} stays with sepsis, lactate day 1, LoS > 24h, adult patient \n({demo3})\n")

df4 = df3.sort_values(by=["subject_id", "stay_id"], ascending=True).groupby(
    'subject_id').apply(lambda group: group.iloc[0, 1:])
print(f"Removed {len(df3) - len(df4)} recurrent stays")
demo4 = print_demo(get_demography(df4))
print(f"{len(df4)} adults with sepsis, lactate day 1, LoS > 24h, adult patient, 1 stay per patient \n({demo4})\n")

cols_na = ['major_surgery', 'hypertension_present', 'heart_failure_present', 
            'copd_present', 'asthma_present', 'cad_present', 'ckd_stages', 
            'connective_disease', 'pneumonia', 'uti', 'biliary', 'skin', 'respiration',
            'coagulation', 'cardiovascular', 'cns', 'liver']

for c in cols_na:
    df4[c] = df4[c].fillna(0)

lab_ranges = {'po2_min': [0, 90, 1000],
        'pco2_max': [0, 40, 200],
        'ph_min': [5, 7.35, 10],
        'lactate_max': [0, 1.05, 30],
        'glucose_max': [0, 95, 2000],
        'sodium_min': [0, 140, 160],
        'potassium_max': [0, 3.5, 9.9],
        'cortisol_min': [0, 20, 70],
        'fibrinogen_min': [0, 200, 1000],
        'inr_max': [0, 1.1, 10],
        'resp_rate_mean': [0, 15, 50],
        'heart_rate_mean': [0, 90, 250],
        'mbp_mean': [0, 85, 200],
        'temperature_mean': [32, 36.5, 45],
        'spo2_mean': [0, 95, 100]
}

for lab in lab_ranges.keys():
    df4[lab] = np.where(df4[lab] < lab_ranges[lab][0], 0, df4[lab])
    df4[lab] = np.where(df4[lab] > lab_ranges[lab][2], 0, df4[lab])
    df4[lab] = np.where(df4[lab] == 0, lab_ranges[lab][1], df4[lab])
    df4[lab] = df4[lab].fillna(lab_ranges[lab][1])

df4['hemoglobin_min'] = df4['hemoglobin_min'].apply(lambda x: 0 if x < 3 else x)
df4['hemoglobin_min'] = df4['hemoglobin_min'].apply(lambda x: 0 if x > 30 else x)
df4['hemoglobin_min'] = df4['hemoglobin_min'].fillna(0)
df4['hemoglobin_min'] = df4.apply(lambda row: 12 if (row.hemoglobin_min == 0) \
                                                        & (row.sex_female == 1) \
                                                    else row.hemoglobin_min, axis=1)

df4['hemoglobin_min'] = df4.apply(lambda row: 13.5 if (row.hemoglobin_min == 0) \
                                                        & (row.sex_female == 0) \
                                                        else row.hemoglobin_min, axis=1)

df4['fluids_volume_norm_by_los_icu'] = df4['fluids_volume_norm_by_los_icu'].fillna(df4['fluids_volume_norm_by_los_icu'].mean())

print(f"df4 length after confounder imputation {len(df4)}")

#df4.to_csv(os.path.join(root_dir, 'data/cohorts', 'SCCM_cohort.csv'))