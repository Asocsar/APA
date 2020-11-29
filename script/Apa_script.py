# *----------------------------------*
#           BASIC LIBRARIES
# *----------------------------------*

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

# *------------------------------------------------------*
#                       READING DATA
# *------------------------------------------------------*
onlyfiles = ['../' + f for f in listdir('../') if isfile(join('../', f))]
onlyfiles

def treatement(file, limit=-1):
    file = open(file, encoding='iso-8859-1')
    data = file.read()
    file.close()
    data = data.replace('\n', ' ')
    if limit != -1:
        data = np.array(data.split(' name '))[:-limit]
    else:
        data = np.array(data.split(' name '))
    data = list(filter(lambda x: len(x) > 0, list(map(lambda x: np.array(x.split(' ')).flatten(), data))))[:-1]
    data = np.float32(np.array(data))
    condlist = [2, 3, 8, 9, 11, 13, 14, 15, 18, 31, 32, 33, 34, 37, 39, 40, 43, 50, 57]
    data_selected = []
    for i in range(len(data)):
        data_selected.append([data[i][x] for x in range(len(data[i])) if x in condlist])
    data_selected = np.array(data_selected)
    return data_selected


data = []
for file in onlyfiles:
    if 'cleveland' in file:
        data = data + list(treatement(file, 11))
    else:
        data = data + list(treatement(file))


data = np.array(data)
print(data.shape)


df = pd.DataFrame(data, columns=['age', 'sex', 'chest_pain', 'presure_blood_resting', 
                                          'colesterol','cigarettes_per_day', 'smoker_years',
                                           'sugar', 'electrocardio',
                                          'max_heart_rate',  'res_heart_rate',
                                          'blood_presure_sistoles', 'blood_presure_diastoles', 
                                          'angina', 'rest_after_exercicie_presure',
                                          'slope_of_rest_after_exercise', 'major_vessels', 'thal',
                                          'prob'])


# *------------------------------------------------------*
#       NAN DEFINED VALUES AND INCHORENENT VALUES
# *------------------------------------------------------*
df[df == -9.0] = np.nan
df[df['presure_blood_resting'] == 0] = np.nan
df[df['colesterol'] == 0] = np.nan



# *------------------------------------------------------*
#               FIRST VIEW OF DATA VALUES
# *------------------------------------------------------*

'''
for name in df.columns:
    print('\033[1m'+name+'\033[0;0m')
    print(df[name].unique())
'''

# *------------------------------------------------------*
#               REMOVING OUTLIERS
# *------------------------------------------------------*

col =['colesterol', 'cigarettes_per_day', 'max_heart_rate', 'presure_blood_resting', 
     'blood_presure_diastoles', 'blood_presure_sistoles', 'res_heart_rate']
for name in col:
    df = df[~(np.abs(df[name] - df[name].mean()) > 3*df[name].std())]

# *------------------------------------------------------*
#                   TOTAL NUMBER OF NULLS
# *------------------------------------------------------*
print("Total of nulls", df.isna().sum().sum()/(df.shape[0]*df.shape[1])*100, "%")
print(df.columns)


# *------------------------------------------------------*
#                   CHANGING DATATYPE
# *------------------------------------------------------*
bool_columns = ['angina', 'sugar']
for name in bool_columns:
    df.loc[df[name].notna(), name] = df[df[name].notna()][name].astype(int)

df.loc[:,'angina'] = df['angina'].apply(lambda x: x > 0)
df.loc[:,'sugar'] = df['sugar'].apply(lambda x: x > 0)
df.loc[:,'sex'] = df['sex'].apply(lambda x: x > 0)

# *------------------------------------------------------*
#                   ERROR VALUES
# *------------------------------------------------------*
df['thal'].replace(1.0, np.nan, inplace=True)
df['thal'].replace(5.0, np.nan, inplace=True)

df['slope_of_rest_after_exercise'].replace(0.0, np.nan, inplace=True)
df['major_vessels'].replace(9.0, np.nan, inplace=True)

# *------------------------------------------------------*
#               ONE HOT ENCODING FOR CATEGORIACAL
# *------------------------------------------------------*
categoric_columns = ["thal", "slope_of_rest_after_exercise", "prob", "major_vessels", "electrocardio", "chest_pain"]
for name in categoric_columns:
    OneHot = pd.get_dummies(df[name], prefix=name)
    df = df.join(OneHot)
    del df[name]


# *------------------------------------------------------*
#           NORMALIZING AND IMPUTING MISSING VALUES
# *------------------------------------------------------*



min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df.values)
df2 = pd.DataFrame(df_scaled)



imputer = KNNImputer()

df2[:]= imputer.fit_transform(df2)


df2 = pd.DataFrame(min_max_scaler.inverse_transform(df2), columns=['age', 'sex', 'presure_blood_resting', 'colesterol',
       'cigarettes_per_day', 'smoker_years', 'sugar', 'max_heart_rate',
       'res_heart_rate', 'blood_presure_sistoles', 'blood_presure_diastoles',
       'angina', 'rest_after_exercicie_presure', 'thal_3.0', 'thal_6.0',
       'thal_7.0', 'slope_of_rest_after_exercise_1.0',
       'slope_of_rest_after_exercise_2.0', 'slope_of_rest_after_exercise_3.0',
       'prob_0.0', 'prob_1.0', 'prob_2.0', 'prob_3.0', 'prob_4.0',
       'major_vessels_0.0', 'major_vessels_1.0', 'major_vessels_2.0',
       'major_vessels_3.0', 'electrocardio_0.0', 'electrocardio_1.0',
       'electrocardio_2.0', 'chest_pain_1.0', 'chest_pain_2.0',
       'chest_pain_3.0', 'chest_pain_4.0'])



'''
for name in df2.columns:
    print('\033[1m'+name+'\033[0;0m')
    print(df2[name].unique())

'''

# *------------------------------------------------------*
#                   STATISTICAL ANALYSIS
# *------------------------------------------------------*
#print(df2.describe)
#hist = df2.hist(bins=50, figsize=(20,20))
#box = df2.boxplot()
#plt.xticks(rotation='vertical')
#plt.xticks(rotation='horizontal')
#for name in df2.columns:
#    fig, axes= plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 4]}, figsize=(9,5))
#    df2.boxplot(column=name,ax=axes[0])
#    df2.hist(column=name, ax=axes[1])
#plt.show()




# *------------------------------------------------------*
#                       UTILITIES
# *------------------------------------------------------*


'''
# *------------------------------------------------------*
#           DIFFERENCES BETWEEN TWO DATAFRAMES
# *------------------------------------------------------*
ne_stacked = (df3 != df2).stack()
changed = ne_stacked[ne_stacked]
changed.index.names = ['id', 'col']

difference_locations = np.where(df3 != df2)
changed_from = df3.values[difference_locations]
changed_to = df2.values[difference_locations]
a = pd.DataFrame({'from': changed_from, 'to': changed_to}, index=changed.index)
'''


'''
# *------------------------------------------------------*
#           OUTLIERS AFTER IMPUTATION
# *------------------------------------------------------*
col = ['age', 'sex', 'presure_blood_resting', 'colesterol',
       'cigarettes_per_day', 'smoker_years', 'max_heart_rate',
       'res_heart_rate', 'blood_presure_sistoles', 'blood_presure_diastoles']

for name in col:
    df2 = df2[~(np.abs(df2[name] - df2[name].mean()) > 3*df2[name].std())]  
'''