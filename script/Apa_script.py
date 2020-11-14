import pandas as pd
import numpy as np

from os import listdir
from os.path import isfile, join


onlyfiles = ['../' + f for f in listdir('../') if isfile(join('../', f))]
onlyfiles = onlyfiles[0]
print(onlyfiles)

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
    #condlist = [2, 3, 8, 9, 11, 15, 18, 31, 32, 33, 34, 37, 39, 40, 41, 42, 43, 50, 57]
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
print(data[0])



df = pd.DataFrame(data, columns=['age', 'sex', 'chest_pain', 'presure_blood_resting', 
                                          'colesterol','cigarettes_per_day', 'smoker_years',
                                           'sugar', 'electrocardio',
                                          'max_heart_rate',  'res_heart_rate',
                                          'blood_presure_sistoles', 'blood_presure_diastoles', 
                                          'angina', 'rest_after_exercicie_presure',
                                          'slope_of_rest_after_exercice', 'major_vessels', 'thal',
                                          'prob'])


df[df == -9.0] = np.nan
df[df['presure_blood_resting'] == 0] = np.nan
df[df['colesterol'] == 0] = np.nan





for name in df.columns:
    print('\033[1m'+name+'\033[0;0m')
    print(df[name].unique())


print(df.isna().sum().sum()/(df.shape[0]*df.shape[1])*100)


print(df.isna().sum().sum())


for name in df.columns:
    n = float(int(df[name].isna().sum()/(df.shape[0])*10000))/100
    print('column {} : \n'.format(name) + '\t\033[1m' + '{}%'.format(n) + '\033[0;0m' + ' nulls\n')
