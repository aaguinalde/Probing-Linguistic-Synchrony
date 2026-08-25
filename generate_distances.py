import numpy as np
import os
import pandas as pd
from scipy.spatial.distance import cosine

directory = ## insert folder including each mbert layer vector

cols = ['filename'] + ['pair_num'] + list(range(13))
df = pd.DataFrame(columns=cols)


for fname in os.listdir(directory):
    file_path = os.path.join(directory, fname)
    filename = fname.split("mbert_")[1].split("_layer")[0]
    layer_num = fname.split("layer-")[1].split(".")[0]
    print('Processing file: ', filename, 'layer:', layer_num )

    arr = np.genfromtxt(file_path, delimiter=",", dtype=float, missing_values="NULL", filling_values=np.nan)

    if layer_num == '0':
        pair_num = 0
        for i in range(len(arr)-1):
            utt_1 = arr[i]
            utt_2 = arr[i+1]
            dist = cosine(utt_1, utt_2)
            
            df.loc[len(df)] = {
                'filename': filename,
                'pair_num': pair_num,
                0: dist
            }

            pair_num += 1

        
    if layer_num != '0':
        pair_num = 0
        for i in range(len(arr)-1):
            utt_1 = arr[i]
            utt_2 = arr[i+1]
            dist = cosine(utt_1, utt_2)

            mask = (df['filename'] == filename) & (df['pair_num'] == pair_num)
            row_idx = df.index[mask][0]

            df.loc[row_idx, int(layer_num)] = dist

            pair_num +=1

print(df.head())

df.to_csv('layerwise_synch.csv')


