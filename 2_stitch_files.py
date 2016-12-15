import pandas as pd
import pickle
import os
from tqdm import tqdm

files = [x for x in os.listdir('output') if 'result' in x]
master_df = None
for i, file in enumerate(tqdm(files)):
    try:
        tmp_df = pickle.load(open('output/' + file, 'rb'))
        if 'result' in file:
            if master_df is None:
                master_df = tmp_df
            else:
                master_df = master_df.append(tmp_df)
    except EOFError:
        pass
    os.remove('output/' + file)
pickle.dump(master_df.reset_index().drop(['index'], axis=1), open('output/final_df.pkl', 'wb'))
