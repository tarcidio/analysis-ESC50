import pandas as pd
import os

CSV_PATH : str = 'csv_feature'

dfs : list = [pd.read_csv(os.path.join(CSV_PATH,i)) for i in os.listdir(CSV_PATH)]

for i in dfs:
   i.set_index(['filename'], drop = True, inplace = True)

result : pd.DataFrame = pd.concat(dfs)
result.reset_index(inplace=True)

result.to_csv(os.path.join(CSV_PATH, 'feature.csv'))