import pandas as pd
import os

CSV_PATH : str = 'csv_feature'

dataFrames : list = [pd.read_csv(os.path.join(CSV_PATH,i)) for i in os.listdir(CSV_PATH)]

for i in dataFrames:
   i.set_index(['filename'], drop = True, inplace = True)

result : pd.DataFrame = pd.concat(dataFrames)
result.reset_index(inplace=True)

result.to_csv('feature.csv', index = False)