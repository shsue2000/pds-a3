import pandas as pd
import numpy as np
import pandas as pd
from surprise import Dataset
from surprise import Reader
from surprise import KNNBaseline
from surprise import accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

df_train = pd.read_csv('train.tsv',sep='\t',
                         names=['RowID','BeerID','ReviewerID',
                                  'BeerName','BeerType','rating'])


df_val = pd.read_csv('val.tsv',sep='\t',
                         names=['RowID','BeerID','ReviewerID',
                                  'BeerName','BeerType','rating'])

df_test = pd.read_csv('test.tsv',sep='\t',
                         names=['RowID','BeerID','ReviewerID',
                                  'BeerName','BeerType'])

cleant = df_train.drop(['RowID','BeerName','BeerType'],axis=1)
cleanv = df_val.drop(['RowID','BeerName','BeerType'],axis=1)
cleantest = df_test.drop(['RowID','BeerName','BeerType'], axis=1)
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(cleant[['BeerID','ReviewerID',
                                    'rating']],reader)

datav = Dataset.load_from_df(cleanv[['BeerID','ReviewerID',
                                     'rating']],reader)


trainset = data.build_full_trainset()
NA,valset = train_test_split(datav, test_size=1.0)

knn = KNNBaseline(k=100)
model = knn.fit(trainset)

predictions = knn.test(valset)

accuracy.mae(predictions, verbose=True)

cleantest.loc[:, 'rating'] = 0
test_processed = Dataset.load_from_df(cleantest[['BeerID','ReviewerID','rating']], reader)
NA, test = train_test_split(test_processed, test_size=1.0, shuffle=False)
final = knn.test(test)

final_df = pd.DataFrame(final)

test_id = df_test.drop(['BeerID', 'ReviewerID','BeerName','BeerType'], axis=1)
final_pred = final_df.drop(['uid', 'iid', 'r_ui', 'details'], axis=1)
df_final = pd.concat([test_id, final_pred], axis=1)
df_final = df_final.rename(columns={'est':'Prediction'})

df_final.to_csv('A3-3.tsv', sep = '\t', header=False, index=False)