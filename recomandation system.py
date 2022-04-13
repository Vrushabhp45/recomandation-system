
# Import Libraries
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine,correlation
# Import Dataset
books=pd.read_csv('C:/Users/HP/PycharmProjects/Excelrdatascience/book_rec.csv',encoding='Latin1')
books2=books.iloc[:,1:]
# Sort by User IDs
books2.sort_values(['User.ID'])
# number of unique users in the dataset
len(books2['User.ID'].unique())
# number of unique books in the dataset
len(books2['Book.Title'].unique())
# converting long data into wide data using pivot table
books3=books2.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating').reset_index(drop=True)
# Replacing the index values by unique user Ids
books3.index=books2['User.ID'].unique()
# Impute those NaNs with 0 values
books3.fillna(0,inplace=True)
# Calculating Cosine Similarity between Users on array data
user_sim=1-pairwise_distances(books3.values,metric='cosine')
# Store the results in a dataframe format
user_sim2=pd.DataFrame(user_sim)
# Set the index and column names to user ids
user_sim2.index=books2['User.ID'].unique()
user_sim2.columns=books2['User.ID'].unique()
# Nullifying diagonal values
np.fill_diagonal(user_sim,0)
# Most Similar Users
user_sim2.idxmax(axis=1)
# extract the books which userId 162107 & 276726 have watched
books2[(books2['User.ID']==162107) | (books2['User.ID']==276726)]
# extract the books which userId 276729 & 276726 have watched
books2[(books2['User.ID']==276729) | (books2['User.ID']==276726)]
user_1=books2[(books2['User.ID']==276729)]
user_2=books2[(books2['User.ID']==276726)]
user_1['Book.Title']
user_2['Book.Title']
pd.merge(user_1,user_2,on='Book.Title',how='outer')