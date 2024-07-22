import pandas as pd
import useritemmatrix
import normalization
from items_embedding import items_embeddings
import pickle

dim = 40
data = pd.read_csv('last.csv')
csr = useritemmatrix.UserItemMatrix(data).csr_matrix
norm = normalization.Normalization.tf_idf(csr)
embed = items_embeddings(norm, dim)

with open('embed_.pickle', 'wb') as file:
    pickle.dump(embed, file)





