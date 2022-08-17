# import libraries
from heapq import merge
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

# import training data
train_df=pd.read_csv('/home/gunnar/playgroundProject/data/train.csv')
test_df=pd.read_csv('/home/gunnar/playgroundProject/data/test.csv')




# One-hot encoding of categorical features attribute_0 and attribute_1 ##################################
#first get dummy dataframes
a1_train = pd.get_dummies(train_df['attribute_0'], prefix='attribute_0')
a2_train = pd.get_dummies(train_df['attribute_1'], prefix='attribute_1')

#create dataframe of one-hot encoded features; note that this dataframe does not include product_code
categorical_df_train = pd.concat([a1_train, a2_train], axis=1, join='inner')



# One-hot encoding of categorical features attribute_0 and attribute_1 ##################################
#first get dummy dataframes
a1_test = pd.get_dummies(train_df['attribute_0'], prefix='attribute_0')
a2_test = pd.get_dummies(train_df['attribute_1'], prefix='attribute_1')

#create dataframe of one-hot encoded features; note that this dataframe does not include product_code
categorical_df_test = pd.concat([a1_test, a2_test], axis=1, join='inner')




#  Standardize ########################
#lists of columns to not convert to z-scores
non_standardized_cols_train = ['id', 'product_code', 'attribute_0', 'attribute_1', 'failure']
non_standardized_cols_test = ['id', 'product_code', 'attribute_0', 'attribute_1']

#create scaler objects and convert desired features to z-scores
scaler_train = StandardScaler().fit(train_df.drop(columns=non_standardized_cols_train)).transform(train_df.drop(columns=non_standardized_cols_train))
scaler_test = StandardScaler().fit(test_df.drop(columns=non_standardized_cols_test)).transform(test_df.drop(columns=non_standardized_cols_test))

#create dataframes of features which are now converted to z-scores
numerical_df_train = pd.DataFrame(scaler_train, columns=train_df.drop(columns=non_standardized_cols_train).columns)
numerical_df_test = pd.DataFrame(scaler_test, columns=test_df.drop(columns=non_standardized_cols_test).columns)


#merge categorical and numerical dataframes
merged_df_train = pd.concat([train_df['id'], train_df['product_code'], train_df['failure'], numerical_df_train, categorical_df_train], axis = 1, join='inner')
merged_df_test = pd.concat([test_df['id'], test_df['product_code'], numerical_df_test, categorical_df_test], axis = 1, join='inner')



# Impute Numerical features ##################
#in the training data, there are five distinct product_codes: A,B,C,D,E
#we will impute nan values wrt each chunk of train_df corresponding to a given product code

#define empty dataframe which we will append each imputed chunk_df to
imputed_df_train = pd.DataFrame()
imputed_df_test = pd.DataFrame()

#iterate over all distinct product codes in the training dataframe and impute the numerical features of each block corresponding to a given product code
for prod_code in merged_df_train['product_code'].unique().tolist():
    chunk_df = merged_df_train[merged_df_train['product_code'] == prod_code]

    #create an object for KNNImputer
    imputer = KNNImputer(n_neighbors=3)

    #fill nan entries in merged_df with the mean of the feature variables in the n_neighbors closest to the row of the given nan entry wrt euclidean distance
    temp_imputed_df = pd.DataFrame(imputer.fit_transform(chunk_df.drop(columns=['product_code', 'id', 'failure'])), columns=chunk_df.drop(columns=['product_code', 'id', 'failure']).columns)
    temp_imputed_df['product_code'] = [prod_code] * len(chunk_df)
    temp_imputed_df['id'] = chunk_df['id'].tolist()
    temp_imputed_df['failure'] = chunk_df['failure'].tolist()
    imputed_df_train = imputed_df_train.append(temp_imputed_df)
    #imputed_df_train = pd.concat([imputed_df_train, temp_imputed_df], axis=1, join='inner')


#iterate over all distinct product codes in the testing dataframe and impute the numerical features of each block corresponding to a given product code
for prod_code in merged_df_test['product_code'].unique().tolist():
    chunk_df = merged_df_test[merged_df_test['product_code'] == prod_code]

    #create an object for KNNImputer
    imputer = KNNImputer(n_neighbors=3)

    #fill nan entries in merged_df with the mean of the feature variables in the n_neighbors closest to the row of the given nan entry wrt euclidean distance
    temp_imputed_df = pd.DataFrame(imputer.fit_transform(chunk_df.drop(columns=['product_code', 'id'])), columns=chunk_df.drop(columns=['product_code', 'id']).columns)
    temp_imputed_df['product_code'] = [prod_code] * len(chunk_df)
    temp_imputed_df['id'] = chunk_df['id'].tolist()
    imputed_df_test = imputed_df_test.append(temp_imputed_df)
    #imputed_df_test = pd.concat([imputed_df_test, temp_imputed_df], axis=1, join='inner')



imputed_df_train = imputed_df_train.drop(columns=['attribute_1_material_8'])
imputed_df_test = imputed_df_test.drop(columns=['attribute_1_material_8'])


#update the new merged dataframes
merged_df_train = imputed_df_train
merged_df_test = imputed_df_test



# Determine which value of k to use for K-means clustering
# #building the clustering model and calculating values of distortion and intertia
# distortions = []
# inertias = []
# mapping1 = {}
# mapping2 = {}
# K = range(1, 10)
  
# for k in K:
#     # Building and fitting the model
#     kmeanModel = KMeans(n_clusters=k).fit(merged_df.drop(columns=['product_code', 'id', 'failure']))
#     kmeanModel.fit(merged_df.drop(columns=['product_code', 'id', 'failure']))
  
#     distortions.append(sum(np.min(cdist(merged_df.drop(columns=['product_code', 'id', 'failure']), kmeanModel.cluster_centers_,
#                                         'euclidean'), axis=1)) / merged_df.drop(columns=['product_code', 'id', 'failure']).shape[0])
#     inertias.append(kmeanModel.inertia_)
  
#     mapping1[k] = sum(np.min(cdist(merged_df.drop(columns=['product_code', 'id', 'failure']), kmeanModel.cluster_centers_,
#                                    'euclidean'), axis=1)) / merged_df.drop(columns=['product_code', 'id', 'failure']).shape[0]
#     mapping2[k] = kmeanModel.inertia_

# for key, val in mapping1.items():
#     print(f'{key} : {val}')

# plt.plot(K, distortions, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method using Distortion')
# plt.show()


#save lists of features to drop before creating a cluster feature
product_code_list_train = merged_df_train['product_code'].tolist()
id_list_train = merged_df_train['id'].tolist()
product_code_list_test = merged_df_test['product_code'].tolist()
id_list_test = merged_df_test['id'].tolist()
failure_list = merged_df_train['failure'].tolist()

#drop the features we don't want to include in the K-means clustering implementation
merged_df_train = merged_df_train.drop(columns=['product_code', 'id', 'failure'])
merged_df_test = merged_df_test.drop(columns=['product_code', 'id'])


#create the kmeans object
kmeans = KMeans(n_clusters=6, n_init=10, random_state=0)

#create the 'Cluster' feature
merged_df_train['Cluster'] = kmeans.fit_predict(merged_df_train)
merged_df_test['Cluster'] = kmeans.fit_predict(merged_df_test)

# One-hot encoding of Cluster labels ##################################
#first get dummy dataframes
dummy_train = pd.get_dummies(merged_df_train['Cluster'], prefix='Cluster')
dummy_test = pd.get_dummies(merged_df_test['Cluster'], prefix='Cluster')

#create dataframe of one-hot encoded features; note that this dataframe does not include product_code
merged_df_train = pd.concat([merged_df_train, dummy_train], axis=1, join='inner')
merged_df_test = pd.concat([merged_df_test, dummy_test], axis=1, join='inner')

#drop the original Cluster feature
merged_df_train = merged_df_train.drop(columns=['Cluster'])
merged_df_test = merged_df_test.drop(columns=['Cluster'])



# Create features as being the sum of each pair of numerical features
#get list of numerical feature column names
num_col_list = merged_df_train.columns.tolist()

#construct a new feature for every pair of numerical features that represents the sum of a given pair of features
num_feature_pairs_list = [(a, b) for idx, a in enumerate(num_col_list) for b in num_col_list[idx + 1:]]

for pair in num_feature_pairs_list:
    merged_df_train[str(pair[0] + '+' + pair[1])] = merged_df_train[pair[0]] + merged_df_train[pair[1]]
    merged_df_test[str(pair[0] + '+' + pair[1])] = merged_df_test[pair[0]] + merged_df_test[pair[1]]



#include these three features which we had dropped throughout the majority of processing
merged_df_train['product_code'] = product_code_list_train
merged_df_train['id'] = id_list_train
merged_df_test['product_code'] = product_code_list_test
merged_df_test['id'] = id_list_test
merged_df_train['failure'] = failure_list

merged_df_train = merged_df_train.drop(columns=['measurement_6+measurement_11', 'measurement_11+measurement_15', 'measurement_9+measurement_11', 'measurement_4+measurement_11', 'Cluster_0+Cluster_1', 'measurement_4+measurement_9', 'measurement_11+measurement_13', 'measurement_11+measurement_16', 'measurement_1+measurement_11', 'measurement_9+measurement_15', 'measurement_3+measurement_11', 'measurement_4+measurement_15', 'attribute_0_material_5+Cluster_0', 'measurement_2+measurement_4', 'measurement_11+measurement_12', 'measurement_16+attribute_0_material_7', 'measurement_9+measurement_16', 'attribute_3+measurement_7', 'measurement_7+Cluster_1', 'attribute_2+measurement_2', 'attribute_2+measurement_17', 'attribute_1_material_5+attribute_1_material_6', 'attribute_1_material_6+Cluster_2', 'attribute_1_material_6+Cluster_4', 'attribute_0_material_5+attribute_0_material_7', 'attribute_0_material_7+Cluster_1', 'measurement_1+measurement_9', 'measurement_0+measurement_11', 'measurement_6+measurement_9', 'measurement_4+measurement_12', 'attribute_1_material_6+Cluster_3', 'attribute_3+measurement_0', 'measurement_4+measurement_8', 'measurement_9+measurement_13', 'measurement_4+measurement_6', 'measurement_1+measurement_4', 'measurement_7+Cluster_0', 'measurement_4+measurement_17', 'measurement_10+measurement_11', 'Cluster_0+Cluster_2', 'attribute_2+measurement_7', 'attribute_2+attribute_1_material_6', 'measurement_4+measurement_13', 'measurement_8+measurement_11', 'measurement_7+attribute_1_material_5', 'attribute_1_material_6+Cluster_5', 'measurement_0+measurement_4', 'measurement_9+measurement_12', 'attribute_1_material_6+Cluster_1', 'measurement_12+measurement_15', 'measurement_2+measurement_9', 'measurement_14+Cluster_0', 'measurement_4+measurement_16', 'measurement_6+Cluster_1', 'attribute_2+measurement_6', 'attribute_1_material_5+Cluster_0', 'measurement_4+measurement_5', 'measurement_2+measurement_11', 'measurement_8+measurement_9', 'measurement_3+measurement_9', 'measurement_10+Cluster_1', 'measurement_6+measurement_15', 'measurement_7+measurement_11', 'measurement_0+Cluster_0', 'measurement_14+Cluster_1', 'measurement_11+measurement_14', 'measurement_9+attribute_1_material_6', 'measurement_2+measurement_6', 'measurement_3+measurement_4', 'measurement_4+attribute_1_material_6', 'measurement_4+measurement_10', 'measurement_7+Cluster_2', 'measurement_9+measurement_10', 'measurement_11+measurement_17', 'attribute_2+measurement_10', 'measurement_1+measurement_15', 'measurement_13+Cluster_1', 'Cluster_0+Cluster_3', 'attribute_3+measurement_16', 'measurement_0+Cluster_1', 'attribute_0_material_7+Cluster_0', 'measurement_6+measurement_12', 'measurement_10+attribute_1_material_5', 'measurement_14+attribute_1_material_5', 'attribute_2+measurement_0', 'loading+attribute_1_material_6', 'measurement_9+measurement_17', 'measurement_15+Cluster_0', 'Cluster_1+Cluster_2', 'attribute_0_material_5+Cluster_2', 'measurement_4+measurement_7', 'measurement_4+measurement_14', 'measurement_15+measurement_16', 'attribute_3+measurement_12', 'attribute_3+measurement_10', 'Cluster_0+Cluster_4', 'measurement_6+measurement_17', 'measurement_10+Cluster_0', 'Cluster_1+Cluster_4', 'measurement_7+Cluster_4', 'attribute_2+measurement_14', 'measurement_17+Cluster_1', 'measurement_14+Cluster_3', 'measurement_7+Cluster_5', 'measurement_12+measurement_16', 'measurement_5+Cluster_0', 'attribute_2+measurement_13', 'measurement_7+Cluster_3', 'measurement_1+measurement_12', 'attribute_3+measurement_14', 'attribute_1_material_6+Cluster_0', 'measurement_17+Cluster_0', 'measurement_5+measurement_11', 'measurement_14+Cluster_2', 'attribute_3+Cluster_4', 'measurement_2+measurement_12', 'attribute_3+measurement_15', 'measurement_13+Cluster_0', 'measurement_1+Cluster_1', 'measurement_16+Cluster_0', 'measurement_9+measurement_14', 'attribute_3+Cluster_0', 'measurement_2+attribute_0_material_7', 'measurement_6+measurement_13', 'measurement_0+attribute_1_material_5', 'measurement_5+measurement_9', 'measurement_14+Cluster_5', 'measurement_6+attribute_1_material_6', 'measurement_1+measurement_6', 'measurement_0+measurement_13', 'measurement_0+Cluster_2', 'attribute_3+measurement_17', 'attribute_2+measurement_5', 'measurement_0+measurement_9', 'attribute_3+measurement_6', 'attribute_2+measurement_16', 'measurement_0+Cluster_5', 'measurement_15+Cluster_1', 'measurement_1+measurement_16', 'measurement_2+measurement_15', 'measurement_8+Cluster_0', 'measurement_5+measurement_6', 'measurement_6+attribute_1_material_5', 'measurement_11+attribute_1_material_6', 'measurement_12+measurement_13', 'measurement_0+measurement_3', 'measurement_5+measurement_15', 'measurement_15+attribute_1_material_6', 'measurement_6+measurement_16', 'attribute_2+measurement_1', 'measurement_7+measurement_9', 'attribute_0_material_7+Cluster_3', 'measurement_3+measurement_14', 'measurement_16+Cluster_1', 'measurement_10+measurement_15', 'attribute_2+measurement_15', 'measurement_0+measurement_6', 'measurement_3+measurement_17', 'measurement_1+Cluster_0', 'attribute_3+measurement_1', 'measurement_13+measurement_15', 'measurement_8+measurement_14', 'measurement_5+Cluster_1', 'loading+Cluster_2', 'measurement_14+Cluster_4', 'measurement_13+Cluster_4', 'measurement_0+Cluster_3', 'Cluster_2+Cluster_4', 'measurement_3+attribute_1_material_6', 'attribute_0_material_5+Cluster_4', 'measurement_10+attribute_0_material_7', 'measurement_3+measurement_5', 'measurement_10+Cluster_2', 'measurement_5+attribute_1_material_5', 'measurement_1+measurement_2', 'measurement_13+measurement_17', 'measurement_1+measurement_8', 'measurement_0+Cluster_4', 'measurement_0+measurement_17', 'measurement_15+Cluster_4', 'measurement_10+Cluster_3', 'measurement_10+attribute_0_material_5', 'measurement_15+measurement_17', 'measurement_6+measurement_8', 'measurement_6+Cluster_4', 'measurement_15+Cluster_2', 'measurement_3+measurement_12', 'measurement_8+measurement_12', 'measurement_2+measurement_10', 'measurement_3+measurement_15', 'measurement_0+measurement_2', 'loading+Cluster_4', 'loading+Cluster_0', 'measurement_7+attribute_0_material_5', 'measurement_3+measurement_6', 'attribute_3+Cluster_2', 'measurement_10+Cluster_4', 'measurement_5+measurement_13', 'attribute_3+measurement_8', 'Cluster_0+Cluster_5', 'attribute_2+measurement_3', 'attribute_3+measurement_5', 'measurement_3+measurement_10', 'measurement_5+measurement_12', 'Cluster_1+Cluster_3', 'measurement_2+measurement_14', 'measurement_6+Cluster_0', 'measurement_12+measurement_17', 'measurement_5+attribute_0_material_7'])
merged_df_test = merged_df_test.drop(columns=['measurement_6+measurement_11', 'measurement_11+measurement_15', 'measurement_9+measurement_11', 'measurement_4+measurement_11', 'Cluster_0+Cluster_1', 'measurement_4+measurement_9', 'measurement_11+measurement_13', 'measurement_11+measurement_16', 'measurement_1+measurement_11', 'measurement_9+measurement_15', 'measurement_3+measurement_11', 'measurement_4+measurement_15', 'attribute_0_material_5+Cluster_0', 'measurement_2+measurement_4', 'measurement_11+measurement_12', 'measurement_16+attribute_0_material_7', 'measurement_9+measurement_16', 'attribute_3+measurement_7', 'measurement_7+Cluster_1', 'attribute_2+measurement_2', 'attribute_2+measurement_17', 'attribute_1_material_5+attribute_1_material_6', 'attribute_1_material_6+Cluster_2', 'attribute_1_material_6+Cluster_4', 'attribute_0_material_5+attribute_0_material_7', 'attribute_0_material_7+Cluster_1', 'measurement_1+measurement_9', 'measurement_0+measurement_11', 'measurement_6+measurement_9', 'measurement_4+measurement_12', 'attribute_1_material_6+Cluster_3', 'attribute_3+measurement_0', 'measurement_4+measurement_8', 'measurement_9+measurement_13', 'measurement_4+measurement_6', 'measurement_1+measurement_4', 'measurement_7+Cluster_0', 'measurement_4+measurement_17', 'measurement_10+measurement_11', 'Cluster_0+Cluster_2', 'attribute_2+measurement_7', 'attribute_2+attribute_1_material_6', 'measurement_4+measurement_13', 'measurement_8+measurement_11', 'measurement_7+attribute_1_material_5', 'attribute_1_material_6+Cluster_5', 'measurement_0+measurement_4', 'measurement_9+measurement_12', 'attribute_1_material_6+Cluster_1', 'measurement_12+measurement_15', 'measurement_2+measurement_9', 'measurement_14+Cluster_0', 'measurement_4+measurement_16', 'measurement_6+Cluster_1', 'attribute_2+measurement_6', 'attribute_1_material_5+Cluster_0', 'measurement_4+measurement_5', 'measurement_2+measurement_11', 'measurement_8+measurement_9', 'measurement_3+measurement_9', 'measurement_10+Cluster_1', 'measurement_6+measurement_15', 'measurement_7+measurement_11', 'measurement_0+Cluster_0', 'measurement_14+Cluster_1', 'measurement_11+measurement_14', 'measurement_9+attribute_1_material_6', 'measurement_2+measurement_6', 'measurement_3+measurement_4', 'measurement_4+attribute_1_material_6', 'measurement_4+measurement_10', 'measurement_7+Cluster_2', 'measurement_9+measurement_10', 'measurement_11+measurement_17', 'attribute_2+measurement_10', 'measurement_1+measurement_15', 'measurement_13+Cluster_1', 'Cluster_0+Cluster_3', 'attribute_3+measurement_16', 'measurement_0+Cluster_1', 'attribute_0_material_7+Cluster_0', 'measurement_6+measurement_12', 'measurement_10+attribute_1_material_5', 'measurement_14+attribute_1_material_5', 'attribute_2+measurement_0', 'loading+attribute_1_material_6', 'measurement_9+measurement_17', 'measurement_15+Cluster_0', 'Cluster_1+Cluster_2', 'attribute_0_material_5+Cluster_2', 'measurement_4+measurement_7', 'measurement_4+measurement_14', 'measurement_15+measurement_16', 'attribute_3+measurement_12', 'attribute_3+measurement_10', 'Cluster_0+Cluster_4', 'measurement_6+measurement_17', 'measurement_10+Cluster_0', 'Cluster_1+Cluster_4', 'measurement_7+Cluster_4', 'attribute_2+measurement_14', 'measurement_17+Cluster_1', 'measurement_14+Cluster_3', 'measurement_7+Cluster_5', 'measurement_12+measurement_16', 'measurement_5+Cluster_0', 'attribute_2+measurement_13', 'measurement_7+Cluster_3', 'measurement_1+measurement_12', 'attribute_3+measurement_14', 'attribute_1_material_6+Cluster_0', 'measurement_17+Cluster_0', 'measurement_5+measurement_11', 'measurement_14+Cluster_2', 'attribute_3+Cluster_4', 'measurement_2+measurement_12', 'attribute_3+measurement_15', 'measurement_13+Cluster_0', 'measurement_1+Cluster_1', 'measurement_16+Cluster_0', 'measurement_9+measurement_14', 'attribute_3+Cluster_0', 'measurement_2+attribute_0_material_7', 'measurement_6+measurement_13', 'measurement_0+attribute_1_material_5', 'measurement_5+measurement_9', 'measurement_14+Cluster_5', 'measurement_6+attribute_1_material_6', 'measurement_1+measurement_6', 'measurement_0+measurement_13', 'measurement_0+Cluster_2', 'attribute_3+measurement_17', 'attribute_2+measurement_5', 'measurement_0+measurement_9', 'attribute_3+measurement_6', 'attribute_2+measurement_16', 'measurement_0+Cluster_5', 'measurement_15+Cluster_1', 'measurement_1+measurement_16', 'measurement_2+measurement_15', 'measurement_8+Cluster_0', 'measurement_5+measurement_6', 'measurement_6+attribute_1_material_5', 'measurement_11+attribute_1_material_6', 'measurement_12+measurement_13', 'measurement_0+measurement_3', 'measurement_5+measurement_15', 'measurement_15+attribute_1_material_6', 'measurement_6+measurement_16', 'attribute_2+measurement_1', 'measurement_7+measurement_9', 'attribute_0_material_7+Cluster_3', 'measurement_3+measurement_14', 'measurement_16+Cluster_1', 'measurement_10+measurement_15', 'attribute_2+measurement_15', 'measurement_0+measurement_6', 'measurement_3+measurement_17', 'measurement_1+Cluster_0', 'attribute_3+measurement_1', 'measurement_13+measurement_15', 'measurement_8+measurement_14', 'measurement_5+Cluster_1', 'loading+Cluster_2', 'measurement_14+Cluster_4', 'measurement_13+Cluster_4', 'measurement_0+Cluster_3', 'Cluster_2+Cluster_4', 'measurement_3+attribute_1_material_6', 'attribute_0_material_5+Cluster_4', 'measurement_10+attribute_0_material_7', 'measurement_3+measurement_5', 'measurement_10+Cluster_2', 'measurement_5+attribute_1_material_5', 'measurement_1+measurement_2', 'measurement_13+measurement_17', 'measurement_1+measurement_8', 'measurement_0+Cluster_4', 'measurement_0+measurement_17', 'measurement_15+Cluster_4', 'measurement_10+Cluster_3', 'measurement_10+attribute_0_material_5', 'measurement_15+measurement_17', 'measurement_6+measurement_8', 'measurement_6+Cluster_4', 'measurement_15+Cluster_2', 'measurement_3+measurement_12', 'measurement_8+measurement_12', 'measurement_2+measurement_10', 'measurement_3+measurement_15', 'measurement_0+measurement_2', 'loading+Cluster_4', 'loading+Cluster_0', 'measurement_7+attribute_0_material_5', 'measurement_3+measurement_6', 'attribute_3+Cluster_2', 'measurement_10+Cluster_4', 'measurement_5+measurement_13', 'attribute_3+measurement_8', 'Cluster_0+Cluster_5', 'attribute_2+measurement_3', 'attribute_3+measurement_5', 'measurement_3+measurement_10', 'measurement_5+measurement_12', 'Cluster_1+Cluster_3', 'measurement_2+measurement_14', 'measurement_6+Cluster_0', 'measurement_12+measurement_17', 'measurement_5+attribute_0_material_7'])

#save final processed dataframes as csv files
merged_df_train.to_csv('/home/gunnar/playgroundProject/data/processed_train.csv', index = False)
merged_df_test.to_csv('/home/gunnar/playgroundProject/data/processed_test.csv', index = False)







