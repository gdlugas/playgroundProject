# import libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# import training data
test_df = pd.read_csv('/home/gunnar/playgroundProject/data/test.csv')




# One-hot encoding of categorical features attribute_0 and attribute_1 ##################################
#first get dummy dataframes
a1 = pd.get_dummies(test_df['attribute_0'], prefix='attribute_0')
a2 = pd.get_dummies(test_df['attribute_1'], prefix='attribute_1')

#create dataframe of one-hot encoded features; note that this dataframe does not include product_code
categorical_df = pd.concat([a1, a2], axis=1, join='inner')




#  Standardize ########################
non_standardized_cols = ['id', 'product_code', 'attribute_0', 'attribute_1']
scaler = StandardScaler().fit(test_df.drop(columns=non_standardized_cols)).transform(test_df.drop(columns=non_standardized_cols))
numerical_df = pd.DataFrame(scaler, columns=test_df.drop(columns=non_standardized_cols).columns)

#merge categorical and numerical dataframes
#merged_df = pd.concat([numerical_df, categorical_df, test_df['id'],  test_df['product_code'],  test_df['failure']], axis=1, join='inner')
#merged_df = pd.concat([numerical_df, categorical_df], axis=1, join='inner')


merged_df = pd.concat([test_df['id'], test_df['product_code'], numerical_df, categorical_df], axis = 1, join='inner')



# Impute Numerical features ##################
#in the training data, there are five distinct product_codes: A,B,C,D,E
#we will impute nan values wrt each chunk of test_df corresponding to a given product code

#define empty dataframe which we will append each imputed chunk_df to
processed_df = pd.DataFrame()

#iterate over all distinct product codes in the training dataframe
for prod_code in merged_df['product_code'].unique().tolist():
    chunk_df = merged_df[merged_df['product_code'] == prod_code]

    #create an object for KNNImputer
    imputer = KNNImputer(n_neighbors=2)

    #fill nan entries in merged_df with the mean of the feature variables in the n_neighbors closest to the row of the given nan entry wrt euclidean distance
    temp_imputed_df = pd.DataFrame(imputer.fit_transform(chunk_df.drop(columns=['product_code', 'id'])), columns=chunk_df.drop(columns=['product_code', 'id']).columns)
    temp_imputed_df['product_code'] = [prod_code] * len(chunk_df)
    temp_imputed_df['id'] = chunk_df['id'].tolist()
    processed_df = processed_df.append(temp_imputed_df)

processed_df.to_csv('/home/gunnar/playgroundProject/data/processed_test.csv', index = False)



# #Multilayer Perceptron (MLP) ##############################################################
# mlp_model = MLPClassifier(max_iter=300, activation='relu', solver='adam', alpha=0.1, learning_rate='constant', hidden_layer_sizes=(300,))

# #fit the model on the training data and predict the desired probabilities for the test data
# y_pred = mlp_model.fit(x_train, y_train.values.ravel()).predict_proba(x_test)[:,1]

# unique_customer_id_list = test_df['customer_ID'].tolist()
# submission_df = pd.DataFrame({'customer_ID':unique_customer_id_list, 'prediction':y_pred})
# submission_df.to_csv('/wsu/home/fy/fy73/fy7392/amex/data/submission.csv', index=False)








