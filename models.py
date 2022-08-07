from re import T, X
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('/home/gunnar/playgroundProject/data/processed_train.csv')
test_df = pd.read_csv('/home/gunnar/playgroundProject/data/processed_test.csv')

x_train = train_df.drop(columns=['failure', 'id', 'product_code', 'attribute_1_material_8'])
y_train = train_df['failure']
x_test = test_df.drop(columns=['id', 'product_code', 'attribute_1_material_7'])
test_id_list = test_df['id'].tolist()




# #iterate over all distinct product codes in the training dataframe
# for prod_code in train_df['product_code'].unique().tolist():
#     chunk_df = train_df[train_df['product_code'] == prod_code]
#     x = chunk_df.drop(columns=['failure'])
#     y = chunk_df['failure'].values.ravel()
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)





# #Multilayer Perceptron (MLP) ##############################################################
# mlp_model = MLPClassifier(max_iter=500, activation='relu', solver='adam', alpha=0.1, learning_rate='constant', hidden_layer_sizes=(25,))

# #fit the model on the training data and predict the desired probabilities for the test data
# clf = mlp_model.fit(x_train, y_train.values.ravel())
# y_pred_mlp = clf.predict_proba(x_test)[:,1]

# #write submissions to csv file
# submission_df = pd.DataFrame({'id': test_id_list, 'failure': y_pred_mlp})
# submission_df.to_csv('/home/gunnar/playgroundProject/submissions/submission_mlp.csv', index=False)




# # Random Forest Model #####################################################################
# clf = RandomForestClassifier(max_depth=2).fit(x_train, y_train.values.ravel())
# y_pred_rf = clf.predict_proba(x_test)[:,1]

# #write submissions to csv file
# submission_df = pd.DataFrame({'id': test_id_list, 'failure': y_pred_rf})
# submission_df.to_csv('/home/gunnar/playgroundProject/submissions/submission_rf.csv', index=False)





# # Light Gradient-Boosted Model (LGBM) ######################################################
# clf = lgb.LGBMClassifier().fit(x_train, y_train.values.ravel())
# y_pred_lgbm = clf.predict_proba(x_test)[:,1]

# #write submissions to csv file
# submission_df = pd.DataFrame({'id': test_id_list, 'failure': y_pred_lgbm})
# submission_df.to_csv('/home/gunnar/playgroundProject/submissions/submission_lgbm.csv', index=False)




# Logistic Regression #################################################################
clf = LogisticRegression().fit(x_train, y_train.values.ravel())
y_pred_lr = clf.predict_proba(x_test)[:,1]

#write submissions to csv file
submission_df = pd.DataFrame({'id': test_id_list, 'failure': y_pred_lr})
submission_df.to_csv('/home/gunnar/playgroundProject/submissions/submission_lr.csv', index=False)



