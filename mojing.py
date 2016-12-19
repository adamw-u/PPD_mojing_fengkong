# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:17:13 2016
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from xgboost.sklearn import XGBClassifier
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.constraints import maxnorm
from keras.utils import np_utils
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense

np.random.seed(11)
need_normalise=True
need_validataion=True
need_categorical=False
save_categorical_file=False
#nb_epoch=180

def save2model(submission,file_name,y_pre):
    assert len(y_pre)==len(submission)
    submission['score']=y_pre
    submission.to_csv(file_name,index=False)
    print ("saved files %s" % file_name)

def load_data():
    
    path = 'D:/JDO/PPD-Second-Round-Data/'
    df_testM = pd.read_csv(path+'chusai_test/Kesci_Master_9w_gbk_2.csv')
    df_testL = pd.read_csv(path+'chusai_test/LogInfo_9w_2.csv')
    df_testU = pd.read_csv(path+'chusai_test/Userupdate_Info_9w_2.csv')

    train1_M = pd.read_csv(path+'chusai_train/PPD_Training_Master_GBK_3_1_Training_Set.csv')
    train1_L = pd.read_csv(path+'chusai_train/PPD_LogInfo_3_1_Training_Set.csv')
    train1_U = pd.read_csv(path+'chusai_train/PPD_Userupdate_Info_3_1_Training_Set.csv')
    train2_M = pd.read_csv(path+'fusai_train/Kesci_Master_9w_gbk_3_2.csv')
    train2_L = pd.read_csv(path+'fusai_train/LogInfo_9w_3_2.csv')
    train2_U = pd.read_csv(path+'fusai_train/Userupdate_Info_9w_3_2.csv')

    df_trainM = pd.concat([train1_M,train2_M],ignore_index=True)
    df_trainL = pd.concat([train1_L,train2_L],ignore_index=True)
    df_trainU = pd.concat([train1_U,train2_U],ignore_index=True)   
    
    df_testM['UserInfo_2'] = df_testM['UserInfo_2'].apply(lambda x:str(x)[:4])
    df_testM['UserInfo_4'] = df_testM['UserInfo_4'].apply(lambda x:str(x)[:4])
    df_testM['UserInfo_8'] = df_testM['UserInfo_8'].apply(lambda x:str(x)[:4])
    df_testM['UserInfo_7'] = df_testM['UserInfo_7'].apply(lambda x:str(x)[:4])
    df_testM['UserInfo_20'] = df_testM['UserInfo_20'].apply(lambda x:str(x)[:4])
    df_testM['UserInfo_19'] = df_testM['UserInfo_19'].apply(lambda x:str(x)[:4])
    df_trainM['UserInfo_2'] = df_trainM['UserInfo_2'].apply(lambda x:str(x)[:4])
    df_trainM['UserInfo_4'] = df_trainM['UserInfo_4'].apply(lambda x:str(x)[:4])
    df_trainM['UserInfo_8'] = df_trainM['UserInfo_8'].apply(lambda x:str(x)[:4])
    df_trainM['UserInfo_7'] = df_trainM['UserInfo_7'].apply(lambda x:str(x)[:4])
    df_trainM['UserInfo_20'] = df_trainM['UserInfo_20'].apply(lambda x:str(x)[:4])
    df_trainM['UserInfo_19'] = df_trainM['UserInfo_19'].apply(lambda x:str(x)[:4])
    
    df_trainM['UserInfo_24'] = df_trainM['UserInfo_24'].apply(lambda x:str(x)[:10])
    df_testM['UserInfo_24'] = df_testM['UserInfo_24'].apply(lambda x:str(x)[:10])

    df_trainM = df_trainM.replace(u'不详',np.nan)
    df_testM  = df_testM.replace(u'不详',np.nan)
    df_testM['Date'] = pd.to_datetime(pd.Series(df_testM['ListingInfo']))
    df_testM = df_testM.drop('ListingInfo', axis=1)
    df_testM['Year'] = df_testM['Date'].apply(lambda x: int(str(x)[:4]))
    df_testM['Month'] = df_testM['Date'].apply(lambda x: int(str(x)[5:7]))
    df_testM['weekday'] = [df_testM['Date'][i].dayofweek for i in range(len(df_testM['Date']))]
    
    df_trainM['Date'] = pd.to_datetime(pd.Series(df_trainM['ListingInfo']))
    df_trainM = df_trainM.drop('ListingInfo', axis=1)
    df_trainM['Year'] = df_trainM['Date'].apply(lambda x: int(str(x)[:4]))
    df_trainM['Month'] = df_trainM['Date'].apply(lambda x: int(str(x)[5:7]))
    df_trainM['weekday'] = [df_trainM['Date'][i].dayofweek for i in range(len(df_trainM['Date']))]
    frame1 = [df_testL,df_trainL]
    frame2 = [df_testU,df_trainU]
    df_L = pd.concat(frame1,ignore_index=True)    
    df_U = pd.concat(frame2,ignore_index=True)
    df_U['UserupdateInfo1'] = df_U['UserupdateInfo1'].apply(lambda x:str(x).upper())
    df_Uu = pd.get_dummies(df_U['UserupdateInfo1']).join(df_U['Idx'])
    df_L1 = pd.get_dummies(df_L['LogInfo1']).join(df_L['Idx'])
    df_L1 = df_L1.groupby('Idx',as_index=False).sum()
    df_L2 = pd.get_dummies(df_L['LogInfo2']).join(df_L['Idx'])
    df_L2 = df_L2.groupby('Idx',as_index=False).sum()
    df_L3 = pd.merge(df_L1,df_L2,on = 'Idx',how = 'left')
    result_L = df_L3.groupby('Idx',as_index=False).sum()
    result_U = df_Uu.groupby('Idx',as_index=False).sum()
    
    df_trainM = df_trainM.fillna(-1)
    df_testM = df_testM.fillna(-1)
    for f in df_testM.columns:
        if df_testM[f].dtype=='object':
            lbl = LabelEncoder()
            lbl.fit(list(df_testM[f])+list(df_trainM[f]))
            df_trainM[f] = lbl.transform(list(df_trainM[f].values))
            df_testM[f] = lbl.transform(list(df_testM[f].values))
    df_train = pd.merge(df_trainM,result_L,on = 'Idx')
    train = pd.merge(df_train,result_U,on = 'Idx')
    df_test = pd.merge(df_testM,result_L,on = 'Idx',how = 'left')
    test = pd.merge(df_test,result_U,on = 'Idx',how = 'left')
    
    train = train.fillna(-1)
    test = test.fillna(-1)
    submission=pd.DataFrame()
    submission["Idx"]= test["Idx"] 
    
    drop_feature = ['WeblogInfo_10']
    X = train.drop(['Date','target','Idx'],axis = 1)
    X = X.drop(drop_feature,axis = 1)
    #X = train[golden_feature]
    Y = train['target'] 
    final = test['target']
    TEST = test.drop(['Date','Idx','target'],axis = 1)
    TEST = TEST.drop(drop_feature,axis = 1)
    #TEST = TEST[golden_feature]
    return [X,Y,TEST,submission,final]

def cross_validation():
    datasets=load_data()
    x_train,x_test,y_train,y_test = train_test_split(datasets[0],datasets[1],test_size=13000)

    encoder = LabelEncoder()
    train_y,valid_y = y_train.values,y_test.values
    train_y,valid_y = encoder.fit_transform(train_y).astype(np.int32),encoder.fit_transform(valid_y).astype(np.int32)
    train_y,valid_y = np_utils.to_categorical(train_y),np_utils.to_categorical(valid_y)

    print ("processsing finished")
    valid=None
    x = np.array(datasets[0])
    x = x.astype(np.float32)
    train,valid = np.array(x_train),np.array(x_test)
    train,valid = train.astype(np.float32),valid.astype(np.float32)
    test=np.array(datasets[2])
    test=test.astype(np.float32)
    if need_normalise:
        scaler = StandardScaler().fit(x)
        train,valid = scaler.transform(train),scaler.transform(valid)
        test = scaler.transform(test)
    
    return [(train,train_y),(test,datasets[3]),(valid,valid_y),
            (x_train,y_train),(datasets[2],datasets[3]),(x_test,y_test)]

print('Loading data...')

datasets=cross_validation()

X_train, y_train = datasets[0]
X_test, submission = datasets[1]
X_valid, y_valid = datasets[2]

X_Train, y_Train = datasets[3]
X_Test, submission = datasets[4]
X_Valid, y_Valid = datasets[5]

nb_classes = y_train.shape[1]
print(nb_classes, 'classes')

dims = X_train.shape[1]
print(dims, 'dims')

model = Sequential()

model.add(Dense(1024,input_shape=(dims,), init = 'glorot_normal', W_constraint = maxnorm(4)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(360, init = 'glorot_normal', W_constraint = maxnorm(4)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
'''
model.add(Dense(420, init = 'glorot_normal', W_constraint = maxnorm(4)))
model.add(PReLU())
model.add(BatchNormalization())
model.add(Dropout(0.5))
'''
model.add(Dense(nb_classes))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer="sgd")


model.fit(X_train, y_train, nb_epoch=100, batch_size=128,
          callbacks = [EarlyStopping(monitor='val_loss', patience=20)])
y_pre = model.predict_proba(X_valid)
scores = roc_auc_score(y_valid,y_pre)
model_predprob = model.predict_proba(X_valid)[:,1]
print ("\nknnModel Report")
print ("AUC Score (Test): %f" %scores)

gbdt = GradientBoostingClassifier(n_estimators=400, learning_rate=0.03,
    max_depth=9,).fit(X_Train, y_Train)
gbdt_predprob = gbdt.predict_proba(X_Valid)[:,1]

print "\ngbdtModel Report"
print "AUC Score (test): %f" % roc_auc_score(y_Valid, gbdt_predprob)

lr = LogisticRegression()
lr.fit(X_Train, y_Train)
lr_predprob = lr.predict_proba(X_Valid)[:,1]

print "\nlrModel Report"
print "AUC Score (test): %f" % roc_auc_score(y_Valid, lr_predprob)

xgb1 = XGBClassifier(
 learning_rate =0.03,
 n_estimators=408,
 max_depth=9,
 min_child_weight=3,
 subsample=0.75,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,)
xgb1.fit(X_Train, y_Train,eval_metric='auc')
    
dtrain_predictions = xgb1.predict(X_Valid)
dtrain_predprob = xgb1.predict_proba(X_Valid)[:,1]
    
print "\nxgbModel Report"
#print "Accuracy : %.4g" % accuracy_score(y_Valid, dtrain_predictions)
print "AUC Score: %f" % roc_auc_score(y_Valid, dtrain_predprob)

AUC = []
for i in range(0,101,1):
    a = round(float(i)/100,2)*dtrain_predprob+(1-round(float(i)/100,2))*model_predprob
    auc = roc_auc_score(y_Valid,a)
    AUC.append(auc)
i = AUC.index(max(AUC))
print "\nensembleModel Report"
print i,"AUC Score: %f" % max(AUC)

test = load_data()[4].values[0:10000]

yprelr = lr.predict_proba(X_Test).head(10000)[:,1]
ypregbdt = gbdt.predict_proba(X_Test).head(10000)[:,1]
y_pre1 = model.predict_proba(X_test)[0:10000][:,1]
y_pre2 = xgb1.predict_proba(X_Test).head(10000)[:,1]
y_pre = round(float(i)/100,2)*y_pre2 + (1-round(float(i)/100,2))*y_pre1

print "\ntest Report"
print "lr AUC Score: %f" % roc_auc_score(test,yprelr)
print "gbdt AUC Score: %f" % roc_auc_score(test,ypregbdt)
print "knn AUC Score: %f" % roc_auc_score(test,y_pre1)
print "xgb AUC Score: %f" % roc_auc_score(test,y_pre2)
print "ensemble AUC Score: %f" % roc_auc_score(test,y_pre)
#print roc_auc_score(y_test,y_pre)
save2model(submission, 'keras_nn_test111.csv',y_pre)