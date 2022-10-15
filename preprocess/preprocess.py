from pandas import read_csv,DataFrame
from sklearn.datasets import load_breast_cancer
# from catboost import CatBoostClassifier
from sklearn import preprocessing,model_selection,metrics
import pathlib,os

data_in_path = 'opt/ml/processing/input/data'
output_dir = "/opt/ml/processing/preprocessed"
pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

def get_data():
    X = read_csv(f"{data_in_path}/train.csv")
    y = X.pop('target')
    return X,y 

def basic_processing(X,y):
    "This will seperate dataset into train and test and scale"
    xtrain,xtest,ytrain,ytest = model_selection.train_test_split(X,y,test_size=0.2,random_state=221014)
    scaler = preprocessing.StandardScaler()
    xtrain_scaled = scaler.fit_transform(xtrain)
    xtest_scaled = scaler.transform(xtest)
    return xtrain_scaled,xtest_scaled,ytrain,ytest

def save_dataset():
    xtrain,xtest,ytrain,ytest = basic_processing(*get_data())
    xtrain = DataFrame(xtrain)
    xtest = DataFrame(xtest)
    xtrain.loc[:,"TARGET"] = ytrain
    xtest.loc[:,'TARGET'] = ytest

    xtrain.to_csv(f"{output_dir}/train.csv",index=False)
    xtest.to_csv(f"{output_dir}/test.csv",index=False)

if __name__ == '__main__':
    save_dataset()