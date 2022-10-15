from pandas import read_csv
from sklearn.datasets import load_breast_cancer
from catboost import CatBoostClassifier
from sklearn import preprocessing,model_selection,metrics

data_in_path = '/opt/ml/input/data/train'
model_output_path = '/opt/ml/model'

def train_get_data():
    xtrain = read_csv(f"{data_in_path}/train.csv").dropna()
    xtest = read_csv(f"{data_in_path}/test.csv").dropna()
    
    print(xtrain)
    print(xtrain.columns)
    
    ytrain = xtrain.pop("TARGET")
    ytest = xtest.pop("TARGET")

    return xtrain,xtest,ytrain,ytest

def train():
    xtrain,xtest,ytrain,ytest = train_get_data()
    model = CatBoostClassifier(n_estimators=10,eval_metric='Precision')
    model.fit(xtrain,ytrain,eval_set=[(xtest,ytest)])
    return model

def train_save_model(model):
    model.save_model(f"{model_output_path}/local.model")
    
def main():
    model = train()
    train_save_model(model)

if __name__ =='__main__':
    main()