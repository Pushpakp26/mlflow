import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
 #🐍🐍
import dagshub
dagshub.init(repo_owner='Pushpakp26', repo_name='mlflow', mlflow=True)


mlflow.set_tracking_uri("https://dagshub.com/Pushpakp26/mlflow.mlflow")

wine=load_wine()
x=wine.data
y=wine.target

xtrain,xtext,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=42)

max_depth=1
n_estimators=5

mlflow.set_experiment('mlflow-exp1')

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(xtrain,ytrain)

    ypred=rf.predict(xtext)
    accuracy=accuracy_score(ytest,ypred)

    #🐍🐍🐍
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    #create consusion matrix
    cm=confusion_matrix(ytest,ypred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Market')

    # save plot
    plt.savefig("Confusion_matrix.png")

    mlflow.log_artifact("Confusion_matrix.png")
    mlflow.log_artifact(__file__)

    mlflow.set_tags({"Author":'Pushpak',"Project":"wine classification"})

    #log the model
    mlflow.sklearn.log_model(rf,"random forest model")

    print(accuracy)