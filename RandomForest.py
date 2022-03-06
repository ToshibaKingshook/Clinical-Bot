import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def RandomForst_pred(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    data = pd.read_csv("heart.csv")
    
    train = data.drop('target', axis = 1)
    target = data.target
    
    X_train, X_test, y_train, y_test = train_test_split( train, target, test_size = 0.3, random_state = 109 )
    
   
    classifier = RandomForestClassifier(n_estimators = 10)
    classifier.fit(X_train, y_train)
    
    


    NewData = [[38,1,2,138,175,0,1,173,0,0,2,4,2], [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
   
    result = classifier.predict(NewData)[1]
 
    return result
    
