import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd

import warnings
warnings.filterwarnings('ignore')

data =pd.read_csv('mushrooms_data.csv')
data.drop(['veil-type','stalk-root'],axis=1,inplace=True)
from sklearn.preprocessing import LabelEncoder

lab = LabelEncoder()
for col in data.columns:

    data[col] = lab.fit_transform(data[col])


#splitting training and test set
x = data.drop(['class'],axis=1)
#X will act as predictors variable
y = data['class'] # y will act as response variable


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state =42)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
forest= RandomForestClassifier(n_estimators=100)
forest.fit(x_train,y_train)
y_pred_rf = forest.predict(x_test)

# # Saving model to disk
import pickle
pickle.dump(forest, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_pred_rf)





