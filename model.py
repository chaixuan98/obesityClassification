import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics, svm
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image


data = pd.read_csv(r"C:\Users\chaix\Downloads\ObesityData.csv")
print(data)
le = LabelEncoder()
le.fit(data['Gender'])
data['Gender'] = le.transform(data['Gender'])
le.fit(data['Family'])
data['Family'] = le.transform(data['Family'])
le.fit(data['TDEE'])
data['FAVC'] = le.transform(data['TDEE'])
le.fit(data['Smoker'])
data['Smoker'] = le.transform(data['Smoker'])
le.fit(data['Water'])
data['Water'] = le.transform(data['Water'])
le.fit(data['ActivityLevel'])
data['ActivityLevel'] = le.transform(data['ActivityLevel'])
le.fit(data['Alcohol'])
data['Alcohol'] = le.transform(data['Alcohol'])


# indepedent and dependent columns
attrObese = ["Gender", "Age", "Height", "Weight", "Family", "TDEE", "Smoker", "Water", "ActivityLevel", "Alcohol"]
x = data[attrObese]
y = data['ObeseLevel']

# split in train and test
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=2)

# prepare configuration for cross validation test harness
seed = 7

# prepare models
models = []

models.append(('KNN', KNeighborsClassifier(7)))
models.append(('DT', DecisionTreeClassifier()))
models.append(('SVM', svm.SVC()))

# evaluate each model in turn
results = []
names = []
acc_mean = []
scoring = 'accuracy'
for name, model in models:
	
	kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
	cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	acc_mean.append(cv_results.mean())
	# print(acc_mean)
	high_acc = acc_mean.index(np.max(acc_mean))
	# print(high_acc)
	classify = models[high_acc]
	# print(classify)
	final = classify[1]

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

classifier = final
print("Final choose algorithm : ")
print(classifier)
classifier.fit(x_train, y_train)

# # save the model
file = open("model.pkl", 'wb')
pickle.dump(classifier, file)

	
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()