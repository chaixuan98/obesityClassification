import pandas as pd
import numpy as np
import pickle
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from sklearn.metrics import average_precision_score, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import seaborn as sns


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
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.25, random_state=8675309)
	

# # prepare configuration for cross validation test harness
# seed = 7

# # prepare models
# models = []

# models.append(('KNN', KNeighborsClassifier(7)))
# models.append(('DT', DecisionTreeClassifier()))
# models.append(('SVM', svm.SVC()))

# # evaluate each model in turn
# results = []
# names = []
# acc_mean = []
# scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
# # scoring = ['accuracy', 'precision', 'recall', 'f1']

# # scoring = {'accuracy' : make_scorer(accuracy_score), 
# #            'precision' : make_scorer(precision_score),
# #            'recall' : make_scorer(recall_score), 
# #            'f1_score' : make_scorer(f1_score)}


# for name, model in models:
	
# 	kfold = model_selection.KFold(n_splits=10,shuffle=True, random_state=seed)
# 	cv_results = model_selection.cross_validate (model, x, y, cv=kfold, scoring=scoring)

# 	results.append(cv_results)
# 	names.append(name)
# 	#acc_mean.append(cv_results.mean())
# 	# print(acc_mean)
# 	# high_acc = acc_mean.index(np.max(acc_mean))
# 	# print(high_acc)
# 	# classify = models[high_acc]
# 	# print(classify)
# 	# final = classify[1]




# 	# msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
# 	# print(msg)


# # classifier = final
# # print("Final choose algorithm : ")
# # print(classifier)
# # classifier.fit(x_train, y_train)

# # # save the model
# file = open("model.pkl", 'wb')
# pickle.dump(classifier, file)

	
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.show()

# def run_exps(x_train: pd.DataFrame , y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
dfs = []
models = [('KNN', KNeighborsClassifier()),('SVM',SVC()), ('DT', DecisionTreeClassifier())]
results = []
acc_mean = []
names = []
scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
target_names = ['Insufficient_Weight', 'Normal_Weight','Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
for name, model in models:
			kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=90210)
			cv_results = model_selection.cross_validate(model, x_train, y_train, cv=kfold, scoring=scoring)
			clf = model.fit(x_train, y_train)
			y_pred = clf.predict(x_test)
			print(name)
			print(classification_report(y_test, y_pred, target_names=target_names))
			
			results.append(cv_results)
			names.append(name)
			this_df = pd.DataFrame(cv_results)
			this_df['model'] = name
			dfs.append(this_df)
			final = pd.concat(dfs, ignore_index=True)
			# print(final)
			# high = np.max(final.test_accuracy)
			# print(high)
			# # print(high.model)
			# finalClassifier = clf

			bootstraps = []
			for model in list(set(final.model.values)):
				model_df = final.loc[final.model == model]
				bootstrap = model_df.sample(n=30, replace=True)
				bootstraps.append(bootstrap)
				bootstrap_df = pd.concat(bootstraps, ignore_index=True)
				results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='values')
				time_metrics = ['fit_time','score_time'] # fit time metrics
				## PERFORMANCE METRICS
				results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
				results_long_nofit = results_long_nofit.sort_values(by='values')
				## TIME METRICS
				results_long_fit = results_long.loc[results_long['metrics'].isin(time_metrics)] # df with fit data
				results_long_fit = results_long_fit.sort_values(by='values')
				
				plt.figure(figsize=(55, 25))
				sns.set(font_scale=2.0)
				g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
				plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
				plt.title('Comparison of Model by Classification Metric')
				plt.savefig('./benchmark_models_performance.png',dpi=300)
				#plt.show()



classifier = DecisionTreeClassifier()
print("Final choose algorithm : ")
print(classifier)
classifier.fit(x_train.values, y_train)

# # # save the model
file = open("model.pkl", 'wb')
pickle.dump(classifier, file)

