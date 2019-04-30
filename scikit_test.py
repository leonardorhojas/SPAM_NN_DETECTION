#import csv
#with open('ejemplo_testing.csv', 'r') as f:
#  reader = csv.reader(f)
#  your_list = list(reader)
#print your_list
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#import seaborn as sns


def train():
	df = pd.read_csv('ejemplo_testing.csv', delimiter=',')
	y = df['SPAM']
	x = df.drop(['SPAM'], axis=1)
	print(y)
	print(x)
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.25, random_state=5)
	clf = MLPClassifier(hidden_layer_sizes=(100,100,100,100), max_iter=5000, alpha=0.0001,solver='sgd',verbose=10,random_state=20,tol=0.000000001)

 	#print(x_train)
	#print(x_test)
	#clf.fit(x_train, y_train)                         
	clf.fit(x_train, y_train)
	y_pred = clf.predict(x_test)
	a=accuracy_score(y_test, y_pred)
	print(a)
	tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
	(tn, fp, fn, tp)
	tn=tn/(len(y_pred))
	print("TN: " + str(tn))
	fp=fp/len(y_pred) 
	print("FP: " + str(fp))
	fn=fn/len(y_pred)
	print("FN: " + str(fn))
	tp=tp/len(y_pred)
	print("TP: " + str(tp))

if __name__ == "__main__":
    train()