#!/bin/python


def train_classifier(X, y):
	"""Train a classifier using the given training data.

	Trains logistic regression on the input data with default parameters.
	"""
	from sklearn.linear_model import LogisticRegression
	#train Logistic Regression model
	cls = LogisticRegression(random_state=0, solver='lbfgs', max_iter=10000, C=100)
	cls.fit(X, y)
	return cls

def evaluate(X, yt, cls, name='data', willPrint=True):
	"""Evaluated a classifier on the given labeled data using accuracy."""
	from sklearn import metrics
	#Predict document label using Logistic Regression model
	yp = cls.predict(X)
	#Evaulate accuracy of model's prediction
	acc = metrics.accuracy_score(yt, yp)
	if willPrint:
		print("  Accuracy on %s  is: %s" % (name, acc))
	return acc
