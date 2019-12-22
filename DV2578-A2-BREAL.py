# -*- coding: utf-8 -*-
# author : Adrien Bréal
# created on the 03/12/2019
# Version 1.0
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score,f1_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import friedmanchisquare
from time import time as ti
from scikit_posthocs import posthoc_nemenyi

#Useful constants
ALGORITHM_NAMES = ["LR","RF","SVM"]
EVAL_MEASURE = ["TRAINING TIME","ACCURACY","F-MEASURE"]
comparatorList = []
scoring = {"accuracy":make_scorer(accuracy_score),
			   "f1_score":make_scorer(f1_score)}


@ignore_warnings(category=ConvergenceWarning)
def main():

	#load data and separate features and label
	df = pd.read_csv("spambase_data.data",header=None)
	Y = df.iloc[:,57]
	X = df.iloc[:,:57]

	print("Computing results...\n")
	start = ti()

	#For each model, run stratified 10-fold cross validation tests
	#Logistic Regression model
	logRegModel = LogisticRegression(multi_class="ovr",solver="lbfgs")
	resultForLR = cross_validate(logRegModel,X=X,y=Y,cv=10,scoring=scoring)
	resultForLR = pd.DataFrame(resultForLR)


	#Random Forest model
	rndForModel = RandomForestClassifier(n_estimators=100)
	resultForRF = cross_validate(rndForModel,X=X,y=Y,cv=10,scoring=scoring)
	resultForRF = pd.DataFrame(resultForRF)


	#Support Vector Machine model
	svmModel = LinearSVC(dual=False)
	resultForSVM = cross_validate(svmModel,X=X,y=Y,cv=10,scoring=scoring)
	resultForSVM = pd.DataFrame(resultForSVM)

	#Create tables for comparisons
	timeComparator = pd.concat([resultForLR.iloc[0:,0],
							 resultForRF.iloc[0:,0],
							 resultForSVM.iloc[0:,0]],axis=1)



	accuracyComparator = pd.concat([resultForLR.iloc[0:,2],
								 resultForRF.iloc[0:,2],
								 resultForSVM.iloc[0:,2]],axis=1)


	f1Comparator = pd.concat([resultForLR.iloc[0:,3],
						   resultForRF.iloc[0:,3],
						   resultForSVM.iloc[0:,3]],axis=1)


	#Add them to the comparator list
	comparatorList.append(timeComparator)
	comparatorList.append(accuracyComparator)
	comparatorList.append(f1Comparator)

	end = ti()

	#Compare and display tables
	computedFrames = compareMeasures(comparatorList)
	print("\nTotal time : {0:.2f}s\n".format(end-start))

	#Perform ranking
	rankingFrames = rankMeasures(comparatorList)

	#Display results
	displayResults(computedFrames,rankingFrames)

	#Perform the Friedman test
	friedmanTest(rankingFrames)

	#Save comparators for later usage
	for i in range(len(comparatorList)):
		comparatorList[i].to_pickle("pickefile.pkl")


#This function create the pandas DataFrame containing results for specific
#evaluation measure
def compareMeasures(comparatorList):

	computedFrames=[]

	for elt in comparatorList:

		elt.columns=ALGORITHM_NAMES
		elt.rename_axis("Fold",inplace=True)

		eltMean = pd.DataFrame(elt.mean()).transpose()
		eltMean.rename(index={0:"AVG"},inplace=True)
		elt = pd.concat([elt,eltMean])

		eltStdDev = pd.DataFrame(elt.std()).transpose()
		eltStdDev.rename(index={0:"STD"},inplace=True)
		elt = pd.concat([elt,eltStdDev])

		computedFrames.append(elt)

	return computedFrames
#This function create the pandas Dataframe containing ranking for specific
#evaluation measure
def rankMeasures(comparatorList):

	rankingComparator = []

	for i in range(len(comparatorList)):

		#rank time in reverse mode
		if i==0:
			elt = comparatorList[i].rank(axis=1,method="first")
		else:
			elt = comparatorList[i].rank(axis=1,ascending=False,method="first")

		eltMean = pd.DataFrame(elt.mean()).transpose()
		eltMean.rename(index={0:"AVG RANK"},inplace=True)
		elt = pd.concat([elt,eltMean])
		rankingComparator.append(elt)

	return rankingComparator

#This function display all the results from comparison and ranking
def displayResults(computedFrames,rankingFrames):
	for i in range(len(computedFrames)):

		print("---RESULTS AND RANKING FOR {}---".format(EVAL_MEASURE[i]))
		print(computedFrames[i],"\n",rankingFrames[i],"\n")

#This function perform the friedman test on each evaluation measure, then
#perform the Nemenyi test if necessary
def friedmanTest(measure):

	alpha = 0.05

	for i,elt in enumerate(measure):

		stats, p = friedmanchisquare(elt.iloc[0:10,0],
							   elt.iloc[0:10,1],elt.iloc[0:10,2])

		print("\n---FRIEDMAN RESULTS FOR {}---".format(EVAL_MEASURE[i]))
		print("Statistics : {0:.3f}, p-value : {1:.3f}".format(stats,p))

		if p > alpha:
			print("Fail to reject null hypothesis, all algorithms perform equally")

	#if all algorithms do not perform equally, then perform the
	#nemenyi test
		else :
			print("Reject null hypothesis,algorithms perform differently")

			print("---Nemenyi test---")

			x = [list(elt.iloc[0:10,0]),list(elt.iloc[0:10,1]),
												list(elt.iloc[0:10,2])]
			nemeyi = posthoc_nemenyi(x)
			print(nemeyi)


if __name__ == "__main__":
	main()
	input("Press any key to continue")
