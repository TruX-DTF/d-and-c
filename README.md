# d-and-c
D&C: A Divide-and-Conquer, IR-based, Multi-Classifer Approach to Bug Localization

Setup:
In order to run the code, python3 is necessary with the additional libraries such as LightGBM.

Extract the archives allResults.db.7z and simiSmall.h5 2.7z.00X which are containing the prediction probabilities of all the classifies and similarity score of the bug report / source code files pairs used in the study.


Code Structure:

train.py
	The training module: Please change the local variable 'xgbResultFolder' which points to the simiSmall.h5 database and save the trained models.

predict.py
	The prediction module: It uses the similarity database (simiSmall.h5) and the trained models to make the predictions. The results of the prediction are saved to be used in the evaluation (eval.py) module

results/
	The folders that is containing the prediction probabilities of the D&C classifiers (Name_of_the_Classifier\_TESTALLEpredProb.pickle)
	confusionMatrix/ The confusion matrices that describe the performance of classification models
	MAP and MRR values computes (files that are finising with MRR.pick)
	allResults.db.7z : The SQL database containing all the prediction probabilities, which is used to combine the classifiers into a single one.

eval.py
	The evaluation module: It uses the predictions produced by the prediction module (predict.py); merge into a result database (allResults.db, or loads if it is already available) and computes the MAP and MRR values of the D&C with different combination strageties (max,min,prod,mean etc..) and finally compare the results with the state-of-art approaches.

File Look-up:
	results-state-of-the-art.xlsx
		The MAP, MRR, Top1, Top5, Top10 results of the state-of-the-art approaches.
	bugFixingCommits.txt
		The file that contains the commit ids of the commits and the bug reports ids of the projects that are used in our experiment.
	filteredBugFixingCommits.txt
		The filtered bug reports, as described in our paper.
	allTeknik.pickle
		The MAP, MRR, Top1, Top5, Top10 results of the state-of-the-art approaches for the whole dataset.
	idFramesTeknikFiltered.pickle
		The MAP, MRR, Top1, Top5, Top10 results of the state-of-the-art approaches, filtered, as described in the paper.
	groundTruthAnswer.pickle
		The ground truth object, indicating which are the actually buggy bug-report / source code pairs.



