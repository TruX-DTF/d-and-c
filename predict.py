
import logging
import sys
from os.path import isfile, join
from os import listdir
import pandas as pd
import gzip
import pickle as p
import random
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import time
import numpy as np
from multiprocessing.pool import Pool
from pandas import HDFStore
import shutil
import os
import learning as l

import lightgbm as light


xgbResultFolder ='/Volumes/Lexar/xgbResult3'
database = 'simiSmall.h5'

thread = 8
core = 1



def bigDF():

    logging.info('reading done %s' ,database)
    with HDFStore(database, mode='r') as store:
        res = store.select_column('simi','bugID')
    logging.info(len(res))
    logging.info(len(res.unique()))
    idList= res.unique().tolist()


    return idList



def prepareDataset(listOfSimiFiles, aType, randomSample =False ):
    logging.info('Prepare testing dataset ' + aType)

    idList = [f.replace('simi.gzip', '') for f in listOfSimiFiles ]

    with HDFStore(database, mode='r') as store:
        df = store.select('simi',where='bugID in idList')


    if randomSample:
        simiT = df[df.answer == True]
        simiF = df[df.answer == False]
        simiFalseRandom =simiF.groupby('bugID').apply(lambda x: x.sample(3))
        df = pd.concat([simiT, simiFalseRandom])

    return df

def convert2TrainingData(trainFrame,light):
    logging.info('Convert to training data')
    if light:
        aDf = trainFrame
    else:
        aDf = trainFrame.fillna(-1)

    features = [c for c in aDf.columns if len(c.split('2')) == 2]
    features = [c for c in features if not c.endswith('Rank') and not c.startswith('comment')]
    features = ['bugID', 'file'] + features
    features = aDf[features].columns


    logging.info('Ground truth %s', str(aDf['answer'].value_counts()))

    for f in [features[2:]]:
        aDf[f] = aDf[f].astype(float)

    X = aDf[features].values
    y, _ = pd.factorize(aDf['answer'])
    # groups = groups.values
    return X,y#,groups



def corePredict(aTuple, test = False):

    try:
        X_train, X_test,aType, iTer,allNT1,testType = aTuple
        aType = aType + str(iTer)


        from collections import Counter


        excludeTrain = set(allNT1) - set(X_train.tolist())
        X_test = list(excludeTrain)
        logging.info(len(X_train), len(X_test))
        testNotAll = prepareDataset(X_test, aType)
        Xt, yt = convert2TrainingData(testNotAll,True)
        dataTest = Xt[:, 2:]
        labelTest = yt

        logging.info(len(labelTest))


        logging.info('Load model to predict')
        model = light.Booster(model_file=xgbResultFolder + '/' + testType + "/" + aType+ testType +'.model')
        logging.info('Start predicting')
        pred_prob = model.predict(dataTest, num_iteration=model.best_iteration)



        bugIds = Xt[:, 0]
        files = Xt[:, 1]

        series = [bugIds, files, pred_prob, labelTest]
        df = pd.DataFrame(series)
        df = df.T

        df.columns = ['bugId', 'file', 'pred_prob_1', 'answer']


        df['pred_prob_1'] = pd.to_numeric(df['pred_prob_1'], errors='coerce')
        df['rank_1'] = df.groupby('bugId')['pred_prob_1'].rank(method='first', ascending=0, na_option='keep')

        answer = df[df.answer == True]
        answer['AnsOrder'] = answer.groupby('bugId')['rank_1'].rank(method='first', ascending=1, na_option='keep')
        answer['AP'] = answer.apply(lambda x: (x['AnsOrder']) / (x['rank_1']), axis=1)
        answer['RR'] = answer.apply(lambda x: l.RR_XGB(x, 'rank_1'), axis=1)

        answer['Classifier'] = aType + testType
        df['Classifier'] = aType + testType


        dfSmaller = df.sort_values('rank_1').groupby('bugId').head(20).sort_values('bugId')
        p.dump(dfSmaller, open(xgbResultFolder + '/' + testType + "/" + aType + testType + "predProb.pickle", "wb"))
        printResults(pred_prob, labelTest, aType, testType)



    except Exception as e:
        logging.error(e)
        raise e


def printResults(y_pred, y_test, aType, predType):
    logging.info(82 * '_')
    predictions = [round(value) for value in y_pred]#[:,1]]
    # evaluate predictions
    line0 = '# Predictions(90%-10%): ' + str(len(predictions))
    accuracy = accuracy_score(y_test, predictions)
    line1 = ("Accuracy: %.2f%%" % (accuracy * 100.0))
    line2 = (pd.crosstab(index=y_test, columns=np.asarray(predictions), rownames=['actual'], colnames=['predicted']))
    line3 = (metrics.classification_report(predictions, y_test))

    with open(xgbResultFolder+ '/' +predType +'/' +aType + predType,'a') as f:
        f.write('\n')
        f.write(str(82 * '_'))
        f.write('\n')
        f.write(line0)
        f.write('\n')
        f.write(str(line1))
        f.write('\n')
        f.write(str(line2))
        f.write('\n')
        f.write(str(line3))


def runTrainTopTestAll(aType):
    logging.info('Running ' + aType)

    if aType == 'ALL':
        aND = np.asarray(allTop)
    elif aType == 'INTER':
        aND = np.asarray(interTop)
    elif aType == 'LOCUS':
        aND = np.asarray(locusTop)
    elif aType == 'BLIA':
        aND = np.asarray(bliaTop)
    elif aType == 'BugLocator':
        aND = np.asarray(bugLTop)
    elif aType == 'BRTracer':
        aND = np.asarray(brtTop)
    elif aType == 'AMALGAM':
        aND = np.asarray(amalTop)
    elif aType == 'BLUiR':
        aND = np.asarray(bluirTop)

    elif aType == 'NALL':
        aND = np.asarray(allNTop)


    from sklearn.model_selection import KFold
    if aType == 'AMALGAM':
        logging.info(len(aND))
        kf = KFold(n_splits=len(aND), shuffle=True)

    else:
        kf = KFold(n_splits=10,shuffle=True)  # Define the split - into 2 folds
    if aType != 'BLUiR':
        iTer = 0
        tuples = []
        for train_index, test_index in kf.split(aND):
            X_train, X_test = aND[train_index], aND[test_index]
            aTuple = X_train, X_test, aType, iTer,simiFiles,'_TESTALL'
            tuples.append(aTuple)
            iTer += 1

        pool = Pool(core,maxtasksperchild=1)

        pool.map_async(corePredict, [link for link in tuples], chunksize=1)

        pool.close()
        pool.join()
    else:
        aTuple = aND, None, aType, 0, simiFiles, '_TESTALL'
        corePredict(aTuple, True)

def setLogg():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(funcName)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)


def getRun():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-run', dest='run', help='which part')
    parser.add_argument('-c', dest='core', help='core number')

    parser.add_argument('-t', dest='thread', help='thread number')
    parser.add_argument('-f', dest='folder', help='folder')
    parser.add_argument('-d', dest='database', help='database')


    args = parser.parse_args()

    if args.run is None:
        parser.print_help()
        return None
    return args

if __name__ == '__main__':

    os.system("mkdir %s" % xgbResultFolder)

    os.system("mkdir %s" % xgbResultFolder + '/_TESTALL/')

    simiFiles = bigDF()

    idFrames = pd.read_pickle('idFramesTeknikFiltered.pickle')
    topN = idFrames[idFrames.TopN == 1]

    allT1Ids = topN[topN.columns[1]].values.tolist()[0] | topN[topN.columns[2]].values.tolist()[0] | \
               topN[topN.columns[3]].values.tolist()[0] | topN[topN.columns[4]].values.tolist()[0] | \
               topN[topN.columns[5]].values.tolist()[0] | topN[topN.columns[6]].values.tolist()[0]

    allTop = [f for f in simiFiles if f.replace('simi.gzip', '') in allT1Ids]
    interTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['intersection'].values.tolist()[0]]
    locusTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['OnlyLocus'].values.tolist()[0]]
    bliaTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['OnlyBLIA'].values.tolist()[0]]
    bugLTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['OnlyBugLocator'].values.tolist()[0]]
    brtTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['OnlyBRTracer'].values.tolist()[0]]
    amalTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['OnlyAmaLgam'].values.tolist()[0]]
    bluirTop = [f for f in simiFiles if f.replace('simi.gzip', '') in topN['OnlyBLUiR'].values.tolist()[0]]


    logging.info('Len of all ,inter, locus, blia, bugL, bbrt, amalgam, bluir, allN')

    allNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in allT1Ids]
    logging.info(len(allTop), len(interTop), len(locusTop), len(bliaTop), len(bugLTop), len(brtTop), len(amalTop),
          len(bluirTop), len(allNTop))


    runTrainTopTestAll('INTER')

    runTrainTopTestAll('BRTracer')

    runTrainTopTestAll('BugLocator')

    runTrainTopTestAll('BLIA')

    runTrainTopTestAll('LOCUS')

    runTrainTopTestAll('NALL')

    runTrainTopTestAll('ALL')

    runTrainTopTestAll('AMALGAM')

    runTrainTopTestAll('BLUiR')






