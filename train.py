
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
#simiFolder = 'simiNew'
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

    return X,y#,groups

def coreLearn(aTuple, test = False):
    #core
    try:
        X_train, X_test,aType, iTer,allNT1,testType = aTuple
        aType = aType + str(iTer)

        trainAll = prepareDataset(X_train, aType)
        X, y = convert2TrainingData(trainAll,True)

        from collections import Counter


        data = X[:, 2:]
        label = y


        weight = (label + 0.1) * (Counter(y)[0] / Counter(y)[1])

        sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
        sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
        logging.info('weight statistics: wpos=%g, wneg=%g, ratio=%g' % (sum_wpos, sum_wneg, sum_wneg / sum_wpos))

        dtrain = light.Dataset(data,label=label, weight=weight)

        logging.info(len(label))

        if not test:
            testNotAll = prepareDataset(X_test, aType)
            Xt, yt = convert2TrainingData(testNotAll,True)
            dataEval = Xt[:, 2:]
            labelEval = yt

            deval = light.Dataset(dataEval, label=labelEval)

            logging.info(len(labelEval))

        #if aType != 'ALL':
        excludeTrain = set(allNT1) - set(X_train.tolist())
        X_test = list(excludeTrain)
        logging.info(len(X_train), len(X_test))
        testNotAll = prepareDataset(X_test, aType)
        Xt, yt = convert2TrainingData(testNotAll,True)
        dataTest = Xt[:, 2:]
        labelTest = yt

        logging.info(len(labelTest))


        params = {

            'learning_rate': .05,
            'subsample': 0.5,  #bagging_fraction
            'colsample_bytree': 0.8,  #feature_fraction
            'scale_pos_weight': (sum_wneg / sum_wpos),#*10,
            'nthread': thread,
            'objective': 'binary',
            'metric': [ 'rmse']#'mae']  # 70

        }
        if not test:
            watchlist = [deval]
        else:
            dtest = light.Dataset(dataTest, label=labelTest)
            watchlist = [dtest]
        num_boost_round = 10000


        logging.info('Start training')
        model = light.train(params,dtrain,
            num_boost_round=num_boost_round,
            valid_sets=watchlist,
            # evals=[(dtest, 'test')],
            early_stopping_rounds=10)


        logging.info("Best :  with {} rounds".format(

            model.best_iteration + 1))


        model.save_model(xgbResultFolder + '/' + testType + "/" + aType+ testType +'.model', num_iteration=model.best_iteration)


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
    #line2 = (pd.crosstab(index=y_test, columns=y_pred, rownames=['actual'], colnames=['predicted']))
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
            logging.info('TRAIN:', train_index, 'TEST:', test_index)
            X_train, X_test = aND[train_index], aND[test_index]
            aTuple = X_train, X_test, aType, iTer,simiFiles,'_TESTALL'
            tuples.append(aTuple)
            iTer += 1
            # break

        pool = Pool(core,maxtasksperchild=1)

        pool.map_async(coreLearn, [link for link in tuples],chunksize=1)

        pool.close()
        pool.join()
    else:
        aTuple = aND, None, aType, 0, simiFiles, '_TESTALL'
        coreLearn(aTuple,True)


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
    # runParams = getRun()
    setLogg()
    # logging.info(runParams.run)
    # if runParams.thread is not None:
    #     thread = int(runParams.thread)
    #     logging.info('thread ' + runParams.thread)
    #
    # if runParams.core is not None:
    #     core = int(runParams.core)
    #     logging.info('core ' + runParams.core)
    # if runParams.folder is not None:
    #     xgbResultFolder = runParams.folder
    #     logging.info('folder ' + runParams.folder)
    # if runParams.database is not None:
    #     database = runParams.database
    #     logging.info('database ' + runParams.database)



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






