
import logging
import sys
from os.path import isfile, join
from os import listdir
import pandas as pd
import gzip
import pickle as p
import random
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
import time
import numpy as np
from multiprocessing.pool import Pool
from pandas import HDFStore
import shutil
import os
# import learning as l

import lightgbm as light

# bufferFolder = 'buffer5'
# modelFolder = 'model5'
xgbResultFolder ='xgbResult3'
database = 'simiSmall.h5'
#simiFolder = 'simiNew'
thread = 8
core = 2



def bigDF():
    # fname = 'simi.h5'  # modelFolder + '/' + testType + aType+'my_storeTest' + str(iTer)
    # fname = 'simiSmaller.h5'


    # with HDFStore(database, mode='r') as store:
    #     df = store.select('simi')
    logging.info('reading done %s' ,database)
    with HDFStore(database, mode='r') as store:
        res = store.select_column('simi','bugID')
    logging.info(len(res))
    logging.info(len(res.unique()))
    idList= res.unique().tolist()

    # l.save_zipped_pickle(df,modelFolder + '/' +testType + '/'+aType+'.test')
    return idList



def prepareDataset(listOfSimiFiles, aType, randomSample =False ):
    logging.info('Prepare testing dataset ' + aType)
    # if isfile(modelFolder + '/' +testType + '/'+aType+'.test'):
    #     return l.load_zipped_pickle(modelFolder + '/' +testType+ '/'+aType+'.test')
    # else:

    # fname = 'simi.h5'#modelFolder + '/' + testType + aType+'my_storeTest' + str(iTer)
    # fname = 'my_storeAMQP'
    idList = [f.replace('simi.gzip', '') for f in listOfSimiFiles ]

    with HDFStore(database, mode='r') as store:
        df = store.select('simi',where='bugID in idList')

    #df = readSimi[readSimi.bugID.isin(idList)]
    if randomSample:
        simiT = df[df.answer == True]
        simiF = df[df.answer == False]
        simiFalseRandom =simiF.groupby('bugID').apply(lambda x: x.sample(3))
        df = pd.concat([simiT, simiFalseRandom])
    # print('Dataset lenght %s',len(df))
    # with HDFStore(fname,mode='r') as store:
    #     df = store.select('simi', where=[
    #         'bugID in idList'])
    # print('reading done')

    #l.save_zipped_pickle(df,modelFolder + '/' +testType + '/'+aType+'.test')
    return df

def convert2TrainingData(trainFrame,light):
    #logging.info('Convert to training data')
    if light:
        aDf = trainFrame
    else:
        aDf = trainFrame.fillna(-1)

    features = [c for c in aDf.columns if len(c.split('2')) == 2]
    features = [c for c in features if not c.endswith('Rank') and not c.startswith('comment')]
    features = ['bugID', 'file'] + features
    features = aDf[features].columns
    # groups = aDf.bugID.values.tolist()
    # x = pd.Series(groups).astype('category')
    # groups = x.cat.codes.value_counts(sort=False)
    #

    logging.info('Ground truth %s', str(aDf['answer'].value_counts()))

    for f in [features[2:]]:
        aDf[f] = aDf[f].astype(float)

    X = aDf[features].values
    y, _ = pd.factorize(aDf['answer'])
    # groups = groups.values
    return X,y#,groups

def coreLearn(aTuple, test = False):
    #core
    try:
        X_train, X_test,aType, iTer,allNT1,testType = aTuple
        aType = aType + str(iTer)
        #print(len(X_train),len(X_test))
        trainAll = prepareDataset(X_train, aType)
        X, y = convert2TrainingData(trainAll,True)

        from collections import Counter


        data = X[:, 2:]
        label = y


        # weight = (label) * Counter(y)[0] / Counter(y)[1] + 0.1  # * float(len(X_test)) / len(label)
        weight = (label + 0.1) * (Counter(y)[0] / Counter(y)[1])
        # weight = (label + 0.1) / 10000
        sum_wpos = sum(weight[i] for i in range(len(label)) if label[i] == 1.0)
        sum_wneg = sum(weight[i] for i in range(len(label)) if label[i] == 0.0)
        print('weight statistics: wpos=%g, wneg=%g, ratio=%g' % (sum_wpos, sum_wneg, sum_wneg / sum_wpos))
        # dtrain = xgb.DMatrix(data, label=label, missing=-1.0, weight=weight)
        dtrain = light.Dataset(data,label=label, weight=weight)
        # dtrain.set_group(groups)
        print(len(label))
        # print(len(groups))

        if not test:
            testNotAll = prepareDataset(X_test, aType)
            Xt, yt = convert2TrainingData(testNotAll,True)
            dataEval = Xt[:, 2:]
            labelEval = yt

            # deval= xgb.DMatrix(dataEval, label=labelEval, missing=-1.0)  # , weight=weightTest)
            deval = light.Dataset(dataEval, label=labelEval)
            # dtest.set_group(groupst)
            print(len(labelEval))

        #if aType != 'ALL':
        excludeTrain = set(allNT1) - set(X_train.tolist())
        X_test = list(excludeTrain)
        print(len(X_train), len(X_test))
        testNotAll = prepareDataset(X_test, aType)
        Xt, yt = convert2TrainingData(testNotAll,True)
        dataTest = Xt[:, 2:]
        labelTest = yt

        print(len(labelTest))
        # print(len(groupst))

        # dtest = xgb.DMatrix(dataTest, label=labelTest, missing=-1.0)#, weight=weightTest)
        # dtest = light.Dataset(dataTest, label=labelTest,reference=dtrain, free_raw_data = False)
        #dtest.set_group(groupst)


        # params = {
        #     # Parameters that we are going to tune.
        #     # 'max_depth': 2,
        #     # 'min_child_weight': Counter(y)[0] / Counter(y)[1],
        #     'eta': .1,
        #     'subsample': 0.5,
        #     'colsample_bytree': 0.8,
        #     'scale_pos_weight': sum_wneg / sum_wpos,
        #     'gamma': .6,
        #     'nthread':thread,
        #     # Other parameters
        #     'objective': 'binary:logistic',
        #     #'eval_metric': [ 'auc','rmse'], #71
        #     'eval_metric': ['auc','logloss','rmse','mae'] #70
        #
        # }

        params = {
            # Parameters that we are going to tune.
            # 'num_leaves': 255,
            # 'num_trees':500,
            # 'min_data_in_leaf': 0,
            # 'min_sum_hessian_in_leaf':100,  #min_child_weight
            # 'boosting':'dart',
            # 'num_leaves':90,
            # 'min_child_weight': Counter(y)[0] / Counter(y)[1],
            # 'sigmoid':'0.95',
            'learning_rate': .05,
            'subsample': 0.5,  #bagging_fraction
            'colsample_bytree': 0.8,  #feature_fraction
            'scale_pos_weight': (sum_wneg / sum_wpos),#*10,
            # 'is_unbalance':True,
            # 'gamma': .6,
            'nthread': thread,
            # Other parameters
            'objective': 'binary',
            # 'eval_metric': [ 'auc','rmse'], #71
            'metric': [ 'rmse']#'mae']  # 70

        }
        if not test:
            # watchlist = [(dtrain, 'train'),(deval, 'eval')]
            watchlist = [deval]
        else:
            # watchlist = [(dtrain, 'train')]
            dtest = light.Dataset(dataTest, label=labelTest)
            watchlist = [dtest]
        # watchlist = [ (deval, 'eval')]
        num_boost_round = 10000

        # model = xgb.train(
        #     params,
        #     dtrain,
        #     num_boost_round=num_boost_round,
        #     evals=watchlist,
        #     # evals=[(dtest, 'test')],
        #     early_stopping_rounds=10
        # )
        logging.info('Start training')
        model = light.train(params,dtrain,
            num_boost_round=num_boost_round,
            valid_sets=watchlist,
            # evals=[(dtest, 'test')],
            early_stopping_rounds=10)

        # print("Best : {:.2f} with {} rounds".format(
        #     model.best_score,
        #     model.best_iteration + 1))

        print("Best :  with {} rounds".format(

            model.best_iteration + 1))


        # pred_prob = model.predict(dtest,ntree_limit=model.best_iteration)
        logging.info('Start predicting')
        pred_prob = model.predict(dataTest, num_iteration=model.best_iteration)


        # tmp_train = model.predict(dtrain, output_margin=True, ntree_limit=model.best_ntree_limit)
        # tmp_test = model.predict(deval, output_margin=True, ntree_limit=model.best_ntree_limit)
        # dtrain.set_base_margin(tmp_train)
        # deval.set_base_margin(tmp_test)
        # # param['scale_pos_weight'] = (sum_wneg / sum_wpos ) * 100
        # bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals =watchlist, early_stopping_rounds=10)
        # pred_prob = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)

        bugIds = Xt[:, 0]
        files = Xt[:, 1]
        # series = [bugIds, files, pred_prob[:, 0], pred_prob[:, 1], labelTest]
        series = [bugIds, files, pred_prob, labelTest]
        df = pd.DataFrame(series)
        df = df.T
        # df.columns = ['bugId', 'file', 'pred_prob_0', 'pred_prob_1', 'answer']
        df.columns = ['bugId', 'file', 'pred_prob_1', 'answer']

        # df['pred_prob_0'] = pd.to_numeric(df['pred_prob_0'], errors='coerce')
        df['pred_prob_1'] = pd.to_numeric(df['pred_prob_1'], errors='coerce')
        # df['rank_0'] = df.groupby('bugId')['pred_prob_0'].rank(method='first', ascending=0, na_option='keep')
        df['rank_1'] = df.groupby('bugId')['pred_prob_1'].rank(method='first', ascending=0, na_option='keep')

        answer = df[df.answer == True]
        answer['AnsOrder'] = answer.groupby('bugId')['rank_1'].rank(method='first', ascending=1, na_option='keep')
        answer['AP'] = answer.apply(lambda x: (x['AnsOrder']) / (x['rank_1']), axis=1)
        answer['RR'] = answer.apply(lambda x: l.RR_XGB(x, 'rank_1'), axis=1)

        answer['Classifier'] = aType + testType
        df['Classifier'] = aType + testType

        p.dump(df, open(xgbResultFolder + '/' + testType + "/" + aType+ testType + ".pickle", "wb"))
        printResults(pred_prob, labelTest, aType, testType)


        #predictCore(Xt, yt, aType+ str(iTer),testType)

    except Exception as e:
        logging.error(e)
        raise e

def printResults(y_pred, y_test, aType, predType):
    print(82 * '_')
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

def evalResults(folder):


    if isfile(xgbResultFolder +'/'+folder + "/" + 'allClassifier.pick'):
        return pd.read_pickle(xgbResultFolder +'/'+folder+ "/" + 'allClassifier.pick')
    logging.info('Merging all results to a single file')
    simiFiles = [f for f in listdir(xgbResultFolder +'/'+folder) if isfile(join(xgbResultFolder +'/'+folder, f))]
    pick = [c for c in simiFiles if c.endswith('.pickle')]
    data = []
    for i in pick:
        a = pd.read_pickle(xgbResultFolder +'/'+folder+ '/' +i)
        #data.append(a[['bugId','file','pred_prob_1','rank_1','AnsOrder','AP','RR','Classifier']])
        data.append(a)
    dataframe = pd.concat(data)
    p.dump(dataframe, open(xgbResultFolder +'/'+folder+ "/" + 'allClassifier.pick' , "wb"))
    return dataframe
    #excelWriter(dataframe,'allClassifier')

def confidenceWeightedMVScore(folder, fn=''):
    logging.info('Calculte MRR,MAP ')
    for method in ['min','max','mean','prod']:
        ranks = pd.read_pickle(xgbResultFolder +'/'+folder+ "/" + method+ fn+ 'ranked.pick')
        filesGT = pd.read_pickle('groundTruthAnswer.pickle')
        filesGT['answer'] = True
        answers = ranks.merge(filesGT, on=['file', 'bugID'], how="inner")
        answers['AnsOrder'] = answers.groupby('bugID')['rank'].rank(method='first', ascending=1,
                                                                             na_option='keep')
        answers['AP'] = answers.apply(lambda x: (x['AnsOrder']) / (x['rank']), axis=1)
        answers['RR'] = answers.apply(lambda x: l.RR_XGB(x, 'rank'), axis=1)

        valMAP = answers.groupby('bugID')['AP'].mean()
        valMRR = answers.groupby('bugID')['RR'].max()
        series = []
        series.append(valMAP.values)
        series.append(valMRR.values)
        idx = valMAP.index.values
        series.insert(0, idx)
        df = pd.DataFrame(series)
        df = df.T
        df.columns = ['bugID','MAP','MRR']
        print(method)
        print(len(df))
        p.dump(df, open(xgbResultFolder +'/'+folder+ "/" +method+ fn+ 'MRR.pick', "wb"))
        print(df.MAP.mean(),df.MRR.mean())
    all = pd.read_pickle("allTeknik.pickle")
    tekniks = all.Approach.unique()
    allSelected = all[all.id.isin(df.bugID)]
    print(len(allSelected.id.unique()))

    dataList = dict()
    for teknik in tekniks:
        select = allSelected[allSelected.Approach == teknik]

        aps = [i for i in select.columns if i.endswith('AP')]
        # top1s = [i for i in select.columns if i.endswith('Top1')]
        # top5s = [i for i in select.columns if i.endswith('Top5')]
        # top10s = [i for i in select.columns if i.endswith('Top10')]
        rrs = [i for i in select.columns if i.endswith('TP')]

        for ap, rr in zip(aps, rrs):
            select[rr] = pd.to_numeric(select[rr], errors='coerce')
            select[ap] = pd.to_numeric(select[ap], errors='coerce')

        series = []
        colNames = []
        idx = []
        for ap, rr in zip(aps, rrs):
            # colName = ap.replace('RankAP', '')
            valMAP = select.groupby('id')[ap].mean()
            valMRR = select.groupby('id')[rr].max()
            # valTop1 = select.groupby('id')[top1].mean()
            # valTop5 = select.groupby('id')[top5].mean()
            # valTop10 = select.groupby('id')[top10].mean()
            series.append(valMAP.values)
            series.append(valMRR.values)
            # series.append(valTop1)
            # series.append(valTop5)
            # series.append(valTop10)
            colNames.append('MAP')
            colNames.append('MRR')
            # colNames.append('Top1s')
            # colNames.append('Top5s')
            # colNames.append('Top10s')
            idx = valMAP.index.values

        colNames.insert(0, 'bugID')
        series.insert(0, idx)
        df = pd.DataFrame(series)
        df = df.T
        df.columns = colNames
        dataList[teknik] = df
        print(df.MAP.mean(), df.MRR.mean())
        p.dump(df, open(xgbResultFolder +'/'+folder+ "/" + teknik + 'MRR.pick', "wb"))

# multi classifier based on confidence weighted majority voting
def confidenceWeightedMV(folder, method ='mean' , fn = ''):
    logging.info('Calculate multi classifier ' + method)

    if isfile(xgbResultFolder +'/'+folder+ "/" +method+ fn+ 'ranked.pick'):
        return
    # import math
    # def g(x, min, max):
    #     if pd.isnull(x):
    #         return np.nan
    #     elif min == max:
    #         return np.nan
    #     else:
    #         Nx = (x - min) / (max - min)
    #         ex = math.exp(-Nx)
    #         g = 1 / (1 + ex)
    #         return g
    #bir = allRes[allRes.bugId =='AMQP-Spring109']
    #bir.groupby(['bugId', 'Classifier']).apply(lambda x: confidenceWeightedMVScore(x))
    from sklearn.preprocessing import minmax_scale
    allRes['test'] = allRes.groupby(['bugId', 'Classifier'])['pred_prob_1'].transform(lambda x: minmax_scale(x.astype(float)))
    pair = allRes.groupby(['bugId', 'file'])
    if method == 'mean':
        # meanVal = allRes.groupby(['bugId', 'file'])['pred_prob_1'].mean()
        meanVal = allRes.groupby(['bugId', 'file'])['test'].mean()
    elif method == 'max':
        meanVal = allRes.groupby(['bugId', 'file'])['test'].max()
    elif method == 'min':
        meanVal = allRes.groupby(['bugId', 'file'])['test'].min()
    elif method =='prod':
        meanVal = allRes.groupby(['bugId', 'file'])['test'].prod()
    elif method =='rank':
        meanVal = allRes.groupby(['bugId', 'file']).agg({'test': 'max', 'rank_1': 'min','answer':'max'}).reset_index()
        print(meanVal)
    series = []
    idNames = [i for i, j in meanVal.index.values]
    fileNames = [j for i, j in meanVal.index.values]
    answer = pair['answer'].max()
    series.append(idNames)
    series.append(fileNames)
    series.append(meanVal.values.tolist())
    series.append(answer.values.tolist())
    df = pd.DataFrame(series)
    df = df.T
    df.columns = ['bugID','file','avgPred','answer']

    df['avgPred'] = pd.to_numeric(df['avgPred'], errors='coerce')
    df['rank'] = df.groupby('bugID')['avgPred'].rank(method='first', ascending=0, na_option='keep')
    p.dump(df, open(xgbResultFolder +'/'+folder+ "/" +method+ fn+ 'ranked.pick', "wb"))

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
        print(len(aND))
        kf = KFold(n_splits=len(aND), shuffle=True)

    else:
        kf = KFold(n_splits=10,shuffle=True)  # Define the split - into 2 folds
    if aType != 'BLUiR':
        iTer = 0
        tuples = []
        for train_index, test_index in kf.split(aND):
            #print('TRAIN:', train_index, 'TEST:', test_index)
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
    # setLogg()
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
    #shutil.rmtree(xgbResultFolder + '/_TESTTOP/')
    #shutil.rmtree(xgbResultFolder + '/_TESTALL/')
    # shutil.rmtree(bufferFolder + '/_TESTTOP/')
    # shutil.rmtree(modelFolder + '/_TESTTOP/')
    #os.system("mkdir %s" % modelFolder + '/_TESTTOP/')
    #os.system("mkdir %s" % bufferFolder + '/_TESTTOP/')
    # os.system("mkdir %s" % xgbResultFolder + '/_TESTTOP/')
    os.system("mkdir %s" % xgbResultFolder + '/_TESTALL/')
    # os.system("mkdir %s" % xgbResultFolder + '/_NTOPNTOP/')
    # os.system("mkdir %s" % xgbResultFolder + '/_NTOPALL/')


    # simiFiles = ['AMQP-Spring551simi.gzip']
    # random_items = random.choices(population=simiFiles, k=600)
    # trainAll = prepareTrainingDataset(random_items, 'ALL')
    simiFiles = bigDF()
    # simiFiles = readSimi.bugID.unique()
    # idFrames = pd.read_pickle('idFramesTeknik.pickle')
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


    print('Len of all ,inter, locus, blia, bugL, bbrt, amalgam, bluir, allN')

    allNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in allT1Ids]
    print(len(allTop), len(interTop), len(locusTop), len(bliaTop), len(bugLTop), len(brtTop), len(amalTop),
          len(bluirTop), len(allNTop))
    # interNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['intersection'].values.tolist()[0]]
    # locusNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['OnlyLocus'].values.tolist()[0]]
    # bliaNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['OnlyBLIA'].values.tolist()[0]]
    # bugLNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['OnlyBugLocator'].values.tolist()[0]]
    # brtNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['OnlyBRTracer'].values.tolist()[0]]
    # amalNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['OnlyAmaLgam'].values.tolist()[0]]
    # bluirNTop = [f for f in simiFiles if f.replace('simi.gzip', '') not in topN['OnlyBLUiR'].values.tolist()[0]]


    # if runParams.run == 'AMALGAM':
    runTrainTopTestAll('AMALGAM')
    # # if runParams.run == 'ALL':
    runTrainTopTestAll('ALL')
    # # if runParams.run == 'INTER':
    runTrainTopTestAll('INTER')
    # # if runParams.run == 'BRTracer':
    runTrainTopTestAll('BRTracer')
    # # if runParams.run == 'BugLocator':
    runTrainTopTestAll('BugLocator')
    # # if runParams.run == 'BLIA':
    runTrainTopTestAll('BLIA')
    # # if runParams.run == 'LOCUS':
    runTrainTopTestAll('LOCUS')
    # if runParams.run == 'NALL':
    runTrainTopTestAll('NALL')
    # if runParams.run == 'BLUiR':
    runTrainTopTestAll('BLUiR')


    # allRes = evalResults('_TESTALL')
    # # allRes = pd.read_pickle(xgbResultFolder + '/' + '_TESTALL' + "/" + 'allClassifier.pick')
    # confidenceWeightedMV('_TESTALL')
    # confidenceWeightedMV('_TESTALL', method='max')
    # confidenceWeightedMV('_TESTALL', method='min')
    # confidenceWeightedMV('_TESTALL', method='prod')
    # # confidenceWeightedMV('_TESTALL', method='rank')
    # confidenceWeightedMVScore('_TESTALL')
    #
    # allRes = allRes[allRes.Classifier.str.startswith('ALL')]
    # confidenceWeightedMV('_TESTALL', method='mean', fn='ALL')
    # confidenceWeightedMV('_TESTALL', method='max',fn='ALL')
    # confidenceWeightedMV('_TESTALL', method='min' ,fn='ALL')
    # confidenceWeightedMV('_TESTALL', method='prod',fn='ALL')
    # confidenceWeightedMVScore('_TESTALL',fn='ALL')




