import pickle as p
import gzip
import pandas as pd
from os.path import isfile, join
from os import listdir
import logging
import sys
from multiprocessing.pool import Pool
from pandas import HDFStore
import sqlite3


def setLogg():
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(process)d - %(levelname)s - %(funcName)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch)

def mergeAll():

    if isfile(xgbResultFolder + '/' +'allResults.db'):
        logging.info('skip merge')
        return
    conn = sqlite3.connect(xgbResultFolder + '/' + "allResults.db")

    files = [f for f in listdir(xgbResultFolder) if isfile(join(xgbResultFolder, f)) and f.endswith('predProb.pickle')]
    for file in files:
        logging.info(file)
        df = pd.read_pickle(xgbResultFolder + '/'+ file)
        df['answer'] = df['answer'].astype(int)
        df.to_sql("result", conn, if_exists="append",index=False)



def confidenceWeightedMV(aTuple):
    idx = aTuple
    dbNAme = 'file:' + xgbResultFolder + '/' + 'allResults.db' + '?mode=ro'
    conn = sqlite3.connect(dbNAme, uri=True)

    if tip == 'BLUiR':
        allRes = pd.read_pickle(xgbResultFolder + "/BLUiR0_TESTALL.pickle")
    elif tip == 'MULTI':
        query = "select * from result where  bugId == '{param}';".format(param=idx)
        allRes = pd.read_sql_query(query, conn)
    else:
        query = "select * from result where Classifier like '{tip}%' and bugId == '{param}';".format(param=idx, tip=tip)
        allRes = pd.read_sql_query(query, conn)

    from sklearn.preprocessing import minmax_scale
    allRes['test'] = allRes.groupby(['bugId', 'Classifier'])['pred_prob_1'].transform(
        lambda x: minmax_scale(x.astype(float)))

    pair = allRes.groupby(['bugId', 'file'])

    meanN = pair['test'].mean()
    maxN = pair['test'].max()
    minN = pair['test'].min()
    prodN = pair['test'].prod()
    meanVal = pair['pred_prob_1'].mean()
    maxVal = pair['pred_prob_1'].max()
    minVal = pair['pred_prob_1'].min()
    prodVal = pair['pred_prob_1'].prod()
    import numpy as np
    wavg= pair.apply(lambda x: np.average(x['pred_prob_1'], weights=1/x['rank_1']*100))
    wavgN = pair.apply(lambda x: np.average(x['test'], weights=1 / x['rank_1']*100))


    series = []
    idNames = [i for i, j in meanVal.index.values]
    fileNames = [j for i, j in meanVal.index.values]
    answer = pair['answer'].max()
    series.append(idNames)
    series.append(fileNames)
    series.append(meanN.values.tolist())
    series.append(maxN.values.tolist())
    series.append(minN.values.tolist())
    series.append(prodN.values.tolist())
    series.append(meanVal.values.tolist())
    series.append(maxVal.values.tolist())
    series.append(minVal.values.tolist())
    series.append(prodVal.values.tolist())
    series.append(wavg.values.tolist())
    series.append(wavgN.values.tolist())


    series.append(answer.values.tolist())
    df = pd.DataFrame(series)
    df = df.T
    df.columns = ['bugID','file','meanN','maxN','minN','prodN','meanPred','maxPred','minPred','prodPred','wavgPred','wavgPredN','answer']

    df['meanN'] = pd.to_numeric(df['meanN'], errors='coerce')
    df['meanNRank'] = df.groupby('bugID')['meanN'].rank(method='dense', ascending=0, na_option='keep')

    df['meanPred'] = pd.to_numeric(df['meanPred'], errors='coerce')
    df['meanRank'] = df.groupby('bugID')['meanPred'].rank(method='dense', ascending=0, na_option='keep')

    df['maxN'] = pd.to_numeric(df['maxN'], errors='coerce')
    df['maxNRank'] = df.groupby('bugID')['maxN'].rank(method='dense', ascending=0, na_option='keep')

    df['maxPred'] = pd.to_numeric(df['maxPred'], errors='coerce')
    df['maxRank'] = df.groupby('bugID')['maxPred'].rank(method='dense', ascending=0, na_option='keep')

    df['minN'] = pd.to_numeric(df['minN'], errors='coerce')
    df['minNRank'] = df.groupby('bugID')['minN'].rank(method='dense', ascending=0, na_option='keep')

    df['minPred'] = pd.to_numeric(df['minPred'], errors='coerce')
    df['minRank'] = df.groupby('bugID')['minPred'].rank(method='dense', ascending=0, na_option='keep')

    df['prodN'] = pd.to_numeric(df['prodN'], errors='coerce')
    df['prodNRank'] = df.groupby('bugID')['prodN'].rank(method='dense', ascending=0, na_option='keep')

    df['prodPred'] = pd.to_numeric(df['prodPred'], errors='coerce')
    df['prodRank'] = df.groupby('bugID')['prodPred'].rank(method='dense', ascending=0, na_option='keep')

    df['wavgPred'] = pd.to_numeric(df['wavgPred'], errors='coerce')
    df['wavgRank'] = df.groupby('bugID')['wavgPred'].rank(method='dense', ascending=0, na_option='keep')

    df['wavgPredN'] = pd.to_numeric(df['wavgPredN'], errors='coerce')
    df['wavgRankN'] = df.groupby('bugID')['wavgPredN'].rank(method='dense', ascending=0, na_option='keep')



    df = df[df.answer == True]
    return df

def RR_XGB(x,column):
    if x['AnsOrder'] == 1:
        return (1.0 / (x[column]))
    elif pd.isnull(x['AnsOrder']):
        return None
    else:
        return 0



def confidenceWeightedMVScore(answer):
    logging.info('Calculte MRR,MAP ')
    answer.reset_index(inplace=True)
    for method in ['meanNRank','maxNRank','minNRank','prodNRank','meanRank','maxRank','minRank','prodRank','wavgRank','wavgRankN']:

        answer['AnsOrder'] = answer.groupby('bugID')[method].rank(method='dense', ascending=1,
                                                                             na_option='keep')
        answer['AP'] = answer.apply(lambda x: (x['AnsOrder']) / (x[method]), axis=1)
        answer['RR'] = answer.apply(lambda x: RR_XGB(x, method), axis=1)

        valMAP = answer.groupby('bugID')['AP'].mean()
        valMRR = answer.groupby('bugID')['RR'].max()
        series = []
        series.append(valMAP.values)
        series.append(valMRR.values)
        idx = valMAP.index.values
        series.insert(0, idx)
        df = pd.DataFrame(series)
        df = df.T
        df.columns = ['bugID','MAP','MRR']
        logging.info(method + ' '+ str(len(df)))
        p.dump(df, open(xgbResultFolder + "/" +tip +'_' +method+ 'MRR.pick', "wb"))
        logging.info(df.MAP.mean(),df.MRR.mean())
    all = pd.read_pickle("allTeknik.pickle")
    tekniks = all.Approach.unique()
    allSelected = all[all.id.isin(df.bugID)]
    logging.info('AllTeknik:' + str(len(allSelected.id.unique())))

    dataList = dict()
    for teknik in tekniks:
        select = allSelected[allSelected.Approach == teknik]

        aps = [i for i in select.columns if i.endswith('AP')]
        top1s = [i for i in select.columns if i.endswith('Top1')]
        top5s = [i for i in select.columns if i.endswith('Top5')]
        top10s = [i for i in select.columns if i.endswith('Top10')]
        rrs = [i for i in select.columns if i.endswith('TP')]

        for ap, rr in zip(aps, rrs):
            select[rr] = pd.to_numeric(select[rr], errors='coerce')
            select[ap] = pd.to_numeric(select[ap], errors='coerce')

        series = []
        colNames = []
        idx = []
        for ap, rr, top1, top5, top10 in zip(aps, rrs, top1s, top5s, top10s):
            # colName = ap.replace('RankAP', '')
            valMAP = select.groupby('id')[ap].mean()
            valMRR = select.groupby('id')[rr].max()
            valTop1 = select.groupby('id')[top1].mean()
            valTop5 = select.groupby('id')[top5].mean()
            valTop10 = select.groupby('id')[top10].mean()
            series.append(valMAP.values)
            series.append(valMRR.values)
            series.append(valTop1)
            series.append(valTop5)
            series.append(valTop10)
            colNames.append('MAP')
            colNames.append('MRR')
            colNames.append('Top1s')
            colNames.append('Top5s')
            colNames.append('Top10s')
            idx = valMAP.index.values

        colNames.insert(0, 'bugID')
        series.insert(0, idx)
        df = pd.DataFrame(series)
        df = df.T
        df.columns = colNames
        dataList[teknik] = df
        logging.info(df.MAP.mean(), df.MRR.mean())
        p.dump(df, open(xgbResultFolder + "/" + teknik + 'MRR.pick', "wb"))

def read():

    try:

        dbNAme  = 'file:' + xgbResultFolder + '/' + 'allResults.db' + '?mode=ro'
        conn = sqlite3.connect(dbNAme, uri=True)
        query = "select distinct(bugId) from result;"
        files = pd.read_sql_query(query, conn)
        idList= files.bugId.unique().tolist()
        logging.info('read done')


        pool = Pool(8, maxtasksperchild=1)

        if tip == 'BLUiR':
            dataL = confidenceWeightedMV(idList)
        else:
            data = pool.map(confidenceWeightedMV, [link for link in idList], chunksize=1)

            dataL = pd.concat(data)

        confidenceWeightedMVScore( dataL)
    except Exception as e:
        logging.error(e)
        raise e


def getRun():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-run', dest='run', help='which part')
    parser.add_argument('-c', dest='core', help='core number')

    parser.add_argument('-t', dest='thread', help='thread number')
    parser.add_argument('-f', dest='folder', help='folder')


    args = parser.parse_args()


    return args
xgbResultFolder = '/Volumes/Lexar/xgbResult3/_TESTALL'

if __name__ == '__main__':

    runParams = getRun()
    setLogg()
    if runParams.folder is not None:
        xgbResultFolder = runParams.folder
        logging.info('folder ' + runParams.folder)
    mergeAll()
    tip = 'MULTI'
    read()

