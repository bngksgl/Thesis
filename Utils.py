import copy
import os

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

# libact classes
from libact.base.dataset import Dataset, import_libsvm_sparse
from sklearn.linear_model import LogisticRegression
from libact.models import *
from libact.query_strategies import *
from libact.labelers import IdealLabeler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression as ls
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import libact
from libact.models import SVM
from libact.query_strategies import QUIRE, UncertaintySampling, RandomSampling, ActiveLearningByLearning, HintSVM
import warnings
from sklearn.cluster import KMeans
import random
warnings.filterwarnings("ignore")


def prepare_ytrain_foract(y_train, samp):
    """
    Used to prepare training dataset of target variables
    """
    i = 0
    yy_train=pd.DataFrame(y_train)
    model_y_train=[]
    while (i<len(y_train)):
        if(i in samp):
            model_y_train.append(yy_train[0].loc[i])
        else:
            model_y_train.append(None)
        i=i+1
        
    return model_y_train

def get_stats_table(phone_result_sub,only_page_columns,only_cat_columns):
    """
    Produces statistics table
    """
    no_count=[phone_result_sub[i].count() for i in list(phone_result_sub.columns)]
    no_unique =[phone_result_sub[i].nunique() for i in list(phone_result_sub.columns)]
    no_missing=pd.DataFrame(phone_result_sub.isna().sum())[0].values
    percent_mis=np.round(no_missing/phone_result_sub.shape[0],3)
    page_mis=['1' if i in only_page_columns else '0' for i in list(phone_result_sub.columns)]
    cat_mis=['1' if i in only_cat_columns else '0' for i in list(phone_result_sub.columns)]
    top_row= [phone_result_sub[i].value_counts(dropna=False).index[0] for i in list(phone_result_sub.columns)]
    no_freq=[phone_result_sub[i].value_counts(dropna=False).values[0] for i in list(phone_result_sub.columns)]
    phone_stat=pd.DataFrame({'column_name':list(phone_result_sub.columns),'count':no_count, 'no_unique': no_unique, 'no_missing':no_missing, 'percent_missing': percent_mis, 
                  'only in page':page_mis, 'only in catalog':cat_mis, 'most_freq':top_row, 'no_most_freq': no_freq})
    phone_stat['most_freq']=phone_stat['most_freq'].replace(np.NaN, 'nan')

    phone_stat=phone_stat.set_index('column_name')
    return phone_stat


def get_seed_random(y_train, seed=10):
    """
    Get the seeds randomly. In order to make sure that the randomly selected seeds, have at least a match, draw random samples
    indefinetly until you find a sample that have at least a match.
    """
    df2=pd.DataFrame(y_train)
    i = 0
    count = 0
    while (i==0): 
        samp=df2.sample(seed)
        if 1 in list(samp[0]):
            break
        count=count+1
    return list(samp.index), count



def get_seed_cluster(y_train, df2, seed=10):
    """
    Get the seeds according to the clusters. In order to make sure that the randomly selected seeds from each cluster, have at least a match, draw random samples indefinetly until you find a sample that have at least a match.
    """
    
    kmeans = KMeans(n_clusters=seed, random_state=0).fit(df2)
    labels_kmean=kmeans.labels_
    cluster = labels_kmean

    i = 0
    count = 0
    
    while (i==0):
        samp_index =[]
        for k in range(0,seed):
            samp_index.append(random.sample(count_in_list(cluster, k),1)[0])
        
        matchs=[y_train[j] for j in samp_index]
        
        if 1 in list(matchs):
            break
        count=count+1
    
    return list(samp_index), count

def count_in_list(ls, ct):
    sel=[]
    i=0
    while i<len(ls): 
        if (ls[i]==ct):
            sel.append(i)
        i=i+1
    
    return sel

def get_columns(cols_all, jac_columns_el):
    cols=[]
    c=0
    for i in cols_all:
        if i in jac_columns_el:
            cols.append(c)
        c=c+1
    return cols



def eliminate_columns(data,col_list):
    """
    For removing columns that returns the same value for the entire column
    """
    for i in col_list:
        if (len(data[i].unique())==1):
            col_list.remove(i)
    return col_list


def feature_selection(X_train, y_train, all_cols, f_class=True):
    """
    Feature selection of F Classification
    """
    if (f_class==True):
        X_new = SelectKBest(f_classif, k=20).fit(X_train, y_train)
    else:
        X_new = SelectKBest(mutual_info_classif, k=20).fit(X_train, y_train)
    
    mask = X_new.get_support() #list of booleans
    new_features_f = [] # The list of your K best features

    for bool, feature in zip(mask, all_cols):
        if bool:
            new_features_f.append(feature)

    col_index=[all_cols.index(i) for i in new_features_f]
    
    return col_index,new_features_f

def run_featureselection(trn_dss, tst_ds, y_train, model, method_, qs, X_test, y_test, all_cols, save_name, save, type_, part=20):
    """
    Batch active learning algorithm with feature selection
    """
    E_in, E_out = [], []
    f1score  =[]
    features_ls = []
    label_holder, asked_id = [], []
    tn, fp, fn, tp =[], [], [], []
    
    k= trn_dss.len_labeled()
    k_beg= trn_dss.len_labeled()
    quota = len(trn_dss.data)
    iter_ = 0
    
    while (k<quota):
        clear_output(wait=True)

        # Standard usage of libact objects
        # make_query returns the index of the sample that the active learning algorithm would like to query
        lbls, asks =[],[]
        
        if(part<trn_dss.len_unlabeled()):
            part1=part
        else:
            part1=trn_dss.len_unlabeled()
        
        # -------------------> Feature Selection
        # select features with feature selection
        X_train_feature=[i[0] for i in trn_dss.get_labeled_entries()]
        y_train_feature=[i[1] for i in trn_dss.get_labeled_entries()]
        col_index, features_f=feature_selection(X_train_feature, y_train_feature, all_cols, f_class=True)
        
        features_ls.append(features_f)
        
        # update the X_train dataset and y_train with the current selection of variables
        X_train_updated=[i[0][col_index] for i in trn_dss.data]
        y_train_updated=[i[1] for i in trn_dss.data]
        trn_dss_updated=Dataset(X_train_updated, y_train_updated)
        
        # update X_test 
        X_test_feature=[i[col_index] for i in X_test]    
        
        if(type_=='random'):
            qs = RandomSampling(trn_dss_updated, method=method_, model=model)
            model1=model
        elif(type_=='unc'):
            qs=UncertaintySampling(trn_dss_updated, method=method_, model=model)
            model1=model
        elif(type_=='qbc'):
            qs = QueryByCommittee(trn_dss_updated, models=model)
            model1=method_
        elif(type_=='dens'):
            qs = DWUS(trn_dss_updated, model=model)
            model1=model
        
        for i in range(0,part1):
            # ask id only asks for particular id, not all, everytime
            ask_id=qs.make_query()
            asks.append(ask_id)
            # lbl label returns the label of a given sample
            lb = y_train[ask_id]
            lbls.append(lb)
            # update updates the unlabeled sample with queried sample
            trn_dss.update(ask_id, lb)
            trn_dss_updated.update(ask_id, lb)
            
        label_holder.append(lbls)
        asked_id.append(asks)
        
        # trains only on the labeled examples and chosen values
        model1.train(trn_dss_updated)
        # predict it 
        pred_y=model1.predict(X_test_feature)
        
        # save the results
        f1score.append(f1_score(y_test, pred_y))
        tn.append(confusion_matrix(y_test,pred_y)[0][0])
        fp.append(confusion_matrix(y_test,pred_y)[0][1])
        fn.append(confusion_matrix(y_test,pred_y)[1][0])
        tp.append(confusion_matrix(y_test,pred_y)[1][1])
        
        # score returns the mean accuracy of the results
        #E_in = np.append(E_in, 1 - model.score(trn_dss)) #train
        #E_out = np.append(E_out, 1 - model.score(tst_ds)) #test

        k=trn_dss_updated.len_labeled()
        print(k)
        print(quota)
        print('iteration:', iter_)
        print(len(f1score))
        print('train dataset labeled:', trn_dss.len_labeled())
        print('train dataset shape:',trn_dss.format_sklearn()[0].shape)
        print('train dataset sum:',trn_dss.format_sklearn()[1].sum())
        print('Current f1 score:',f1_score(y_test, pred_y))
        print('Current progress:',np.round(k/quota*100,2),'%')
        print('Chosen_features:',features_f)
        
        # number of iterations
        iter_=iter_+1
        
    q= [i for i in range(k_beg,quota,part)]
    iter_=[i for i in range(0,len(f1score))]

        
    if (save==True):
        #q= [i for i in range(k_beg,quota,part)]
        #iter_=[i for i in range(0,len(f1score))]
        saved_file=pd.DataFrame({'iter':iter_,'quota':q, 'f1_score':f1score,'tn': tn, 'fp':fp, 'fn': fn, 'tp':tp, 'id_index':asked_id,'label':label_holder, 'features':features_ls})
        saved_file.to_csv(save_name)
        
        
        
    return q, iter_, f1score, tn, fp,fn, tp, k, trn_dss.data, label_holder, asked_id, features_ls



def run_faster(trn_dss, tst_ds, y_train, model, qs, X_test, y_test, save_name, save, part=20):
    """
    Batch active learning algorithm
    """
    # the main active learning algorithm
    E_in, E_out = [], []
    f1score  =[]
    label_holder, asked_id = [], []
    tn, fp, fn, tp =[], [], [], []
    
    k= trn_dss.len_labeled()
    k_beg= trn_dss.len_labeled()
    
    quota = len(trn_dss.data)
    iter_ = 0
    while (k<quota):
        clear_output(wait=True)

        # Standard usage of libact objects
        # make_query returns the index of the sample that the active learning algorithm would like to query
        lbls, asks =[],[]
       
        if(part<trn_dss.len_unlabeled()):
            part1=part
        else:
            part1=trn_dss.len_unlabeled()
            
        for i in range(0,part1):
            # ask id only asks for particular id, not all, everytime
            ask_id=qs.make_query()
            asks.append(ask_id)
            # lbl label returns the label of a given sample
            lb = y_train[ask_id]
            lbls.append(lb)
            # update updates the unlabeled sample with queried sample
            trn_dss.update(ask_id, lb)
            
        label_holder.append(lbls)
        asked_id.append(asks)
        
            
        
        # trains only on the labeled examples
        model.train(trn_dss)
        # predict it 
        pred_y=model.predict(X_test)
        
        # save the results
        f1score.append(f1_score(y_test, pred_y))
        tn.append(confusion_matrix(y_test,pred_y)[0][0])
        fp.append(confusion_matrix(y_test,pred_y)[0][1])
        fn.append(confusion_matrix(y_test,pred_y)[1][0])
        tp.append(confusion_matrix(y_test,pred_y)[1][1])
        
        # score returns the mean accuracy of the results
        #E_in = np.append(E_in, 1 - model.score(trn_dss)) #train
        #E_out = np.append(E_out, 1 - model.score(tst_ds)) #test

        k=trn_dss.len_labeled()
        print(k)
        print(quota)
        print('train dataset labeled:', trn_dss.len_labeled())
        print('train dataset sum:',trn_dss.format_sklearn()[1].sum())
        print('Current f1 score:',f1_score(y_test, pred_y))
        print('Current progress:',np.round(k/quota*100,2),'%')
        # number of iterations
        #iter_=iter_+1
        
    q= [i for i in range(k_beg,quota,part)]
    iter_=[i for i in range(0,len(f1score))]

    if (save==True):
        #q= [i for i in range(k_beg,quota,part)]
        #iter_=[i for i in range(0,len(f1score))]
        saved_file=pd.DataFrame({'iter':iter_,'quota':q, 'f1_score':f1score, 'tn': tn, 'fp':fp, 'fn': fn, 'tp':tp,'id_index':asked_id,'label':label_holder})
        saved_file.to_csv(save_name)
    
        
    return q, iter_, f1score, tn, fp,fn, tp, k, trn_dss.data, label_holder, asked_id
    

def get_result_statistics(y_test, pred_y):
    """
    Return the general statistics
    """
    
    f1scor=f1_score(y_test, pred_y)
    tn=confusion_matrix(y_test,pred_y)[0][0]
    fp=confusion_matrix(y_test,pred_y)[0][1]
    fn=confusion_matrix(y_test,pred_y)[1][0]
    tp=confusion_matrix(y_test,pred_y)[1][1]
    
    return f1scor, tn, fp, fn, tp

def get_base_model_result(y_test, y_train, X_test, X_train, regular_model):
    """
    Get a base model without active learning
    """
    regular_model.fit(X_train, y_train)
    pred_y=regular_model.predict(X_test)
    f1_score, tn, fp, fn, tp=get_result_statistics(y_test,pred_y)
    return f1_score, tn, fp, fn, tp

def get_clean_collist(df2, substr):
    """
    Used to remove specific columns from the df
    """
    col_list=list(df2.columns)
    col_list.remove('match')
    col_list.remove('id_webpage')
    col_list.remove('prodcat_id')
    cols=[]
    for i in col_list:
        if('page' in i):
            cols.append(substr+i[4:])
            
    return cols

