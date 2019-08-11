# in jaccard, we are treating them as q-grams
# in jarowinkler and leinchewisten we are treating them as one single string
import string
import math
import pandas as pd
from similarity.normalized_levenshtein import NormalizedLevenshtein
from similarity.jarowinkler import JaroWinkler
import re

_SPACE_PATTERN = re.compile("\\s+")

normalized_levenshtein = NormalizedLevenshtein()
jarowinkler = JaroWinkler()
tokenize = lambda doc: doc.lower().split(" ")

def jaccard_similarity(query, document):
    """
    Jaccard similarity as tokens
    """
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def jaccard_similarity_qgram(s0, s1,k):
    """
    Jaccard similarity as qgrams
    """
    if s0 is None:
        raise TypeError("Argument s0 is NoneType.")
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")
    if s0 == s1:
        return 1.0
    if len(s0) < k or len(s1) < k:
        return 0.0

    profile0 = get_profile_jaccard(s0,k)
    profile1 = get_profile_jaccard(s1,k)
    union = set()
    for ite in profile0.keys():
        union.add(ite)
    for ite in profile1.keys():
        union.add(ite)
    inter = int(len(profile0.keys()) + len(profile1.keys()) - len(union))
    return 1.0 * inter / len(union)

def get_profile_jaccard(string,k):
    """
    Make the strings qgram to be used in jaccard
    """
    shingles = dict()
    no_space_str = _SPACE_PATTERN.sub("", string)
    for i in range(len(no_space_str) - k + 1):
        shingle = no_space_str[i:i + k]
        old = shingles.get(shingle)
        if old:
            shingles[str(shingle)] = int(old + 1)
        else:
            shingles[str(shingle)] = 1
    return shingles
    
    
def normalize_list(ls):
    """
    Remove punctuations from a list of inputs and also make them lowercase
    """
    for i in range(0,len(ls)):
        if(pd.isna(ls[i])):
            ls[i]=ls[i]
        elif isinstance(ls[i], (int,float)):
            ls[i]=str(ls[i])
        else:
            ls[i]=ls[i].translate(str.maketrans('','','\r\n\t\,)};{$.|:(#"[]/>+=?')).lower()
    return ls

def calculate_similarities(data,page_columns,cat_columns,jac_columns,lev_columns,jaro_columns):
    """
    Calculation of similarities in the first missing handling option. Its asumed that missing
    values have already been taken care of before employing this method. 
    """
    # in here we already filled the mising values with zeros
    i=0
    while(i<len(jac_columns)):
        jc_sim=[]
        lev_sim=[]
        jaro_sim=[]
        k=0
        # normalize the columns and make them list 
        list_page=normalize_list(list(data[page_columns[i]]))
        list_ca=normalize_list(list(data[cat_columns[i]]))        
        tokenized_page = [tokenize(d) for d in list_page] # tokenized docs
        tokenized_ca = [tokenize(d) for d in list_ca] # tokenized docs
        while(k<data.shape[0]):
            jc_sim.append(jaccard_similarity_qgram(list_ca[k], list_page[k],2))
            lev_sim.append(1- (normalized_levenshtein.distance(list_ca[k], list_page[k])))
            jaro_sim.append(jarowinkler.similarity(list_ca[k], list_page[k]))
            k=k+1
        data[jac_columns[i]]=jc_sim
        data[lev_columns[i]]=lev_sim
        data[jaro_columns[i]]=jaro_sim
        i=i+1
        
    return data


def calculate_similarities_option2(data,page_columns,cat_columns,jac_columns,lev_columns,jaro_columns):
    """
    Calculation of similarities in the second missing handling option, where 0 is assigned if either of
    the columns are null
    """
    # in here we are assigning 0 if either of the columns are null 
    i=0
    while(i<len(jac_columns)):
        jc_sim=[]
        lev_sim=[]
        jaro_sim=[]
        k=0
        list_page=normalize_list(list(data[page_columns[i]]))
        list_ca=normalize_list(list(data[cat_columns[i]]))        
        tokenized_page = ['99999' if pd.isna(d) else tokenize(d) for d in list_page] # tokenized docs
        tokenized_ca = ['99999' if pd.isna(d) else tokenize(d) for d in list_ca] # tokenized docs
        while(k<data.shape[0]):
            if ('99999' not in tokenized_page[k]):
                if ('99999' not in tokenized_ca[k]):
                    jc_sim.append(jaccard_similarity_qgram(list_ca[k], list_page[k],2))
                    lev_sim.append(1- (normalized_levenshtein.distance(list_ca[k], list_page[k])))
                    jaro_sim.append(jarowinkler.similarity(list_ca[k], list_page[k]))
                else:
                    jc_sim.append(0)
                    lev_sim.append(0)
                    jaro_sim.append(0)
            else:
                jc_sim.append(0)
                lev_sim.append(0)
                jaro_sim.append(0)
            k=k+1
        data[jac_columns[i]]=jc_sim
        data[lev_columns[i]]=lev_sim
        data[jaro_columns[i]]=jaro_sim
        i=i+1
    
    return data
    
    
def calculate_similarities_option3(data,page_columns,cat_columns,jac_columns,lev_columns,jaro_columns, miss_columns):
    """
    Calculation of similarities in the third missing handling option, where 0 is assigned if either of
    the columns are null and additional columns are created that represent whether the column is missing or not
    """
    i=0
    while(i<len(jac_columns)):
        jc_sim=[]
        lev_sim=[]
        jaro_sim=[]
        miss=[]
        k=0
        list_page=normalize_list(list(data[page_columns[i]]))
        list_ca=normalize_list(list(data[cat_columns[i]]))        
        tokenized_page = ['99999' if pd.isna(d) else tokenize(d) for d in list_page] # tokenized docs
        tokenized_ca = ['99999' if pd.isna(d) else tokenize(d) for d in list_ca] # tokenized docs
        while(k<data.shape[0]):
            if ('99999' not in tokenized_page[k]):
                if ('99999' not in tokenized_ca[k]):
                    jc_sim.append(jaccard_similarity_qgram(list_ca[k], list_page[k],2))
                    lev_sim.append(1- (normalized_levenshtein.distance(list_ca[k], list_page[k])))
                    jaro_sim.append(jarowinkler.similarity(list_ca[k], list_page[k]))
                    miss.append(0)
                else:
                    jc_sim.append(0)
                    lev_sim.append(0)
                    jaro_sim.append(0)
                    miss.append(1)
            else:
                jc_sim.append(0)
                lev_sim.append(0)
                jaro_sim.append(0)
                miss.append(1)
            k=k+1
        data[jac_columns[i]]=jc_sim
        data[lev_columns[i]]=lev_sim
        data[jaro_columns[i]]=jaro_sim
        data[miss_columns[i]]=miss
        i=i+1
        
    return data

def calculate_similarities_option4(data,page_columns,cat_columns,jac_columns, lev_columns,jaro_columns):
    """
    Calculation of similarities in the fourth missing handling option, where -1 is assigned if either of
    the columns are null, without creating additional columns
    """
    # in here we are assigning -1 if either of the column is null, without creating additional columns
    i=0
    while(i<len(jac_columns)):
        jc_sim=[]
        lev_sim=[]
        jaro_sim=[]
        jac_new =[]
        k=0
        list_page=normalize_list(list(data[page_columns[i]]))
        list_ca=normalize_list(list(data[cat_columns[i]]))        
        tokenized_page = ['99999' if pd.isna(d) else tokenize(d) for d in list_page] # tokenized docs
        tokenized_ca = ['99999' if pd.isna(d) else tokenize(d) for d in list_ca] # tokenized docs
        while(k<data.shape[0]):
            if ('99999' not in tokenized_page[k]):
                if ('99999' not in tokenized_ca[k]):
                    jc_sim.append(jaccard_similarity_qgram(list_ca[k], list_page[k],3))
                    lev_sim.append(1- (normalized_levenshtein.distance(list_ca[k], list_page[k])))
                    jaro_sim.append(jarowinkler.similarity(list_ca[k], list_page[k]))
                else:
                    jc_sim.append(-1)
                    lev_sim.append(-1)
                    jaro_sim.append(-1)
            else:
                jc_sim.append(-1)
                lev_sim.append(-1)
                jaro_sim.append(-1)
                
            k=k+1
        data[jac_columns[i]]=jc_sim
        data[lev_columns[i]]=lev_sim
        data[jaro_columns[i]]=jaro_sim
        i=i+1
        
    return data




    