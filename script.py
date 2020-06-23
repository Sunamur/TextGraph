from functools import reduce
import itertools
from chinese_whispers import chinese_whispers, aggregate_clusters
import networkx as nx
from sklearn.metrics import adjusted_rand_score as ari
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity



import pickle as pkl

import tqdm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import GridSearchCV
from deeppavlov.core.common.file import read_json
from deeppavlov import build_model, configs
import stopwords
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from nltk import word_tokenize
import re
from tqdm import tqdm, tqdm_pandas
from maxmax import maxmax_clustering    


tqdm_pandas(tqdm())
russian_stopwords = stopwords.get_stopwords("russian")
punctuation=[',','.',';',':','!','?','–','-']
def remove_target_word(context, positions):
    positions = [i.split('-') for i in positions.split(',')]
    for position in positions:
        start = int(position[0])
        end = int(position[1])
        return context.replace(context[start:end], '')



def bert_preprocess(text):
    result = []
    for sentence in text.split('.'):
        tokens = [re.sub('[^А-Яа-я.!? ]', ' ', token) for token in word_tokenize(sentence) if token not in russian_stopwords\
              and token != " " and token != "" \
              and token.strip() not in punctuation]
        sentence = ' '.join(tokens)
        result.append(sentence)
    return [x for x in result if x]
def elmo_preprocess(text):
    result = []
    for sentence in text.split('.'):
        tokens = [re.sub('[^А-Яа-я.!? ]', ' ', token) for token in word_tokenize(sentence) if token not in russian_stopwords\
              and token != " " and token != "" \
              and token.strip() not in punctuation]
        result.append(tokens)
    return [x for x in result if x]

bert_config = read_json(configs.embedder.bert_embedder)
bert_config['metadata']['download'][0]['url'] = 'http://files.deeppavlov.ai/deeppavlov_data/bert/rubert_cased_L-12_H-768_A-12_pt.tar.gz'
bert_config['metadata']['variables']['BERT_PATH'] = '{DOWNLOADS_PATH}/bert_models/rubert_cased_L-12_H-768_A-12_pt'
bert = build_model(bert_config, download=True)

from deeppavlov.models.embedders.elmo_embedder import ELMoEmbedder
elmo = ELMoEmbedder("http://files.deeppavlov.ai/deeppavlov_data/elmo_ru-news_wmt11-16_1.5M_steps.tar.gz",elmo_output_names=["elmo"])

def extract_token_vectors(bert_context_vectors, context, positions):
    words = []
    vectors = []
    positions = [i.split('-') for i in positions.split(',')]
    for position in positions:
        start = int(position[0])
        end = int(position[1])
        words.append(re.sub('[^А-Яа-я ]', ' ',context[start:end]).strip())
    for word in words:
        for list_idx, token_list in enumerate(bert_context_vectors[0]):
            if word in token_list:
                word_idx = token_list.index(word)
                bert_token_vector = bert_context_vectors[1][list_idx][word_idx]
                vectors.append(bert_token_vector)
    try:
        result = np.vstack(vectors)
    except ValueError:
        result = f'{len(vectors)}, {words}'
    return result
def stack_vectors(bert_embeddings):
    vectors = []
    for i, vector in enumerate(bert_embeddings):
        vectors.append(vector[0])
        
    X = np.vstack(vectors)
    return X
def average_pooler_outputs_mean(vectors):
    """ Function to average BERT pooler outputs """
    average_vector = np.array([sum(subvector) / len(subvector) for subvector in vectors[6].transpose()])
    return average_vector
def elmo_vectorize(elmo_preprocessed_text):
    return elmo(elmo_preprocessed_text)
def bert_vectorize(bert_preprocessed_text):
    return(bert(bert_preprocessed_text))



def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

dataset_configs={
    'distance_config':list(zip([euclidean_distances, manhattan_distances, cosine_similarity],['euclidean_distances', 'manhattan_distances', 'cosine_similarity'], [True,True,False])),
    'level_similarities':[True,False],
    'normalize_embeds':[2,False],
    # 'k':np.arange(2,25),
    'k':[2,3,4,5,6,8,10,15],
}

chinese_whispers_configs={
    'weighting':['top','lin','log'],
    'iterations':[5,10,20,30,50],
    'algorithm':'ChineseWhispers'
}

maxmax_configs = {
    'algorithm':'ChineseWhispers'
}

def clear_punctuation(x):
    re.sub(r'[^A-ZА-Яа-яa-z ]')


def find_target_embed(x):
    try:
        l=list(itertools.chain(*x['tokens']))
        target = list(filter(lambda y: x['target_word'] in y, l))
        ix = l.index(target[0])
        return np.concatenate(x['token_embs'])[ix]
    except:
        return np.nan




clear_punct = lambda x:re.sub(r'[^A-ZА-Яа-яa-z ]','',x)

def read_data_and_compute_embeds(path_to_train):
    train = pd.read_csv(path_to_train, sep='\t')
    train=train.dropna(subset=['positions']).reset_index(drop=True)
    train['positions_int'] = train.apply(lambda x:   [[int(z) for z in y.split('-')] for y in x['positions'].split(',')]  ,axis=1)
    train['target_word'] = train.apply(lambda x: [clear_punct(x['context'][a:b]).strip() for a,b in x['positions_int']], axis=1)
    bert_preproc = train.dropna(subset=['positions']).loc[:,'context'].apply(bert_preprocess)
    bert_preproc = bert_preproc.loc[bert_preproc.apply(len)!=0]
    bert_tokens = bert_preproc.progress_apply(bert_vectorize)
    bert_out = bert_tokens.apply(pd.Series)
    bert_out.columns=['tokens', 'token_embs', 'subtokens', 'subtoken_embs', 'sent_max_embs', 'sent_mean_embs', 'bert_pooler_outputs']
    assert bert_tokens.shape[0]==train.shape[0]
    bert_out_1 = pd.concat([bert_out,train['target_word'].apply(lambda x: x[0])],axis=1)
    bert_target_embs = bert_out_1.apply(find_target_embed,axis=1)
    
    print(f'dropped {bert_target_embs.isna().sum()} instances: {bert_target_embs[bert_target_embs.isna()].index}')

    train = train.loc[bert_target_embs.notna()]
    bert_target_embs = bert_target_embs.loc[bert_target_embs.notna()]

    return train, bert_target_embs





def find_knn(x,k=5,less_is_closer=False):
    y=x.copy()
    thres = x.sort_values(ascending=less_is_closer).iloc[k-int(less_is_closer)]
    y[y<thres]=0
    # y[y>=thres]=1
    return y

def construct_weighted_graph(data):
    g = nx.from_edgelist(data.loc[:,['variable','index']].values,create_using=nx.DiGraph)
    nx.set_edge_attributes(G=g, name='weight', values={(x[1],x[0]): x[2] for x in data.values})
    return g


def make_chinese_whispers(data, **chinese_args):
    g = construct_weighted_graph(data)
    chinese_whispers(g, **chinese_args)
    clusters = [g.nodes[i]['label'] for i in sorted(g.nodes)]
    return clusters

def evaluate(pred, target):
    return ari(pred, target)


def make_dataset(train, bert_target_embs, **kwargs):
    datasets=[]
    distance, distance_name, less_is_closer = kwargs['distance_config']
    for word in train['word'].unique():
        ix = np.array(train.loc[train['word']==word].index)
        btes = bert_target_embs.loc[ix]
        if kwargs['normalize_embeds']:
            data = (btes.apply(pd.Series).T/(btes.apply(pd.Series).applymap(lambda x: x**kwargs['normalize_embeds'])).sum(axis=1)).T
        else:
            data = btes.apply(pd.Series)
        t = pd.DataFrame(distance(data))
        if less_is_closer:
            t=(-t + t.max().max())/t.max().max()
        t1=t.apply(find_knn, k=min(kwargs['k'], t.shape[0]-1),less_is_closer=False)
        t1=t1.reset_index().melt('index')
        if kwargs['level_similarities']:
            t1.loc[t1['value']>0,'value']=1
        t2=t1.loc[(t1['value']>0)&(t1['index']!=t1['variable'])]
        target=train.loc[ix,'gold_sense_id'].reset_index(drop=True)
        dataset={
            'data':{
                'data':t2,
                'target':target
                },
             'config':{
                'distance_name':distance_name,
                'level_similarities':kwargs['level_similarities'],
                'normalize_embeds':kwargs['normalize_embeds'],
                'k':kwargs['k']
                }
            }
        datasets.append(dataset)
    return datasets
        


# train_paths = ['/home/lsherstyuk/Documents/HSE_FTIAD/KR1/WSID_data/RUSSE2018/russe-wsi-kit/data/main/active-dict/train.csv',
# '/home/lsherstyuk/Documents/HSE_FTIAD/KR1/WSID_data/RUSSE2018/russe-wsi-kit/data/main/bts-rnc/train.csv',
# '/home/lsherstyuk/Documents/HSE_FTIAD/KR1/WSID_data/RUSSE2018/russe-wsi-kit/data/main/wiki-wiki/train.csv']
# test_paths = ['/home/lsherstyuk/Documents/HSE_FTIAD/KR1/WSID_data/RUSSE2018/russe-wsi-kit/data/main/active-dict/test-solution.csv',
# '/home/lsherstyuk/Documents/HSE_FTIAD/KR1/WSID_data/RUSSE2018/russe-wsi-kit/data/main/bts-rnc/test-solution.csv',
# '/home/lsherstyuk/Documents/HSE_FTIAD/KR1/WSID_data/RUSSE2018/russe-wsi-kit/data/main/wiki-wiki/test-solution.csv']

# dataset_names = ['active-dict','bts-rnc','wiki-wiki',]

paths_config={
    'test':{
        'active-dict':'./../RUSSE2018/russe-wsi-kit/data/main/active-dict/test-solution.csv',
        'bts-rnc':'./../RUSSE2018/russe-wsi-kit/data/main/bts-rnc/test-solution.csv',
        'wiki-wiki':'./../RUSSE2018/russe-wsi-kit/data/main/wiki-wiki/test-solution.csv',
    },
    'train':{
        'active-dict':'./../RUSSE2018/russe-wsi-kit/data/main/active-dict/train.csv',
        'bts-rnc':'./../RUSSE2018/russe-wsi-kit/data/main/bts-rnc/train.csv',
        'wiki-wiki':'./../RUSSE2018/russe-wsi-kit/data/main/wiki-wiki/train.csv',
    },
}

def run_chinese():
    results=[]
    bean_counter=0
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
            print(f'{dataset_name} started')
            train, bert_target_embs = read_data_and_compute_embeds(dataset_path)
            for dataset_config in product_dict(**dataset_configs):
                datasets = make_dataset(train, bert_target_embs, **dataset_config)
                for chinese_config in product_dict(**chinese_whispers_configs):
                    for i,d in enumerate(datasets):
                        labels = make_chinese_whispers(d['data']['data'], **chinese_config)
                        n_clusters=len(set(labels))
                        score = evaluate(labels, d['data']['target'])
                        r = {**d['config'], **chinese_config,'dataset_name':dataset_name, 'dataset_kind':ds_kind, 'word_id':i, 'n_clusters':n_clusters,'score':score}
                        results.append(r)
                        if (bean_counter%5000)==0:
                            print(bean_counter)
                        if (bean_counter%50000)==0:
                            with open('./train_dump.pkl', 'wb') as f:
                                pkl.dump(results, f)
                        bean_counter+=1
    with open('./train_dump.pkl', 'wb') as f:
        pkl.dump(results, f)
    train_results = pd.DataFrame(results)
    train_results.to_pickle('train_dump_df.pkl')
    word_ids=[]
    dataset_names = ['active-dict','bts-rnc','wiki-wiki',]
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
    # for ds_path, ds_name in zip(train_paths,dataset_names):
            testing_df = pd.read_csv(dataset_path,sep='\t')
            testing_df['dummy']=1
            word_id_df = testing_df['word'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'word_id'})
            word_id_df['dataset_name']=dataset_name
            word_id_df['dataset_kind']=ds_kind
            word_id_df=word_id_df.join(testing_df.groupby('word').agg({'dummy':['sum',lambda x: sum(x)/testing_df.shape[0]]}), on='word')
            word_id_df.columns = ['word_id','word','dataset_name','dataset_kind','contexts_num','contexts_frac']
            word_ids.append(word_id_df)
    word_ids=pd.concat(word_ids,axis=0).reset_index(drop=True)
    train_results_1=pd.merge(
        train_results,
        word_ids,
        left_on=['word_id','dataset_name','dataset_kind'],
        right_on=['word_id','dataset_name','dataset_kind'],
        how='left')
    train_results_1['weighted_score'] = train_results_1['score']*train_results_1['contexts_frac']
    train_results_2 = train_results_1.groupby(['distance_name','level_similarities','normalize_embeds','k','weighting','iterations','dataset_name','dataset_kind']).agg({'n_clusters':'mean','weighted_score':'sum','contexts_num':'mean'})
    with open('./aggregated_training_dump.pkl', 'wb') as f:
        pkl.dump(train_results_2, f)


def run_maxmax():
    results=[]
    bean_counter=0
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
            print(f'{dataset_name} started')
            train, bert_target_embs = read_data_and_compute_embeds(dataset_path)
            for dataset_config in  product_dict(**dataset_configs):
                datasets = make_dataset(train, bert_target_embs, **dataset_config)
                for i,d in enumerate(datasets):
                    g = construct_weighted_graph(d['data']['data'])
                    labels = maxmax_clustering(g)
                    n_clusters=len(set(labels))
                    score = evaluate(labels, d['data']['target'])
                    r = {**d['config'],'dataset_name':dataset_name, 'dataset_kind':ds_kind, 'word_id':i, 'n_clusters':n_clusters,'score':score}
                    results.append(r)
                    if (bean_counter%5000)==0:
                        print(bean_counter)
                    if (bean_counter%50000)==0:
                        with open('./train_maxmax_dump.pkl', 'wb') as f:
                            pkl.dump(results, f)
                    bean_counter+=1
    with open('./train_maxmax_dump.pkl', 'wb') as f:
        pkl.dump(results, f)
    train_results = pd.DataFrame(results)
    train_results.to_pickle('train_maxmax_dump_df.pkl')
    word_ids=[]
    dataset_names = ['active-dict','bts-rnc','wiki-wiki',]
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
    # for ds_path, ds_name in zip(train_paths,dataset_names):
            testing_df = pd.read_csv(dataset_path,sep='\t')
            testing_df['dummy']=1
            word_id_df = testing_df['word'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'word_id'})
            word_id_df['dataset_name']=dataset_name
            word_id_df['dataset_kind']=ds_kind
            word_id_df=word_id_df.join(testing_df.groupby('word').agg({'dummy':['sum',lambda x: sum(x)/testing_df.shape[0]]}), on='word')
            word_id_df.columns = ['word_id','word','dataset_name','dataset_kind','contexts_num','contexts_frac']
            word_ids.append(word_id_df)
    word_ids=pd.concat(word_ids,axis=0).reset_index(drop=True)
    train_results_1=pd.merge(
        train_results,
        word_ids,
        left_on=['word_id','dataset_name','dataset_kind'],
        right_on=['word_id','dataset_name','dataset_kind'],
        how='left')
    train_results_1['weighted_score'] = train_results_1['score']*train_results_1['contexts_frac']
    train_results_2 = train_results_1.groupby(['distance_name','level_similarities','normalize_embeds','k','dataset_name','dataset_kind']).agg({'n_clusters':'mean','weighted_score':'sum','contexts_num':'mean'})
    with open('./aggregated_maxmax_training_dump.pkl', 'wb') as f:
        pkl.dump(train_results_2, f)


if __name__=='__main__':
    run_maxmax()    




def run_training(model_configs, make_model):
    model_name=model_config['model_name']
    results=[]
    bean_counter=0
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
            print(f'{dataset_name} started')
            train, bert_target_embs = read_data_and_compute_embeds(dataset_path)
            for dataset_config in product_dict(**dataset_configs):
                datasets = make_dataset(train, bert_target_embs, **dataset_config)
                for model_config in product_dict(**model_config):
                    for i,d in enumerate(datasets):
                        labels = make_model(d['data']['data'], **model_config)
                        n_clusters=len(set(labels))
                        score = evaluate(labels, d['data']['target'])
                        r = {**d['config'], **model_config,'dataset_name':dataset_name, 'dataset_kind':ds_kind, 'word_id':i, 'n_clusters':n_clusters,'score':score}
                        results.append(r)
                        if (bean_counter%5000)==0:
                            print(bean_counter)
                        if (bean_counter%50000)==0:
                            with open('./train_dump.pkl', 'wb') as f:
                                pkl.dump(results, f)
                        bean_counter+=1
    with open(f'./{model_name}_train_dump.pkl', 'wb') as f:
        pkl.dump(results, f)
    train_results = pd.DataFrame(results)
    train_results.to_pickle(f'./{model_name}_train_dump_df.pkl')
    word_ids=[]
    dataset_names = ['active-dict','bts-rnc','wiki-wiki',]
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
    # for ds_path, ds_name in zip(train_paths,dataset_names):
            testing_df = pd.read_csv(dataset_path,sep='\t')
            testing_df['dummy']=1
            word_id_df = testing_df['word'].drop_duplicates().reset_index(drop=True).reset_index().rename(columns={'index':'word_id'})
            word_id_df['dataset_name']=dataset_name
            word_id_df['dataset_kind']=ds_kind
            word_id_df=word_id_df.join(testing_df.groupby('word').agg({'dummy':['sum',lambda x: sum(x)/testing_df.shape[0]]}), on='word')
            word_id_df.columns = ['word_id','word','dataset_name','dataset_kind','contexts_num','contexts_frac']
            word_ids.append(word_id_df)
    word_ids=pd.concat(word_ids,axis=0).reset_index(drop=True)
    train_results_1=pd.merge(
        train_results,
        word_ids,
        left_on=['word_id','dataset_name','dataset_kind'],
        right_on=['word_id','dataset_name','dataset_kind'],
        how='left')
    train_results_1['weighted_score'] = train_results_1['score']*train_results_1['contexts_frac']

    train_results_2 = train_results_1.groupby(['distance_name','level_similarities','normalize_embeds']+list(model_config.keys())+['dataset_name','dataset_kind']).agg({'n_clusters':'mean','weighted_score':'sum','contexts_num':'mean'})
    with open(f'./{model_name}_aggregated_training_dump.pkl', 'wb') as f:
        pkl.dump(train_results_2, f)

if __name__=='__main__':
    run_training(maxmax_configs, maxmax_clustering)