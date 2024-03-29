from functools import reduce
import itertools
from chinese_whispers import chinese_whispers, aggregate_clusters
import networkx as nx
from sklearn.metrics import adjusted_rand_score as ari
import seaborn as sns
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_similarity,cosine_distances



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
import logging

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('full_script.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(formatter)
# logger.addHandler(consoleHandler)




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






import torch
def cosine_distance(x1, x2=None, eps=1e-8):
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return (1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)).numpy()
def euclidean_distance(x1,x2):
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    return torch.dist(x, y, 2).numpy()
def manhattan_distance(x1,x2):
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    return torch.dist(x, y, 2).numpy()


if torch.cuda.is_available():
    dist_funcs=[euclidean_distance, manhattan_distance, cosine_distance]
else:
    dist_funcs=[euclidean_distances, manhattan_distances, cosine_distances]

dataset_configs={
    'distance_config':list(zip(dist_funcs,['euclidean_distances', 'manhattan_distances', 'cosine_distances'])),
    'level_similarities':[True,False],
    'normalize_embeds':[2,False],
    'k':[2,3,4,5,6,8,10,15],
}

chinese_whispers_configs={
    'weighting':['top','lin','log'],
    'iterations':[5,10,20,30,50],
    'model_name':['ChineseWhispers']
}

maxmax_configs = {
    'model_name':['maxmax']
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
    bert_out_1['bert_target_embs'] = bert_out_1.apply(find_target_embed,axis=1)
    print(f"dropped {bert_out_1['bert_target_embs'].isna().sum()} instances: {bert_out_1.loc[bert_out_1['bert_target_embs'].isna()].index}")
    train = train.loc[bert_out_1['bert_target_embs'].notna()]
    bert_out_1 = bert_out_1.loc[bert_out_1['bert_target_embs'].notna()]
    return train, bert_out_1


def find_marginal_distance(l1,l2, distance_func=cosine_similarity):
    b1=list(itertools.chain(*[x for x in l1]))
    ar1 = np.stack(b1)
    b2=list(itertools.chain(*[x for x in l2]))
    ar2 = np.stack(b2)
    dis = distance_func(ar1,ar2)
    
    
    # if normalize:
    #     dis = (dis-dis.min())/(dis.max()-dis.min())
    return dis.min()



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


def make_chinese_whispers(data, **kwargs):
    g = construct_weighted_graph(data)
    chinese_whispers(g, **kwargs)
    clusters = [g.nodes[i]['label'] for i in sorted(g.nodes)]
    return clusters

def maxmax_clustering(data, **kwargs):
    graph=construct_weighted_graph(data)

    surviving_edges=[]
    for i in graph.nodes:
        sorted_edges=sorted([[x[1],x[2]['weight']] for x in graph.out_edges(i,data=True)],key=lambda x: x[1],reverse=True)
        surviving_edges.append((sorted_edges[0][0], i))
    g = nx.from_edgelist(surviving_edges, create_using=nx.DiGraph)
    nx.set_node_attributes(g, {x:True for x in g.nodes}, 'root')
    for i in g.nodes:
        if g.nodes[i]['root']:
            nx.set_node_attributes(g, {x:False for x in nx.descendants(g, i)}, 'root')
    root_nodes = [x for x in g.nodes if g.nodes[x]['root']]
    clusters = {x:root for root in root_nodes for x in nx.descendants(g,root)}
    clusters.update({x:x for x in root_nodes})
    clusters_list= [y[1] for y in  sorted(list(clusters.items()),key = lambda x: x[0])]
    return clusters_list


def asyn_lpa_communities(data, **kwargs):
    g = construct_weighted_graph(data)
    clusters = nx.algorithms.community.label_propagation.asyn_lpa_communities(g, weight='weight')
    a=list(enumerate(clusters))
    b=list(itertools.chain(*[[ [x[0],i] for i in x[1] ]  for x in a ]))
    c = [y[0] for y in sorted(b, key=lambda x: x[1])]
    return c


def evaluate(pred, target):
    return ari(pred, target)


def make_dataset_from_target_word(train, bert_out, **kwargs):
    bert_target_embs = bert_out['bert_target_embs']
    datasets=[]
    distance, distance_name = kwargs['distance_config']
    for word in train['word'].unique():
        ix = np.array(train.loc[train['word']==word].index)
        btes = bert_target_embs.loc[ix]
        if kwargs['normalize_embeds']:
            data = (btes.apply(pd.Series).T/(btes.apply(pd.Series).applymap(lambda x: x**kwargs['normalize_embeds'])).sum(axis=1)).T
        else:
            data = btes.apply(pd.Series)
        t = pd.DataFrame(distance(data))
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
        
def make_dataset_from_minimal_context(train, bert_out, **kwargs):
    # bert_target_embs = bert_out['bert_target_embs']
    token_embeds = bert_out['token_embs']
    datasets=[]
    distance, distance_name = kwargs['distance_config']
    for word in train['word'].unique():
        ix = np.array(train.loc[train['word']==word].index)
        embs = token_embeds.loc[ix]

        data = pd.DataFrame(np.zeros((embs.shape[0],embs.shape[0],)))
        for i1 in range(embs.shape[0]):
            for i2 in range(embs.shape[0]):
                data.iloc[i1,i2]=find_marginal_distance(embs.iloc[i1],embs.iloc[i2], distance_func=distance)
        data = max(1,data.max().max())-data
        if kwargs['normalize_embeds']:
            data = (data.T/(data.applymap(lambda x: x**kwargs['normalize_embeds'])).sum(axis=1)).T
        t1=data.apply(find_knn, k=min(kwargs['k'], data.shape[0]-1),less_is_closer=False)
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




def run_training(model_configs, make_model, make_dataset):
    model_name=model_configs['model_name']
    results=[]
    bean_counter=0
    for ds_kind, dses in paths_config.items():
        for dataset_name,dataset_path  in dses.items():
            print(f'{dataset_name} started')
            train, bert_target_embs = read_data_and_compute_embeds(dataset_path)
            for dataset_config in product_dict(**dataset_configs):
                datasets = make_dataset(train, bert_target_embs, **dataset_config)
                for model_config in product_dict(**model_configs):
                    for i,d in enumerate(datasets):
                        labels = make_model(d['data']['data'], **model_config)
                        n_clusters=len(set(labels))
                        score = evaluate(labels, d['data']['target'])
                        r = {**d['config'], **model_config,'dataset_name':dataset_name, 'dataset_kind':ds_kind, 'word_id':i, 'n_clusters':n_clusters,'score':score}
                        results.append(r)
                        logger.debug(str(r))
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

    train_results_2 = train_results_1.groupby(list(dataset_configs.keys())+list(model_config.keys())+['dataset_name','dataset_kind']).agg({'n_clusters':'mean','weighted_score':'sum','contexts_num':'mean'})
    with open(f'./{model_name}_aggregated_training_dump.pkl', 'wb') as f:
        pkl.dump(train_results_2, f)

if __name__=='__main__':
    run_training({'model_name':['maxmax_min_context']}, maxmax_clustering, make_dataset_from_minimal_context)    
    run_training({'model_name':['maxmax_target_word']}, maxmax_clustering, make_dataset_from_target_word)    
    run_training({'model_name':['label_prop_min_context']}, asyn_lpa_communities,make_dataset_from_minimal_context)
    run_training({'model_name':['label_prop_target_word']}, asyn_lpa_communities,make_dataset_from_target_word)