# %%
import urllib, urllib.request, xmltodict, feedparser
from datetime import datetime
from time import mktime
import pprint

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

# %%
def get_title_abs_arxiv(url):
    if 'arxiv' not in url: return -1,{}
    if 'pdf' in url: arxiv_id=''.join(url.split('pdf')[-2:-1])[1:-1]
    else: arxiv_id=url.split('/')[-1]
    query_url='http://export.arxiv.org/api/query?id_list='+arxiv_id
    data_dict = xmltodict.parse(urllib.request.urlopen(query_url).read())['feed']
    return {'title':data_dict['entry']['title'],'abstract':' '.join(data_dict['entry']['summary'].split('\n'))}

# %%
def fetch_daily_arxiv_papers(cat='cs.LG'):
    feedurl=f'https://rss.arxiv.org/rss/{cat}'
    feed=feedparser.parse(feedurl)
    datestr=datetime.fromtimestamp(mktime(feed['feed']['published_parsed'])).strftime('%d-%b-%Y')
    daily_papers=[]
    for e in feed['entries']:
        paper_entry={'id':e['id'],'title':e['title'],
                     'abstract':' '.join(e['summary'].split('\n')),
                     'date':datestr}
        daily_papers.append(paper_entry)
    return daily_papers

# # %%
# df = pd.read_csv('./Datasets/papers_of_interest/papers_of_interest.csv')
# df['url'][0]

# # %%
# get_title_abs_arxiv(df['url'][0])

# # %%
# from transformers import AutoTokenizer, AutoModel

# # load model and tokenizer
# tokenizer1 = AutoTokenizer.from_pretrained('bert-base-uncased')
# tokenizer2 = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
# tokenizer3 = AutoTokenizer.from_pretrained('allenai/specter')
# model1 = AutoModel.from_pretrained('bert-base-uncased')
# model2 = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
# model3 = AutoModel.from_pretrained('allenai/specter')

# papers = [ get_title_abs_arxiv(url) for url in df['url'] ]

# # concatenate title and abstract
# title_abs = [d['title'] + tokenizer.sep_token + (d.get('abstract') or '') for d in papers]
# # preprocess the input
# inputs1 = tokenizer1(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
# inputs2 = tokenizer2(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
# inputs3 = tokenizer3(title_abs, padding=True, truncation=True, return_tensors="pt", max_length=512)
# result1 = model1(**inputs1)
# result2 = model2(**inputs2)
# result3 = model3(**inputs3)
# # take the first token in the batch as the embedding
# embeddings1 = result1.last_hidden_state[:, 0, :]
# embeddings2 = result2.last_hidden_state[:, 0, :]
# embeddings3 = result3.last_hidden_state[:, 0, :]

# df_embeddings = pd.DataFrame({
#     'url' : df['url'].values.tolist(),
#     'bert_cat_embeddings': [torch.tensor(embedding) for embedding in embeddings1],
#     'scibert_cat_embeddings': [torch.tensor(embedding) for embedding in embeddings2],
#     'specter_cat_embeddings': [torch.tensor(embedding) for embedding in embeddings3]
# })

# df_embeddings.to_pickle("./Datasets/papers_of_interest/cat_embeddings.pkl")

# %%
# Load CSV file into a DataFrame
df = pd.read_csv('./Datasets/papers_of_interest/papers_of_interest.csv')
print(len(df))


# %%
from transformers import AutoTokenizer, AutoModel

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

models = ['bert', 'scibert', 'specter']
for model_num, model_name in enumerate(['bert-base-uncased', 'allenai/scibert_scivocab_uncased', 'allenai/specter']):

    print(f'{models[model_num]} starting')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    titleabs_embeddings_list = []

    for paper in range(df.shape[0]):
        # print(df['url'][paper])
        print(paper, end=' ')
        paper_data = get_title_abs_arxiv(df['url'][paper])
        title_abs = paper_data['title'] + tokenizer.sep_token + paper_data['abstract']
        
        inputs = tokenizer(title_abs, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        titleabs_embeddings_list.append(embeddings.detach().cpu())

    print()

    # Create DataFrame
    df_embeddings = pd.DataFrame({
        'url' : df['url'].values.tolist(),
        'titleabs_embeddings': [torch.tensor(embedding) for embedding in titleabs_embeddings_list],
    })

    df_embeddings.to_pickle(f"./Datasets/papers_of_interest/{models[model_num]}_cat_embeddings.pkl")
    print('Done')
