import torch
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import numpy as np
import logging
import torch.nn.functional as F
import numpy as np
from utils import build_snp_token_from_str
from torch.utils.data.sampler import WeightedRandomSampler

class CustomDataset():
    def __init__(self, pickle_file, vocabulary_file = None
                 , test_flag = False, stop_words_file = None
                 , stop_words_list = None
                 , filter_breed = ''
                 , mit_flag = 0, oversample_ratio = 1.0) -> None:
        logging.info('init CustomDataset vocabulary_file {}, filter_breed {}'.format(vocabulary_file, filter_breed))
        self.data = pickle.load(open(pickle_file, 'rb')) 
        self.test_flag = test_flag
        
        stop_words = build_stop_words_list(stop_words_file, stop_words_list)
        
        self.sample_ids, self.corpus, self.label, self.breed_info, self.mit_info, addition = parse_data(self.data, filter_breed, mit_flag)
        if not test_flag:
            self.vectorizer = CountVectorizer(token_pattern=r'(?u)\b[\w\\,\\:\\/]+\b', stop_words=stop_words)
            self.x = self.vectorizer.fit_transform(self.corpus)
            self.token_cnt = len(self.vectorizer.vocabulary_)
        else:
            logging.info('restore vector from {}'.format(vocabulary_file))
            self.vectorizer = pickle.load(open(vocabulary_file, 'rb'))
            self.x = self.vectorizer.transform(self.corpus)
            self.token_cnt = len(self.vectorizer.vocabulary_)

        self.label_enc = OrdinalEncoder()
        self.y = self.label_enc.fit_transform(self.label.reshape(-1, 1))
        x_max_len = self.x.shape[-1]
        self.max_len = int(x_max_len * 1.3)
        
        ##below are desc info
        stat_info(self, mit_token_sample=addition['mit_token_sample'])
        ###------------------------------------------------------------------------------------------------------
        self.oversample_ratio = oversample_ratio
        if not test_flag and oversample_ratio > 1.0:
            self._build_balanced_sampler()

    def _build_balanced_sampler(self):
        from torch.utils.data.sampler import WeightedRandomSampler
        import numpy as np

        y_np = self.y.flatten()
        class_counts = np.bincount(y_np)
        minority_class = 1 if class_counts[1] < class_counts[0] else 0
        
        sample_weights = np.ones(len(y_np))
        sample_weights[y_np == minority_class] = self.oversample_ratio
        
        self.sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),  
            replacement=True  
        )
        logging.info(f"Applied oversampling: minority_class={minority_class}, ratio={self.oversample_ratio}")


    def __getitem__(self, idx):
        feat = self.x[idx]
        feat_t = torch.from_numpy(feat.indices)
        feat_t = torch.cat([feat_t, torch.zeros(self.max_len - len(feat_t), dtype=torch.int64)], dim=0)
        label = self.y [idx]
        label = torch.from_numpy(label)
        label = label.type(torch.long)
        return feat_t, label
    
    
    def __len__(self):
        return len(self.corpus)
    
    def save_vocabulary(self, file):
        with open(file, 'wb') as f:
            pickle.dump(self.vectorizer, f)

def parse_data(data, filter_breed, mit_flag):
    docs, sample_id, doc_labels, mit_info, breed_info = [], [], [], [], [] ## 5 cols
    filter_breed_set, all_breed_set = set(), set()
    if filter_breed != '':
        filter_breed_set = set(breed for breed in filter_breed.split(',') )

    mit_token_sample = []
    for one_sample in data.values():
        breed = one_sample.get('breed', 'u')
        all_breed_set.add(breed)

        if filter_breed_set and  breed in filter_breed_set : continue ## remove some breed data
        
    
        doc = []
        for snp_token_str in one_sample['desc']:
            snp_token = build_snp_token_from_str(snp_token_str)
            if snp_token.gt != './.':
                doc.append(str(snp_token))
        
        mit_in_one_sample = one_sample.get('mit', [])
        if mit_flag > 0:
            mit_tokens = ['mit_'+mit_token_str for mit_token_str in mit_in_one_sample]
            doc.extend(mit_tokens)
            mit_token_sample.extend(mit_tokens)    
                
        
        docs.append(' '.join(doc))
        sample_id.append(one_sample['sample_id'])
        doc_labels.append(one_sample['group'])
        breed_info.append(breed)
        mit_info.append( '_'.join(mit_in_one_sample) if len(mit_in_one_sample) > 0 else '0')

    corpus = np.array(docs)
    sample_ids = np.array(sample_id)
    label = np.array(doc_labels)
    breed_info = np.array(breed_info)
    mit_info = np.array(mit_info)
    return sample_ids, corpus, label, breed_info, mit_info, {'mit_token_sample':mit_token_sample, 'all_breed_set': all_breed_set} 


def stat_info(ds, mit_token_sample=None):
    reverse_corpus = ds.vectorizer.inverse_transform(ds.x)
    all_snp_after_stop_words = []
    stop_words_cnt = 0 
    doc_token_cnt = []
    for doc in reverse_corpus:
        doc_token_cnt.append(len(doc))
        for snp_token_str in doc:
            if snp_token_str.startswith('mit'): continue
            snp_token = build_snp_token_from_str(snp_token_str)
            all_snp_after_stop_words.append(snp_token.get_snp())
    logging.info('reverse_corpus avg-len {}, all_snp_after_stop_words cnt {}'.format(np.average(doc_token_cnt), len(set(all_snp_after_stop_words))))

        
    if ds.vectorizer.stop_words is not None:
        stop_words_cnt = len(ds.vectorizer.stop_words)
    logging.info('stop words count {}'.format(stop_words_cnt))

    sample_size = ds.x.shape[0]
    all_token_cnt = len(ds.x.data)
    a_token_cnt = all_token_cnt/sample_size
    logging.info('self.vectorizer.token_cnt {}'.format(ds.token_cnt))
    logging.info('self.x shape {}, docs in x {}, avg_token_cnt'.format(ds.x.shape, sample_size, a_token_cnt))
    logging.info('mit_token info cnt {}, sample {}'.format(len(mit_token_sample), mit_token_sample[:10]))

class SimpleDataset():
    def __init__(self, x, y, max_len=None) -> None:
        self.x = x
        self.y = y
        if max_len is not None:
            self.max_len = max_len
        else:
            x_max_len = self.x.shape[-1]
            self.max_len = int(x_max_len * 1.3)

    def __getitem__(self, idx):
        feat = self.x[idx]
        feat_t = torch.from_numpy(feat.indices)
        feat_t = torch.cat([feat_t, torch.zeros(self.max_len - len(feat_t), dtype=torch.int64)], dim=0)
        label = self.y[idx]
        label = torch.from_numpy(label)
        label = label.type(torch.long)
        return feat_t, label
    
    def __len__(self):
        return len(self.y.squeeze())
    

def build_stop_words_list(stop_words_file, stop_words_list):
    stop_words = None 
    if stop_words_file is not None:
        stop_words_0 = pickle.load(open(stop_words_file, 'rb'))
        stop_words = filter(lambda x : len(x.split(',')) == 5, list(stop_words_0['snp_rm']) ) 
        stop_words = list(stop_words)
        logging.info('init custom_dataset with stop_words_file size {}'.format(len(stop_words)))


    if stop_words_list is not None:
        stop_words = filter(lambda x : len(x.split(',')) == 5, stop_words_list ) 
        stop_words = list(stop_words)
        logging.info('init custom_dataset with stop_words_list size {}'.format(len(stop_words)))
    return stop_words
