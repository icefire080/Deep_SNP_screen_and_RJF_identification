# -*- coding: utf-8 -*- 

from collections import defaultdict, Counter
import pickle
from sklearn import model_selection
import argparse
from utils import build_snp_token_from_str, Snp_token
import pandas as pd


def read_sample_2_mit(mit_info_file):
    sample_2_mit = {}
    all_mit = []
    data = pd.read_csv(mit_info_file)
    for index, row in data.iterrows():
        sample_id = row["Sample_Name"]
        mit = row["Haplogroup"]
        mit_items = [ _.strip() for _ in  mit.strip().split(',')]
        sample_2_mit[sample_id] = mit_items
        all_mit.extend(mit_items)
    
    all_mit = set(all_mit)
    print('all_mit cnt {}, all_mit info {}'.format(len(all_mit), all_mit))
    return sample_2_mit


def read_sample_2_group(sample_2_group_file, breed_info=1):
    sample_id_2_group = {}
    group_l, breed_l = [], []
    for line_no, line in enumerate(open(sample_2_group_file)):
        items = line.strip().split('\t')
        group, breed = items[0], items[breed_info]
        breed = breed.strip()
        breed = breed.replace(' ','_')
        sample_id_2_group[items[-1]] = (group, breed)
        group_l.append(group)
        breed_l.append(breed)

    total_labeled_sample = line_no
    group_l = set(group_l)
    breed_l = set(breed_l)
    print('group cnt {}, breed cnt {}, group {}, breed {}'.format(len(group_l), len(breed_l), group_l, breed_l))
    return sample_id_2_group, total_labeled_sample


def parse_vcf_and_group(args):
    _parse_vcf_and_group(args.vcf_file, args.sample_2_group_file, args.data_file
                         , test_flag=0, mit_info_flag = 0, mit_info_file = args.mit_info_file, breed_info=args.breed_info, mock_group=args.mock_group  )

def parse_vcf_and_group_mit(args):
    _parse_vcf_and_group(args.vcf_file, args.sample_2_group_file
                            , args.data_file,test_flag=args.test_flag
                            , breed_info=args.breed_info
                            , mit_info_flag=1
                            , mit_info_file=args.mit_info_file
                            , mock_group=args.mock_group)

def _parse_vcf_and_group(vcf_file, sample_2_group_file, train_data_file, test_flag=0, mit_info_flag=0, mit_info_file=None, breed_info=1, mock_group = 0 ):
    
    print('begin parse_vcf_and_group ')
    sample_start_col = 9
    total_labeled_sample, sample_in_vcf, missing_group_sample = 0, 0, 0
    line_no_2_sample_info = defaultdict(dict)

    sample_id_2_mit = read_sample_2_mit(mit_info_file) if mit_info_flag > 0 else {}
    if mock_group > 0:
        sample_id_2_group = {}
    else:
        sample_id_2_group, total_labeled_sample = read_sample_2_group(sample_2_group_file, breed_info=breed_info)

    ALT_IDX = 4
    REF_IDX = 3
    for line_no, line in enumerate(open(vcf_file)):
        items = line.strip().split()
        if line.startswith('##'): continue

        if line.startswith('#'):
            print(line)
            for rel, sample_id in enumerate(items[sample_start_col:]) :
                col_no = rel + sample_start_col
                line_no_2_sample_info[col_no]['sample_id'] = sample_id
                line_no_2_sample_info[col_no]['desc'] = []
            sample_in_vcf = rel
            ##col-head
            continue
        for rel, sample in enumerate(items[sample_start_col:]) :
            col_no = rel + sample_start_col
            sample_info = sample.split(':')
            gt = sample_info[0]
            if len( items[ALT_IDX].split(',') ) > 1: continue
            new_snp_token = Snp_token(items[0], items[1],  items[ALT_IDX], items[REF_IDX], gt)
            line_no_2_sample_info[col_no]['desc'].append(str(new_snp_token))
        
        if line_no > 30 and test_flag:
            break
    #add_label
    unlabeld = []
    for k in line_no_2_sample_info:
        sample_id = line_no_2_sample_info[k]['sample_id']
        group_bread = sample_id_2_group.get(sample_id, ('0', '0'))
        group, breed = group_bread
        mit_info = sample_id_2_mit.get(sample_id, ['0'])

        if group in  {'GGS',  'RJF'} :
            pass
        elif group == 'hyb':
            group = '0'
        else:
            group = 'DC'

        line_no_2_sample_info[k]['group'] = group
        line_no_2_sample_info[k]['breed'] = breed
        line_no_2_sample_info[k]['mit'] = mit_info

        if group == '0':
            missing_group_sample += 1
            print('no_group info sample_id {}, col {}'.format(sample_id, k))
            unlabeld.append(k)
    # for k in unlabeld:
    #     line_no_2_sample_info.pop(k)
    
    print('total_labeled_sample ->{}, sample_in_vcf->{}, missing_group_sample->{}'.format(
        total_labeled_sample, sample_in_vcf, missing_group_sample
    ))

    pickle.dump(line_no_2_sample_info, open(train_data_file, 'wb'))

def split_train_test(args):
    reserve_test_dataset(args.data_file)

def split_spec_test(args):
    reverse = True if args.reverse > 0 else False
    reserve_spec_test_dataset(args.data_file, args.sample_id, reverse=reverse)

def split_spec_snp_test(args):
    reserve_spec_snp_test_dataset(args.data_file, args.snp_file)


def reserve_test_dataset(train_data_file):
    data = pickle.load(open(train_data_file, 'rb')) 
    l = [(k,v ) for k, v in data.items()]
    unlabeled = list(filter(lambda x: x[1]['group'] == 'u', l))
    labeled = list(filter(lambda x: x[1]['group'] != 'u', l))
    train, test = model_selection.train_test_split(labeled, test_size=0.15 )
    train_dict = {x[0]:x[1] for x in train}
    test_dict = {x[0]:x[1] for x in test}
    unlabeled_dict = {x[0]:x[1] for x in unlabeled}
    pickle.dump(train_dict, open(train_data_file+'.train', 'wb'))
    pickle.dump(test_dict, open(train_data_file+'.test', 'wb'))
    pickle.dump(unlabeled_dict, open(train_data_file+'.unlabeled', 'wb'))

def reserve_spec_test_dataset(train_data_file, spec_sample_id, reverse=False):
    data = pickle.load(open(train_data_file, 'rb')) 
    l = [(k,v ) for k, v in data.items()]
    unlabeled = list(filter(lambda x: x[1]['group'] == 'u', l))
    labeled = list(filter(lambda x: x[1]['group'] != 'u', l))
    spec_sample_id = [line.strip() for line in open(spec_sample_id)]
    spec_sample_id = set(spec_sample_id)
    if reverse:
        rule = lambda x : x not in spec_sample_id
    else:
        rule = lambda x : x in spec_sample_id
    
    test_dict = { x[0]:x[1] for x in labeled if rule(x[1]['sample_id']) }
    pickle.dump(test_dict, open(train_data_file+'.spec.test', 'wb'))

def reserve_spec_snp_test_dataset(train_data_file, snp_file):
    data = pickle.load(open(train_data_file, 'rb')) 
    #e.g. chrom:9,pos:12089977
    snp_str_set = set( [snp_str.strip() for snp_str in open(snp_file)] )
    # data_list = [(k,v ) for k, v in data.items()]
    for k, v in data.items():
        desc_new = []
        #e.g. 'alt:a,chrom:9,gt:0/1,pos:1208,ref:g'
        desc_old = v['desc']
        for snp_token_str in desc_old:
            snp_token = build_snp_token_from_str(snp_token_str)
            if snp_token.get_snp()  in snp_str_set:
                desc_new.append(str(snp_token))
        v['desc'] = desc_new
    pickle.dump(data, open(train_data_file+'.spec', 'wb'))


def stat_sample(args):
    data = pickle.load(open(args.data_file, 'rb')) 
    l = [ (k, v) for k, v in data.items()]
    vocabulary = []
    samples = {}
    labels_cnt, snp_cnt, breed_cnt = Counter(), Counter(), Counter()
    all_sample_id = set()
    snp_label_cnt = defaultdict(Counter)
    for n_doc, (k, doc) in enumerate(l):
        vocabulary.extend(doc['desc'])
        labels_cnt.update([doc['group']])
        breed_cnt.update([doc.get('breed', 'unknown')])
        all_sample_id.add(doc['sample_id'])
        for snp_token_str in doc['desc']:
            snp_token = build_snp_token_from_str(snp_token_str)
            if snp_token.gt != './.':
                snp_cnt.update([snp_token.get_snp()])
                snp_label_cnt[snp_token.get_snp()].update([doc['group']])
        if n_doc < 200:
            samples[k] = doc
    
    freq_2_snp_cnt = Counter()
    freq_2_snp_cnt.update( cnt for snp, cnt in snp_cnt.items() )

    print('n_doc {}, vocabulary size {}, label info {}, breed info {}, snp_cnt {}'.format(
        n_doc, len(vocabulary), labels_cnt
        , breed_cnt
        , len(snp_cnt.keys())
          
            ))
    pickle.dump(samples, open(args.output_file+'.samples', 'wb'))
    l = sorted([ (snp, cnt) for snp, cnt in snp_cnt.items() ], key=lambda x:x[1])
    with open(args.output_file+'.snp_dist', 'w') as fw:
        for snp, cnt in l:
            line = '{}\t{}\t{}'.format(snp, cnt, snp_label_cnt[snp] )
            fw.write(line + '\n')
    
    with open(args.output_file + '.all_sample_id', 'w') as fw:
        for one_id in all_sample_id:
            fw.write(one_id + '\n')

def stat_vcf(args):
    print('begin state vcf file')
    sample_start_col = 9
    samples, snp_tokens = [], set()
    #chrom:3,pos:45954943
    
    choosed_snp = set(line.strip() for line_no, line in enumerate(open(args.snp_file))) if args.snp_file else {}

    for line_no, line in enumerate(open(args.vcf_file)):
        items = line.strip().split()
        if line.startswith('##'): continue

        if line.startswith('#'):
            print(line)
            for rel, sample_id in enumerate(items[sample_start_col:]) :
                col_no = rel + sample_start_col
                samples.append(sample_id)
            ##col-head
            continue

        snp_str = 'chrom:{},pos:{}'.format(items[0],items[1]) 
        snp_tokens.add(snp_str)
    hit_cnt = len( choosed_snp & snp_tokens)
    print('demand snp cnt {}, test sample snp cnt {}, hit demand cnt {}'.format(
        len(choosed_snp), len(snp_tokens), hit_cnt))
    return snp_tokens

def stat_vcf_dump_snps(args):
    snp_tokens = stat_vcf(args)
    snp_tokens = list(snp_tokens)
    snp_tokens = sorted(snp_tokens)
    for snp_token in snp_tokens:
        print(snp_token)

def snp_diff(args):
    snp_file_1 = args.snp_file
    snp_file_2 = args.snp_file_2
    snp1 = set(line.strip() for line_no, line in enumerate(open(snp_file_1)))
    snp2 = set(line.strip() for line_no, line in enumerate(open(snp_file_2)))
    interset = snp1 & snp2
    union = snp1 | snp2
    diff = snp1 - snp2 
    print('snp_file_1 {}, snp_file_2 {}, interset {}, union {}'.format(
        len(snp1), len(snp2), len(interset), len(union)))
    print('diff {}'.format(diff))

def stat_rm_info(args):
    data_file = args.data_file
    stop_words_info = pickle.load(open(data_file, 'rb'))
    for k, v in stop_words_info.items():
        v = list(v)
        print('k {}, v {}\n'.format(k, v[:10]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='dg_classify'
    )
    
    parser.add_argument('--data_file', type=str, default='./dg_rdg/data_1009/GGS_VS_DC_6k_snp_all.pickle.valid')
    parser.add_argument('--output_file', type=str, default='./rjf_sample_feat_parse_default_output')

    parser.add_argument('--method', type=str, default= "split_spec_snp_test" )
    parser.add_argument('--vcf_file', type=str, default='' )
    parser.add_argument('--sample_2_group_file', type=str, default='' )
    parser.add_argument('--mit_info_file', type=str, default='' )
    parser.add_argument('--test_flag', type=int, default= 0 )
    parser.add_argument('--breed_info', type=int, default= 1)
    parser.add_argument('--snp_file', type=str, default= 'dg_rdg/data_1009/snp_tokens.remove_20p')
    parser.add_argument('--snp_file_2', type=str, default= '')
    parser.add_argument('--mock_group', type=int, default= 0)
    parser.add_argument('--sample_id', type=str, default='./dg_rdg/data_1009/train_sample_id')
    parser.add_argument('--reverse', type=int, default=1)



    args = parser.parse_args()
    print('in rjf_sample_feat_parse.py {}'.format(args))

    method_dict = {
        'stat': stat_sample
        ,'parse_vcf': parse_vcf_and_group
        ,'parse_vcf_mit': parse_vcf_and_group_mit
        ,'split_train_test': split_train_test
        ,'stat_vcf': stat_vcf
        ,'snp_diff': snp_diff
        ,'stat_rm_file': stat_rm_info
        ,'stat_vcf_dump_snps':stat_vcf_dump_snps
        ,'split_spec_test' : split_spec_test
        ,'split_spec_snp_test' : split_spec_snp_test
    }
    func = method_dict.get(args.method, lambda x: print('func not implemented'))
    func(args)

