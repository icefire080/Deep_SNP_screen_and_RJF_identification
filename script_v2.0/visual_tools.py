
import numpy as np
import matplotlib.pyplot as plt
import pickle
from utils import build_snp_token_from_str
import time 
import argparse
from sklearn.decomposition import PCA


def plot_snp(data, file, hline=0, symbol_size=2):
    y_1, x_1 = np.where(data==1)
    y_2, x_2 = np.where(data==2)
    y_3, x_3 = np.where(data==50)
    plt.figure(figsize=(14, 5))
    sc1 = plt.scatter(x= x_1, y=y_1 , s=symbol_size, color='tab:red', marker='o')
    sc2 = plt.scatter(x= x_2, y=y_2 , s=symbol_size, color='tab:blue', marker='o')
    sc3 = plt.scatter(x= x_3, y=y_3 , s=symbol_size, color='tab:blue', marker='o')
    if hline > 0:
        plt.hlines(hline+0.5, -1, 410, linestyles='dashed', colors='tab:brown')
    plt.vlines(330, 0, 45, linestyles='dashed', colors='tab:brown')
    plt.vlines(90, 0, 45, linestyles='dashed', colors='tab:brown')
    plt.legend((sc1,sc2, sc3), ['gt=0/0','gt=0/1', 'gt=1/1'], fontsize=6)
    plt.xlabel('SNP ID chr,pos', fontsize=6)
    plt.ylabel('sample ID', fontsize=6)
    plt.savefig(file, dpi=600)
    pass


def plot_sample_snp_dist(args):
    choosed_snp_list = [snp.strip() for snp in open(args.choosed_snp_file)]
    choosed_snp_list.sort()
    snp_index = { snp:idx for idx, snp in enumerate(choosed_snp_list)}
    gt_index = {'0/0':1, '0/1':2, '1/1':50}
    raw_data = pickle.load( open(args.input_file, 'rb') )
    data = []
    DC_cnt = 0
    for _, doc in raw_data.items():
        tokens = [build_snp_token_from_str(snp_token_str) for snp_token_str in doc['desc']]
        group = doc['group']
        one_matrix_line = np.zeros((1, len(choosed_snp_list)+1))
        
        DC_flag = 1 if group == 'DC' else 0
        if DC_cnt > 20 and DC_flag == 1: continue #prevent too much DC（domestic chicken）sample

        one_matrix_line[0,-1] = DC_flag 
        DC_cnt += DC_flag
        for snp in tokens:
            idx = snp_index.get(snp.get_snp(), -1)
            gt = gt_index.get(snp.gt, -1)
            if idx >= 0 and gt >=0:
                one_matrix_line[0, idx] = gt
        data.append(one_matrix_line)
    data.sort(key=lambda x:x[0, -1]) #horizon sort by label
    data_1 = np.concatenate(data)

    # vertical sort mv exchange snp pos
    result_2 = []
    for i in range(data_1.shape[-1]-1):
        sort_score = sum(data_1[DC_cnt:, i])
        result_2.append((data_1[:, i], sort_score))

    result_2.sort(key=lambda x: x[-1])
    data_2 = np.concatenate([x[0].reshape(-1,1) for x in result_2], axis=1)
    plot_snp(data_2, args.output_file + '_visual_code', hline=DC_cnt)
    plot_snp(data_1, args.output_file + '_visual_code_ori', hline=DC_cnt)

def pca_by_trained_model():
    pass 


def pca_from_sample_emb(args):
    result = pickle.load(open(args.input_file, 'rb'))
    print('all sample_cnt {}'.format(len(result)))
    data0 = [x[1].reshape((1, -1)) for x in result]
    breed_info = [x[-1] for x in result]
    all_breed = list(set(breed_info))
    all_breed.sort()
    breed_info = np.array(breed_info)
    data = np.concatenate(data0, axis=0)
    pca = PCA(n_components=4)
    data_dim4 = pca.fit_transform(data)
    weight = pca.explained_variance_ratio_
    print('explainability weight {}'.format(weight))
    
    plt.figure(figsize=(5, 5))
    x_1, y_1 = data_dim4[:,0], data_dim4[:,1] ## plot the largest component

    colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown']
    handles, breed_list = [], []
    for plt_color, breed in zip(colors, all_breed):
        breed_sample_idx = np.argwhere(breed_info == breed).reshape((-1))
        x_tmp, y_tmp = x_1[breed_sample_idx], y_1[breed_sample_idx]
        handle = plt.scatter(x= x_tmp, y=y_tmp , s=5, color=plt_color, marker='o')
        handles.append(handle)
        breed_list.append(breed)
    plt.legend(handles, breed_list, fontsize=6)
    plt.xlabel('SNP ID sort by chr,pos', fontsize=6)
    plt.ylabel('sample ID', fontsize=6)
    plt.savefig(args.input_file+'_pic', dpi=800)


def default_method(args):
    print('no this method')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='rjf_classify_visual_tools'
    )

    current_time_str = time.strftime("%Y%m%d_%H%M", time.localtime())
    parser.add_argument('--method', type=str, default='sample_snp_dist')
    parser.add_argument('--choosed_snp_file', type=str, default='' )
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--output_file', type=str, default='')
    args = parser.parse_args()
    all_methods = {
        'plot_sample_snp_dist': plot_sample_snp_dist
        ,'pca_from_sample_emb' : pca_from_sample_emb
    }
    func = all_methods.get(args.method, default_method)
    func(args)



