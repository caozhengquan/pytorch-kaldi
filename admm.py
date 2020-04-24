from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from numpy import linalg as LA
import datetime
# from tensorboardX import SummaryWriter
import scipy.misc
# import torchsnooper
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from distutils.util import strtobool

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class ADMM:
    def __init__(self, config, model):
        self.ADMM_U = {}
        self.ADMM_Z = {}
        self.ADMM_alpha = {}  # used for quantization only
        self.ADMM_Q = {}  # used for quantization only
        self.model = model
        self.rhos = {}
        self.prune_ratios = {}  # code name -> prune ratio
        self.cross = {}
        self.init(config, model)

    def init(self, config, model):
        """
        Args:
            config: configuration file that has settings for prune ratios, rhos
        called by ADMM constructor. config should be a .yaml file

        """
        prune_ratios = list(map(float, config['exp']['prune_ratios'].split(',')))
        rho = list(map(float, config['exp']['rho'].split(',')))
        self.sparsity_type = config['exp']['sparsity_type']
        cross = [(4,1), (4,1), (4,1), (4,1), (4,1), (4,1),(4,1),(4,1), (4,1), (4,1), (4,1),(4,1)]

        i = 0
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if  ('bn' not in name) and ('ln' not in name):
                    print(name)
                    self.ADMM_U[name] = torch.zeros(W.shape).cuda()  # add U
                    self.ADMM_Z[name] = torch.Tensor(W.shape).cuda()  # add Z
                    self.prune_ratios[name] = prune_ratios[int(name[3])]
                    self.cross[name] = cross[i]
                    self.rhos[name] = rho[int(name[3])]
                    i +=1
            break


def random_pruning(config, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (config.sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")




def mask_block_max(weight, mask, percent):
    # print("pattern list = ", pattern_list)
    # print("weight shape = ", weight.shape)
    vecs, vec_len = np.shape(mask)
    keep_ratio = 100 - percent
    # blocks = int(vec_len*keep_ratio)
    block_len = (np.rint(100 / keep_ratio)).astype(np.int)   #感觉是每5个里面有一个；
    blocks = int(vec_len / block_len)
    if block_len * blocks < vec_len:        #如果乘积小于总长，那么blocks的个数再加一个。
        blocks += 1
    for i in range(vecs):
        for j in range(blocks):
            if j == blocks - 1:
                cur_block = np.abs(weight[i, j * block_len:])
            else:
                cur_block = np.abs(weight[i, j * block_len:(j + 1) * block_len])
            max_position = np.argmax(cur_block)     #返回当前block下最大数的索引
            mask[i, max_position + j * block_len] = 1

    weight *= mask


def pattern_sort(mask, index, active_len, sum_ratio):
    if active_len < 32:
        return index

    # split
    vecs, vec_len = np.shape(mask)
    if vecs < 4:
        return index
    half_vecs = int(vecs / 2)
    up_msk = mask[:half_vecs]
    up_indx = index[:half_vecs]
    low_msk = mask[half_vecs:]
    low_indx = index[half_vecs:]

    active_len = int(active_len * sum_ratio)

    up_density = np.sum(up_msk[:, :active_len], axis=1)
    up_indx_indx = np.argsort(up_density)
    up_sorted_indx = up_indx[up_indx_indx]
    up_sorted_msk = up_msk[up_indx_indx]
    up_final_indx = pattern_sort(up_sorted_msk, up_sorted_indx, active_len, sum_ratio)

    low_density = np.sum(low_msk[:, :active_len], axis=1)
    low_indx_indx = np.argsort(low_density)
    low_sorted_indx = low_indx[low_indx_indx]
    low_sorted_msk = low_msk[low_indx_indx]
    low_final_indx = pattern_sort(low_sorted_msk, low_sorted_indx, active_len, sum_ratio)

    return np.concatenate((up_final_indx, low_final_indx))



def Hamming_Distance_Matrix(mask):
    vecs, vec_len = np.shape(mask)
    hamdis_matrix = np.zeros((vecs, vecs))
    # hamdis_matrix = np.zeros((vec_len, vec_len))
    # mask = mask.astype(np.int)
    for i in range(vecs - 1):
        for j in range(i + 1, vecs):
            # for i in range(vec_len-1):
            #     for j in range(i+1, vec_len):
            # hamdis_matrix[i,j] = LA.norm(mask[i]-mask[j],2)
            # hamdis_matrix[i,j] = np.sum(1-np.bitwise_and(mask[i], mask[j]))
            # hamdis_matrix[i,j] = np.sum(np.bitwise_xor(mask[i], mask[j]))
            # cur_mask_i = np.reshape(mask[i], (1, -1))
            # cur_mask_j = np.reshape(mask[j], (1, -1))
            # dis_temp = cosine_similarity(cur_mask_i, cur_mask_j)
            # hamdis_matrix[i,j] += dis_temp
            cur_mask_i = np.abs(mask[i])
            cur_mask_j = np.abs(mask[j])
            min_i = np.min(cur_mask_i)
            max_i = np.max(cur_mask_i)
            min_j = np.min(cur_mask_j)
            max_j = np.max(cur_mask_j)
            cur_mask_i /= (max_i - min_i)
            cur_mask_j /= (max_j - min_j)

            # mean_i = np.mean(cur_mask_i)
            # mean_j = np.mean(cur_mask_j)
            # std_i = np.std(cur_mask_i)
            # std_j = np.std(cur_mask_j)
            # cur_mask_i = (cur_mask_i - mean_i) / std_i
            # cur_mask_j = (cur_mask_j - mean_j) / std_j
            hamdis_matrix[i, j] += LA.norm(cur_mask_i - cur_mask_j, 2)
    low_triangle = hamdis_matrix.transpose()
    hamdis_matrix = low_triangle + hamdis_matrix
    return hamdis_matrix





def mask_balanced_block_max(weight, block_len, block_types, percent):
    vecs, vec_len = np.shape(weight)
    print('vecs:')
    print(vecs)
    print('vec_len:')
    print(vec_len)
    if vec_len == 2600:
        nz_blocks = 256
        z_blocks = 68
    # elif vec_len == 10000:  # sparsity = 0.1
    #     nz_blocks = 1000
    #     z_blocks = 250
    elif vec_len == 4096:
        if percent == 87.5:
            nz_blocks = 512
            z_blocks = 0
        elif percent == 93.75:
            nz_blocks = 256
            z_blocks = 256
        elif percent == 95:
            nz_blocks = 200
            z_blocks = 310
    elif vec_len == 10000:  # sparsity = 0.11
        if percent == 96.5:
            nz_blocks = 344
            z_blocks = 902
        elif percent == 96:
            nz_blocks = 400
            z_blocks = 850
        elif percent == 95:
            nz_blocks = 496
            z_blocks = 753
        elif percent == 94:
            nz_blocks = 600
            z_blocks = 650
        elif percent == 90:
            nz_blocks = 1000
            z_blocks = 250
        elif percent == 89:
            nz_blocks = 1096
            z_blocks = 153
        elif percent == 88:
            nz_blocks = 1200
            z_blocks = 50
        elif percent == 87.5:
            nz_blocks = 1248
            z_blocks = 1
        # nz_blocks = 1016
        # z_blocks = 150
    elif vec_len == 6000:
        if percent == 96.5:
            nz_blocks = 208
            z_blocks = 541
        elif percent == 96:
            nz_blocks = 240
            z_blocks = 510
        elif percent == 95:
            nz_blocks = 296
            z_blocks = 453
        elif percent == 94:
            nz_blocks = 360
            z_blocks = 390
        elif percent == 90:
            nz_blocks = 600
            z_blocks = 150
        elif percent == 91:
            nz_blocks = 536
            z_blocks = 213
    elif vec_len == 1500:
        if percent == 90:
            nz_blocks = 144
            z_blocks = 42
        elif percent == 91:
            nz_blocks = 128
            z_blocks = 58
    elif vec_len == 2500:
        if percent == 88:
            nz_blocks = 296
            z_blocks = 15
    elif vec_len == 1024:
        if percent == 87.5:
            nz_blocks = 128
            z_blocks = 0
        elif percent == 93.75:
            nz_blocks = 128
            z_blocks = 128
        elif percent == 95:
            nz_blocks = 48
            z_blocks = 79
    else:
        nz_blocks = 0
        z_blocks = 0

    blocks = nz_blocks + z_blocks
    assert blocks != 0
    irregualr_size = vec_len - blocks * block_len

    wt_abs = np.abs(weight)
    wt_abs_reg = wt_abs[:, :blocks * block_len]
    reg_vecs, reg_vec_len = np.shape(wt_abs_reg)
    reg_mask = np.zeros([reg_vecs, reg_vec_len])
    # irreg_mask = np.zeros([irreg_vecs, irreg_vec_len])

    for i in range(vecs):
        max_value = np.zeros(blocks)
        max_val_pos = np.zeros(blocks, dtype=int)
        block_number = np.arange(blocks)
        # max_value = []
        # max_index = []
        bucket_len = int(nz_blocks / block_types)
        block_buckets = np.zeros([block_types, bucket_len], dtype=int)
        bucket_count = np.zeros(block_types, dtype=int)
        for j in range(blocks):
            cur_block = wt_abs_reg[i, j * block_len:(j + 1) * block_len]
            # max_value.append(np.max(cur_block))
            # max_index.append(np.argmax(cur_block))
            max_value[j] = np.max(cur_block)
            max_val_pos[j] = np.argmax(cur_block)
        val_pos_blk = zip(max_value, max_val_pos, block_number)
        sorted_val_pos_blk = sorted(val_pos_blk, key=lambda x: x[0], reverse=True)  # descending order

        # natural_max_sorted = np.sort(wt_abs_reg[i])
        # natural_max_sorted = natural_max_sorted[::-1]

        classfying_count = block_types * bucket_len
        cur_count = 0
        while cur_count < classfying_count:
            cur_zip = sorted_val_pos_blk[0]
            cur_block_type = cur_zip[1]
            cur_block_number = cur_zip[2]
            # cur_index = sorted_index[0] # block_number
            # cur_block_type = max_index[cur_index] # the position of the max in a block
            cur_bucket_count = bucket_count[cur_block_type]  # number of elements in a bucket
            if cur_bucket_count < bucket_len:
                block_buckets[cur_block_type, cur_bucket_count] = cur_block_number
                bucket_count[cur_block_type] += 1
                # if bucket_count[cur_block_type] == bucket_len:
                #     print("test")
                cur_count += 1
                sorted_val_pos_blk.pop(0)
                # max_value.pop(cur_index)
                # max_index.pop(cur_index)
                # sorted_index.pop(0)
            else:
                # for (max_val, max_val_pos, block_number) in sorted_val_pos_blk:
                #     if max_val_pos == cur_block_type:
                list_len = len(sorted_val_pos_blk)
                for k in range(list_len):
                    tmp_zip = sorted_val_pos_blk[k]
                    tmp_pos = tmp_zip[1]
                    if tmp_pos == cur_block_type:
                        tmp_block_num = tmp_zip[2]
                        tmp_block = wt_abs_reg[i, tmp_block_num * block_len:(tmp_block_num + 1) * block_len]
                        tmp_block[tmp_pos] = 0
                        tmp_max = np.max(tmp_block)
                        tmp_new_pos = np.argmax(tmp_block)
                        sorted_val_pos_blk[k] = (tmp_max, tmp_new_pos, tmp_block_num)
                sorted_val_pos_blk.sort(key=lambda x: x[0], reverse=True)
                # cur_block = wt_abs_reg[i, cur_block_number * block_len:(cur_block_number + 1) * block_len]
                # cur_block[cur_block_type] = 0
                # max_value = np.max(cur_block)
                # max_val_pos = np.argmax(cur_block)
                # sorted_val_pos_blk[0] = (max_value, max_val_pos, cur_block_number)
                # sorted_val_pos_blk.sort(key=lambda x: x[0], reverse=True)

        for t in range(block_types):
            for l in range(bucket_len):
                cur_position = block_buckets[t, l] * block_len + t
                reg_mask[i, cur_position] = 1

    if irregualr_size != 0:
        wt_abs_irreg = wt_abs[:, blocks * block_len:]
        irreg_vecs, irreg_vec_len = np.shape(wt_abs_irreg)
        irreg_percent = (1 - (vec_len * (1 - percent / 100) - nz_blocks) / irreg_vec_len) * 100
        percentile = np.percentile(wt_abs_irreg, irreg_percent)
        above_threshold = wt_abs_irreg > percentile
        irreg_mask = above_threshold.astype(np.float32)
        mask = np.concatenate((reg_mask, irreg_mask), axis=1)
    else:
        mask = reg_mask
    mask = mask.astype(np.float32)

    weight *= mask
    return weight, mask


def weight_pruning(config, weight, prune_ratio, cross_x=4, cross_f=1):
    """
    weight pruning [irregular,column,filter]
    Args:
         weight (pytorch tensor): weight tensor, ordered by output_channel, intput_channel, kernel width and kernel height
         prune_ratio (float between 0-1): target sparsity of weights

    Returns:
         mask for nonzero weights used for retraining
         a pytorch tensor whose elements/column/row that have lowest l2 norms(equivalent to absolute weight here) are set to zero

    """

    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    sparsity_type = config['exp']['sparsity_type']

    percent = prune_ratio * 100
    if (sparsity_type == "irregular"):
        weight_temp = np.abs(weight)  # a buffer that holds weights with absolute values
        percentile = np.percentile(weight_temp, percent)  # get a value for this percentitle
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "block_column"):
        weight = np.transpose(weight)  # 512x784 ==> 784x512
        org_shape = weight.shape
        group_size = int(config['exp']['group_size'])  #每一个block的大小
        org_vecs = org_shape[0]        #全部的行数
        remain_vecs = org_vecs%group_size   #被block_prune后，剩下的行数
        group_wt_org_shape = weight[:(org_vecs-remain_vecs)].shape   # 被block_prune的所有行数
        if remain_vecs == 0:
            weight_groups = weight.reshape((-1, group_size, org_shape[1]))
        else:
            weight_groups = weight[:(org_vecs-remain_vecs)].reshape((-1, group_size, org_shape[1]))
        # weight_groups = weight.reshape((-1, group_size, org_shape[1]))
        groups_shape = weight_groups.shape
        group_mask = np.zeros(groups_shape, dtype=np.float32)
        for gp in range(groups_shape[0]):
            column_l2_norm = LA.norm(weight_groups[gp], 2, axis=0)
            percentile = np.percentile(column_l2_norm, percent)
            under_threshold = column_l2_norm < percentile
            above_threshold = column_l2_norm > percentile
            weight_groups[gp, :, under_threshold] = 0
            above_threshold = above_threshold.astype(np.float32)
            for i in range(groups_shape[2]):
                group_mask[gp, :, i] = above_threshold[i]
        above_threshold_msk = group_mask.reshape(group_wt_org_shape)
        # above_threshold_msk = above_threshold_msk.reshape(org_shape)
        weight_groups = weight_groups.reshape(group_wt_org_shape)

        if remain_vecs != 0:
            group_vecs = org_vecs-remain_vecs
            weight_remain = weight[group_vecs:]
            remain_shape = weight_remain.shape
            column_l2_norm = LA.norm(weight_remain, 2, axis=0)
            percentile = np.percentile(column_l2_norm, percent)
            under_threshold = column_l2_norm < percentile
            above_threshold = column_l2_norm > percentile
            weight_remain[:, under_threshold] = 0
            remain_mask = np.zeros(remain_shape, dtype=np.float32)
            for i in range(groups_shape[2]):
                remain_mask[:, i] = above_threshold[i]
            remain_mask = remain_mask.astype(np.float32)
            weight = np.concatenate((weight_groups, weight_remain), axis=0)
            above_threshold_msk = np.concatenate((above_threshold_msk, remain_mask), axis=0)
        else:
            weight = weight_groups

        weight = np.transpose(weight)  # 784x512 ==> 512x784
        above_threshold_msk = np.transpose(above_threshold_msk)
        return torch.from_numpy(above_threshold_msk).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "block_row"):
        weight = np.transpose(weight)  # 512x784 ==> 784x512
        org_shape = weight.shape
        bank_size = int(org_shape[1] / 2)
        above_threshold_msk = np.zeros(org_shape, dtype=np.float32)
        lft_l2_norm = LA.norm(weight[:, :bank_size], 2, axis=1)
        rgt_l2_norm = LA.norm(weight[:, bank_size:], 2, axis=1)
        percentile_lft = np.percentile(lft_l2_norm, percent)
        percentile_rgt = np.percentile(rgt_l2_norm, percent)
        under_th_lft = lft_l2_norm < percentile_lft
        under_th_rgt = rgt_l2_norm < percentile_rgt
        above_th_lft = lft_l2_norm > percentile_lft
        above_th_rgt = rgt_l2_norm > percentile_rgt

        for i in range(org_shape[0]):
            if under_th_lft[i] and under_th_rgt[i]:
                if lft_l2_norm[i] < rgt_l2_norm[i]:
                    under_th_rgt[i] = False
                    above_th_rgt[i] = True
                else:
                    under_th_lft[i] = False
                    above_th_lft[i] = True

        weight[under_th_lft, :bank_size] = 0
        weight[under_th_rgt, bank_size:] = 0
        above_th_rgt = above_th_rgt.astype(np.float32)
        above_th_lft = above_th_lft.astype(np.float32)
        for j in range(org_shape[0]):
            above_threshold_msk[j, :bank_size] = above_th_lft[j]
            above_threshold_msk[j, bank_size:] = above_th_rgt[j]
        weight = np.transpose(weight)  # 784x512 ==> 512x784
        above_threshold_msk = np.transpose(above_threshold_msk)
        return torch.from_numpy(above_threshold_msk).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "block_max"):
        weight = np.transpose(weight)  # 512x784 ==> 784x512
        org_shape = weight.shape
        retained_mask = np.zeros(org_shape, dtype=np.float32)
        mask_block_max(weight, retained_mask, percent)
        retained_mask = np.transpose(retained_mask)
        weight = np.transpose(weight)  # 784x512 ==> 512x784
        return torch.from_numpy(retained_mask).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "balanced_block"):
        weight = np.transpose(weight)  # 512x784 ==> 784x512
        org_shape = weight.shape
        # retained_mask = np.zeros(org_shape, dtype=np.float32)
        block_len = 8
        block_types = 8
        weight, retained_mask = mask_balanced_block_max(weight, block_len, block_types, percent)
        retained_mask = np.transpose(retained_mask)
        weight = np.transpose(weight)  # 784x512 ==> 512x784
        return torch.from_numpy(retained_mask).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        column_l2_norm = LA.norm(weight2d, 2, axis=0)
        percentile = np.percentile(column_l2_norm, percent)
        under_threshold = column_l2_norm < percentile
        above_threshold = column_l2_norm > percentile
        weight2d[:, under_threshold] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[1]):
            expand_above_threshold[:, i] = above_threshold[i]
        expand_above_threshold = expand_above_threshold.reshape(shape)
        weight = weight2d.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        row_l2_norm = LA.norm(weight2d, 2, axis=1)
        percentile = np.percentile(row_l2_norm, percent)
        under_threshold = row_l2_norm < percentile
        above_threshold = row_l2_norm > percentile
        weight2d[under_threshold, :] = 0
        above_threshold = above_threshold.astype(np.float32)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = above_threshold[i]

        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "bn_filter"):
        ## bn pruning is very similar to bias pruning
        weight_temp = np.abs(weight)
        percentile = np.percentile(weight_temp, percent)
        under_threshold = weight_temp < percentile
        above_threshold = weight_temp > percentile
        above_threshold = above_threshold.astype(
            np.float32)  # has to convert bool to float32 for numpy-tensor conversion
        weight[under_threshold] = 0
        return torch.from_numpy(above_threshold).cuda(), torch.from_numpy(weight).cuda()
    elif (sparsity_type == "balanced_block_prune_filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        if shape2d[0] % cross_f != 0 or shape2d[1] % cross_x != 0:
            print("the layer size is not divisible")
            raise SyntaxError("block_size error")
        else:
            length_f = int(shape2d[0] / cross_f)  #行分块
            length_x = int(shape2d[1] / cross_x)  #列分块

        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)

        for x in range(cross_x):
            # print("x={}/{}".format(x,crossbar_num_x))
            for f in range(cross_f):
                # print("f={}/{}".format(f, crossbar_num_f))
                frag = weight2d[f * length_f:(f + 1) * length_f, x * length_x:(x + 1) * length_x]
                frag_above = expand_above_threshold[f * length_f:(f + 1) * length_f, x * length_x:(x + 1) * length_x]
                row_l2_norm = LA.norm(frag, 2, axis=1)
                percentile = np.percentile(row_l2_norm, percent)
                under_threshold = row_l2_norm <= percentile
                above_threshold = row_l2_norm > percentile
                frag[under_threshold, :] = 0
                # weight2d[weight2d < 1e-40] = 0
                above_threshold = above_threshold.astype(np.float32)

                for i in range(length_f):
                    frag_above[i, :] = above_threshold[i]

                # change frag will change weight2d as well

        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


    elif (sparsity_type == "balanced_block_prune_column"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        if shape2d[0] % cross_f != 0 or shape2d[1] % cross_x != 0:
            print("the layer size is not divisible")
            raise SyntaxError("block_size error")
        else:
            length_f = int(shape2d[0] / cross_f)
            length_x = int(shape2d[1] / cross_x)

        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)

        for x in range(cross_x):
            # print("x={}/{}".format(x,crossbar_num_x))
            for f in range(cross_f):
                # print("f={}/{}".format(f, crossbar_num_f))
                frag = weight2d[f * length_f:(f + 1) * length_f, x * length_x:(x + 1) * length_x]
                frag_above = expand_above_threshold[f * length_f:(f + 1) * length_f, x * length_x:(x + 1) * length_x]
                row_l2_norm = LA.norm(frag, 2, axis=0)
                percentile = np.percentile(row_l2_norm, percent)
                under_threshold = row_l2_norm <= percentile
                above_threshold = row_l2_norm > percentile
                frag[:, under_threshold] = 0
                # weight2d[weight2d < 1e-40] = 0
                above_threshold = above_threshold.astype(np.float32)

                for i in range(length_f):
                    frag_above[:, i] = above_threshold[i]

                # change frag will change weight2d as well

        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (sparsity_type == "block_prune_filter"):  # 1*1（f）
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        if shape2d[0] % cross_f != 0 or shape2d[1] % cross_x != 0:
            length_f = int(shape2d[0] / cross_f)  # 行
            length_x = int(shape2d[1] / cross_x)

            for x in range(cross_x -1):
                # print("f={}/{}".format(f, crossbar_num_f))
                frag = weight2d[:, x * length_x:(x + 1) * length_x]
                frag_above = expand_above_threshold[:, x * length_x:(x + 1) * length_x]
                row_l2_norm = LA.norm(frag, 2, axis=1)
                percentile = np.percentile(row_l2_norm, percent)
                under_threshold = row_l2_norm <= percentile
                above_threshold = row_l2_norm > percentile
                frag[under_threshold, :] = 0
                # weight2d[weight2d < 1e-40] = 0
                above_threshold = above_threshold.astype(np.float32)

                for i in range(length_f):
                    frag_above[i, :] = above_threshold[i]

            frag = weight2d[:, (x+1) * length_x:]
            frag_above = expand_above_threshold[:, (x+1)  * length_x:]
            row_l2_norm = LA.norm(frag, 2, axis=1)
            percentile = np.percentile(row_l2_norm, percent)
            under_threshold = row_l2_norm <= percentile
            above_threshold = row_l2_norm > percentile
            frag[under_threshold, :] = 0
            # weight2d[weight2d < 1e-40] = 0
            above_threshold = above_threshold.astype(np.float32)
            for i in range(length_f):
                frag_above[i, :] = above_threshold[i]


        else:
            length_f = int(shape2d[0] / cross_f)  #行
            length_x = int(shape2d[1] / cross_x)  #列

            for x in range(cross_x):
                # print("f={}/{}".format(f, crossbar_num_f))
                frag = weight2d[:, x * length_x:(x + 1) * length_x]
                frag_above = expand_above_threshold[:, x * length_x:(x + 1) * length_x]
                row_l2_norm = LA.norm(frag, 2, axis=1)
                percentile = np.percentile(row_l2_norm, percent)
                under_threshold = row_l2_norm <= percentile
                above_threshold = row_l2_norm > percentile
                frag[under_threshold, :] = 0
                # weight2d[weight2d < 1e-40] = 0
                above_threshold = above_threshold.astype(np.float32)

                for i in range(length_f):
                    frag_above[i, :] = above_threshold[i]

                # change frag will change weight2d as well

        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

    elif (sparsity_type == "block_prune_column"): #1 * 1(f)
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        if shape2d[0] % cross_f != 0 or shape2d[1] % cross_x != 0:
            print("the layer size is not divisible")
            raise SyntaxError("block_size error")
        else:
            length_f = int(shape2d[0] / cross_f)
            length_x = int(shape2d[1] / cross_x)


        for f in range(cross_f):
            # print("f={}/{}".format(f, crossbar_num_f))
            frag = weight2d[f * length_f:(f + 1) * length_f, :]
            frag_above = expand_above_threshold[f * length_f:(f + 1) * length_f, :]
            row_l2_norm = LA.norm(frag, 2, axis=0)
            percentile = np.percentile(row_l2_norm, percent)
            under_threshold = row_l2_norm <= percentile
            above_threshold = row_l2_norm > percentile
            frag[:, under_threshold] = 0
            # weight2d[weight2d < 1e-40] = 0
            above_threshold = above_threshold.astype(np.float32)

            for i in range(length_x):
                frag_above[:, i] = above_threshold[i]

        # change frag will change weight2d as well

        weight = weight.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()

    else:
        raise SyntaxError("Unknown sparsity type")


def test_sparsity(config, model, ADMM):
    """
    test sparsity for every involved layer and the overall compression rate

    """
    prune_ratios = list(map(float, config['exp']['prune_ratios'].split(',')))
    sparsity_type = config['exp']['sparsity_type']

    total_zeros = 0
    total_nonzeros = 0

    print('<===sparsity type is {}'.format(sparsity_type))
    print('<===layers to be pruned are {}'.format(prune_ratios))

    if sparsity_type == "block_prune_filter":
        total_zeros = 0
        total_nonzeros = 0
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if name not in ADMM.prune_ratios:
                    continue
                W = W.cpu().detach().numpy()
                zeros = np.sum(W == 0)
                total_zeros += zeros
                nonzeros = np.sum(W != 0)
                total_nonzeros += nonzeros
                print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
            break
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(total_weight_number / total_nonzeros))
    elif sparsity_type == "block_prune_column":
        total_zeros = 0
        total_nonzeros = 0
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if name not in ADMM.prune_ratios:
                    continue
                W = W.cpu().detach().numpy()
                zeros = np.sum(W == 0)
                total_zeros += zeros
                nonzeros = np.sum(W != 0)
                total_nonzeros += nonzeros
                print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
            break
        total_weight_number = total_zeros + total_nonzeros
        print('overal compression rate is {}'.format(total_weight_number / total_nonzeros))
    elif sparsity_type == "column":
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if name not in ADMM.prune_ratios:
                    continue
                W = W.cpu().detach().numpy()
                shape = W.shape
                W2d = W.reshape(shape[0], -1)
                column_l2_norm = LA.norm(W2d, 2, axis=0)
                zero_column = np.sum(column_l2_norm == 0)
                nonzero_column = np.sum(column_l2_norm != 0)
                total_zeros += np.sum(W == 0)
                total_nonzeros += np.sum(W != 0)
                print("column sparsity of layer {} is {}".format(name, zero_column / (zero_column + nonzero_column)))
            break

        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif sparsity_type == "filter":
        print('inside if')
        print(prune_ratios)
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if name not in ADMM.prune_ratios:
                    continue
                W = W.cpu().detach().numpy()
                shape = W.shape
                W2d = W.reshape(shape[0], -1)
                row_l2_norm = LA.norm(W2d, 2, axis=1)
                zero_row = np.sum(row_l2_norm == 0)
                nonzero_row = np.sum(row_l2_norm != 0)
                total_zeros += np.sum(W == 0)
                total_nonzeros += np.sum(W != 0)
                print("filter sparsity of layer {} is {}".format(name, zero_row / (zero_row + nonzero_row)))
            break
        print(
            'only consider conv layers, compression rate is {}'.format((total_zeros + total_nonzeros) / total_nonzeros))
    elif sparsity_type == "bn_filter":
        print('inside bn_filter')
        print(prune_ratios)
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if name not in ADMM.prune_ratios:
                    continue
                W = W.cpu().detach().numpy()
                zeros = np.sum(W == 0)
                nonzeros = np.sum(W != 0)
                print("sparsity at layer {} is {}".format(name, zeros / (zeros + nonzeros)))
            break


def predict_sparsity(config):
    # given a model, calculate the sparsity before proceeding.
    model = config.model
    total_parameters = 0  # parameters from  all conv layers
    nonzero_parameters = 0  # all remained non zero parameters
    layers = []
    ratios = []
    for name, W in model.named_parameters():
        if name not in config.prune_ratios:
            continue
        layers.append(W.cpu().detach().numpy())
        ratios.append(config.prune_ratios[name])
    for i in range(len(layers)):
        W = layers[i]
        ratio = ratios[i]
        numel = W.flatten().size
        total_parameters += numel
        cur_nonzero = (1 - ratio) * numel
        if i != 0 and ratios[i - 1] != 0:
            cur_nonzero *= (1 - ratios[i - 1])
        nonzero_parameters += cur_nonzero
    print('predicting sparsity after pruning..... {}'.format(total_parameters / nonzero_parameters))


def admm_initialization(config, ADMM, model):
    admm = strtobool(config['exp']['admm'])
    if not admm:
        return
    for net in model.keys():
        for name, W in model[net].named_parameters():
            if name in ADMM.prune_ratios:
                _, updated_Z = weight_pruning(config, W, ADMM.prune_ratios[name], ADMM.cross[name][0], ADMM.cross[name][1])  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
                ADMM.ADMM_Z[name] = updated_Z
        break


def admm_update(config, ADMM, model, epoch, batch_idx):
    admm = strtobool(config['exp']['admm'])
    admm_epoch = int(config['exp']['admm_epoch'])
    sparsity_type = config['exp']['sparsity_type']
    multi_rho = strtobool(config['exp']['multi_rho'])
    if not admm:
        return
    # sometimes the start epoch is not zero. It won't be valid if the start epoch is not 0
    if epoch == 0 and batch_idx == 0:
        admm_initialization(config, ADMM, model)  # intialize Z, U variable
    if epoch != 0 and epoch % admm_epoch == 0 and batch_idx == 0:
        # if epoch % config.admm_epoch == 0 and batch_idx == 0:
        for net in model.keys():
            for name, W in model[net].named_parameters():
                if sparsity_type != "quantization":
                    if name not in ADMM.prune_ratios:
                        continue

                    # if config.verbose and config.sparsity_type!="quantization":
                    Z_prev = torch.Tensor(ADMM.ADMM_Z[name].cpu()).cuda()
                    # Z_prev = weight.cpu().detach().numpy().cuda()

                    ADMM.ADMM_Z[name] = W + ADMM.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

                    _, _Z = weight_pruning(config, ADMM.ADMM_Z[name],
                                           ADMM.prune_ratios[name], ADMM.cross[name][0], ADMM.cross[name][1])  # equivalent to Euclidean Projection
                    ADMM.ADMM_Z[name] = _Z
                    ADMM.ADMM_U[name] = W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)
                    # ADMM.ADMM_U[name] = ADMM.ADMM_U[name]  # U(k+1) = W(k+1) - Z(k+1) +U(k)

                    if multi_rho:
                        admm_multi_rho_scheduler(ADMM, name, W, Z_prev)  # call multi rho scheduler every admm update

                else:
                    if name not in ADMM.number_bits:
                        continue
                    _Q, _alpha = Q_alpha_update(config, W, ADMM.ADMM_Q, ADMM.ADMM_U, ADMM.ADMM_alpha,
                                                ADMM.number_bits[name])
                    ADMM.ADMM_Q = _Q
                    ADMM.ADMM_alpha = _alpha
            break


def append_admm_loss(config, ADMM, model, ce_loss):
    '''
    append admm loss to cross_entropy loss
    Args:
        args: configuration parameters
        model: instance to the model class
        ce_loss: the cross entropy loss
    Returns:
        ce_loss(tensor scalar): original cross enropy loss
        admm_loss(dict, name->tensor scalar): a dictionary to show loss for each layer
        ret_loss(scalar): the mixed overall loss

    '''
    admm = strtobool(config['exp']['admm'])
    sparsity_type = config['exp']['sparsity_type']

    admm_loss = {}

    if admm:
        if sparsity_type != "quantization":
            for net in model.keys():
                for name, W in model[net].named_parameters():  ## initialize Z (for both weights and bias)
                    if name not in ADMM.prune_ratios:
                        continue

                    admm_loss[name] = 0.5 * ADMM.rhos[name] * (
                            torch.norm(W - ADMM.ADMM_Z[name] + ADMM.ADMM_U[name], p=2) ** 2)
                break
        else:
            for net in model.keys():
                for name, W in model[net].named_parameters():
                    if name not in ADMM.number_bits:
                        continue
                    admm_loss[name] = 0.5 * ADMM.rhos[name] * (
                            torch.norm(W - ADMM.alpha[name] * ADMM.ADMM_Q[name] + ADMM.ADMM_U[name], p=2) ** 2)
                break
        mixed_loss = 0
        mixed_loss += ce_loss
        for k, v in admm_loss.items():
            mixed_loss += v
        return ce_loss, admm_loss, mixed_loss


def admm_multi_rho_scheduler(ADMM, name, W, z_prev):
    """
    It works better to make rho monotonically increasing
    we increase it by 1.9x every admm epoch
    After 10 admm updates, the rho will be 0.91

    """
    dis_w_z = torch.norm(W - ADMM.ADMM_Z[name], p=2) ** 2
    dis_z_z = torch.norm(z_prev - ADMM.ADMM_Z[name], p=2) ** 2
    print('distance between w and z: ', dis_w_z.item())
    print('distance between z_prev and z: ', dis_z_z.item())

    rho_prev = ADMM.rhos[name]
    primal_dual_ratio = dis_w_z / dis_z_z
    # ADMM.rhos[name] *= 2

    if primal_dual_ratio < 0.1:
        ADMM.rhos[name] /= 1.2
    else:
        ADMM.rhos[name] *= 1.2

    # if primal_dual_ratio > 1:
    #     ADMM.rhos[name] *= 2
    # elif primal_dual_ratio < 1:
    #     ADMM.rhos[name] /= 2

    if rho_prev > 0.02:
        ADMM.rhos[name] = rho_prev + 0.02
    if rho_prev > 0.5:
        ADMM.rhos[name] = rho_prev
    print('<=====using multi rho scheduler, rho = ', ADMM.rhos[name])


def zero_masking(config, model):
    masks = {}
    for net in model.keys():
        for name, W in model[net].named_parameters():
            if  ('bn' in name) or ('ln' in name):
                continue
            w_temp = W.cpu().detach().numpy()
            indices = (w_temp != 0)
            indices = indices.astype(np.float32)
            masks[name] = torch.from_numpy(indices).cuda()
        break

    return masks


def masking(config, ADMM,model):
    masks = {}
    prune_ratios = list(map(float, config['exp']['prune_ratios'].split(',')))
    for net in model.keys():
        for name, W in model[net].named_parameters():
            if name not in ADMM.prune_ratios:
                continue
            above_threshold, pruned_weight = weight_pruning(config, W, ADMM.prune_ratios[name], ADMM.cross[name][0], ADMM.cross[name][1])
            W.data = pruned_weight
            masks[name] = above_threshold
        break

    return masks


def post_processing2(conv_layers, config, empty_channels):
    for j in range(len(conv_layers)):
        weight = conv_layers[j]
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)

        for i in range(shape[0]):
            if np.sum(weight2d[i, :]) == 0:
                if j == len(conv_layers) - 1:
                    continue
                if j + 1 not in empty_channels:
                    empty_channels[j + 1] = 0
                else:
                    empty_channels[j + 1] += 1
                next_weight = conv_layers[j + 1]
                next_weight_shape = next_weight.shape
                next_weight_2d = next_weight.reshape(next_weight.shape[0], -1)
                h, w = next_weight_shape[-2:]
                step = h * w
                next_weight_2d[:, i * h * w:(i + 1) * h * w] = 0
                conv_layers[j + 1] = next_weight_2d.reshape(next_weight_shape)


def post_processing(conv_layers, config, empty_filters):
    # out_channel, in_channel, h, w
    for j in range(len(conv_layers)):
        weight = conv_layers[j]
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        conv_layers.append(weight)
        h, w = weight.shape[-2:]
        step = h * w
        # print ('shape {}'.format(shape))
        # print ('column {}'.format(weight2d.shape[1]))
        # print ('step is {}'.format(step))
        # print ('number of step is {}'.format(weight2d.shape[1]/step))
        for i in range(int(weight2d.shape[1] / step)):
            if np.sum(weight2d[:, i * step:i * (step + 1)]) == 0:

                if j == 0:
                    continue
                if j - 1 not in empty_filters:
                    empty_filters[j - 1] = 0
                else:
                    empty_filters[j - 1] += 1
                # print ('find empty channel')
                prev_weight = conv_layers[j - 1]
                prev_weight_shape = prev_weight.shape
                prev_weight_2d = prev_weight.reshape(prev_weight_shape[0], -1)
                prev_weight_2d[i, :] = 0
                conv_layers[j - 1] = prev_weight_2d.reshape(prev_weight_shape)
