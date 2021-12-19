
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from past.builtins import xrange
import pickle 
import numpy as np
import os
import torch
from torch import nn
import sys
import torch
from torch import nn
# import Dataloader
from torch.utils.data import DataLoader
import tqdm
import os
import pprint as pp
import torch.optim as optim

import datetime
import numpy as np
import os
import threading
sys.path
sys.path.append('../../')
# almost similar to the original implementations

class Dataset(object):
    """docstring for Dataset"""
    def __init__(self):
        super(Dataset, self).__init__()
        self.dataset = "mv"
        self.model_type = "PW"
        self.band_size = 20
        #load the data
        #data_filename = os.path.join(args.data_folder, args.dataset+'.pkl')
        f = open("movie.pkl", 'rb')
        data_behavior = pickle.load(f) # time and user behavior
        del data_behavior[0]
        tmp=[]
        for i in range(100):
            tmp.append(data_behavior[i])
        data_behavior=tmp
        #실구현시에는 movie의 개수임.
        #그냥 movie를 원핫 벡터화했는데요..? 그러니까 movie1=[1.0,,0,0,0,0],movie2=[0.1,,0,0,0,0]
        #item_feature = pickle.load(f) # identity matrix
        item_feature = np.eye(3952)
        f.close()

        self.size_item = len(item_feature)
        self.size_user = len(data_behavior)
        self.f_dim = len(item_feature[0])

        # load the index fo train,test,valid split 

        self.train_user = range(70)
        self.vali_user = range(70,80)
        self.test_user = range(80,100)

        # process the data

        # get the most no of suggetion for an individual at a time
        k_max = 0
        for d_b in data_behavior:
            for disp in d_b[1]:
                k_max = max(k_max, len(disp))

        self.data_click = [[] for x in xrange(self.size_user)]
        self.data_disp = [[] for x in xrange(self.size_user)]
        self.data_time = np.zeros(self.size_user, dtype=np.int)
        self.data_news_cnt = np.zeros(self.size_user, dtype=np.int)
        self.feature = [[] for x in xrange(self.size_user)]
        self.feature_click = [[] for x in xrange(self.size_user)]

        for user in xrange(self.size_user):
            print(user)
            # (1) count number of clicks
            click_t = 0
            num_events = len(data_behavior[user][1])
            click_t += num_events
            self.data_time[user] = click_t
            # (2)
            news_dict = {}
            self.feature_click[user] = np.zeros([click_t, self.f_dim])
            click_t = 0
            for event in xrange(num_events):
                disp_list = data_behavior[user][1][event]
                pick_id = data_behavior[user][2][event]
                for id in disp_list:
                    if id not in news_dict:
                        news_dict[id] = len(news_dict)  # for each user, news id start from 0
                if pick_id:
                    id = pick_id
                    self.data_click[user].append([click_t, news_dict[id]])
                    self.feature_click[user][click_t] = item_feature[id-1]
                for idd in disp_list:
                    self.data_disp[user].append([click_t, news_dict[idd]])
                click_t += 1  # splitter a event with 2 clickings to 2 events

            self.data_news_cnt[user] = len(news_dict)

            self.feature[user] = np.zeros([self.data_news_cnt[user], self.f_dim])

            for id in news_dict:
                self.feature[user][news_dict[id]] = item_feature[id-1]
            self.feature[user] = self.feature[user].tolist()
            self.feature_click[user] = self.feature_click[user].tolist()
        self.max_disp_size = k_max
        
    def random_split_user(self):
        # dont think this one is really necessary if the initial split is random enough
        num_users = len(self.train_user) + len(self.vali_user) + len(self.test_user)
        shuffle_order = np.arange(num_users)
        np.random.shuffle(shuffle_order)
        self.train_user = shuffle_order[0:len(self.train_user)].tolist()
        self.vali_user = shuffle_order[len(self.train_user):len(self.train_user)+len(self.vali_user)].tolist()
        self.test_user = shuffle_order[len(self.train_user)+len(self.vali_user):].tolist()

    def data_process_for_placeholder(self, user_set):
        #print ("user_set",user_set)
        if self.model_type == 'PW':
            
            sec_cnt_x = 0
            news_cnt_short_x = 0
            news_cnt_x = 0
            click_2d_x = []
            disp_2d_x = []

            tril_indice = []
            tril_value_indice = []

            disp_2d_split_sec = []
            feature_clicked_x = []

            disp_current_feature_x = []
            click_sub_index_2d = []

            # started with the validation set
            #[703, 713, 723, 733, 743, 753, 763, 773, 783, 793, 803, 813, 823, 833, 843, 853, 863, 873, 883, 893, 903, 913, 923, 933, 943, 953, 963, 973, 983, 993, 1003, 1013, 1023, 1033, 1043, 1053]
            #user_set = [703]
            for u in user_set:
                t_indice = []
               # print(self.data_time[u]-1)
                for kk in xrange(min(self.band_size-1, self.data_time[u]-1)):
                    t_indice += map(lambda x: [x + kk+1 + sec_cnt_x, x + sec_cnt_x], np.arange(self.data_time[u] - (kk+1)))
              # print (t_indice) #[] for 703
                
                tril_indice += t_indice
                tril_value_indice += map(lambda x: (x[0] - x[1] - 1), t_indice)
                #print ("THE Click data is ",self.data_click[u]) #THE Click data is  [[0, 0], [1, 8], [2, 14]] for u =15
                click_2d_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], self.data_click[u])
                click_2d_tmp = list(click_2d_tmp)
                #print (list(click_2d_tmp))
                #print (list(click_2d_tmp))
                click_2d_x += click_2d_tmp
                #print ("tenp is ",click_2d_x,list(click_2d_tmp))  # [[0, 0], [1, 8], [2, 14]] for u15
                #print ("dispaly data is ", self.data_disp[u]) [0,0]
                

                disp_2d_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], self.data_disp[u])
                disp_2d_tmp = list(disp_2d_tmp)
                #y=[]
                #y+=disp_2d_tmp
                #print (disp_2d_tmp, click_2d_tmp)
                click_sub_index_tmp = map(lambda x: disp_2d_tmp.index(x), (click_2d_tmp))
                click_sub_index_tmp = list(click_sub_index_tmp)
                #print ("the mess is ",click_sub_index_tmp)
                click_sub_index_2d += map(lambda x: x+len(disp_2d_x), click_sub_index_tmp)
                #print ("click_sub_index_2d",click_sub_index_2d)
                disp_2d_x += disp_2d_tmp
                #print ("disp_2d_x",disp_2d_x) # [[0, 0]]
                #sys.exit()
                disp_2d_split_sec += map(lambda x: x[0] + sec_cnt_x, self.data_disp[u])

                sec_cnt_x += self.data_time[u]
                news_cnt_short_x = max(news_cnt_short_x, self.data_news_cnt[u])
                news_cnt_x += self.data_news_cnt[u]
                disp_current_feature_x += map(lambda x: self.feature[u][x], [idd[1] for idd in self.data_disp[u]])
                feature_clicked_x += self.feature_click[u]

                out1 ={}
                out1['click_2d_x']=click_2d_x
                out1['disp_2d_x']=disp_2d_x
                out1['disp_current_feature_x']=disp_current_feature_x
                out1['sec_cnt_x']=sec_cnt_x
                out1['tril_indice']=tril_indice
                out1['tril_value_indice']=tril_value_indice
                out1['disp_2d_split_sec']=disp_2d_split_sec
                out1['news_cnt_short_x']=news_cnt_short_x
                out1['click_sub_index_2d']=click_sub_index_2d
                out1['feature_clicked_x']=feature_clicked_x
            # print ("out",out1['tril_value_indice'])
#             # sys.exit()
#             with open('user.pickle','wb') as fw:
#                 pickle.dump(out1, fw)
            return out1


    def prepare_validation_data(self, num_sets, v_user):

        if self.model_type == 'PW':
            vali_thread_u = [[] for _ in xrange(num_sets)]
            click_2d_v = [[] for _ in xrange(num_sets)]
            disp_2d_v = [[] for _ in xrange(num_sets)]
            feature_v = [[] for _ in xrange(num_sets)]
            sec_cnt_v = [[] for _ in xrange(num_sets)]
            tril_ind_v = [[] for _ in xrange(num_sets)]
            tril_value_ind_v = [[] for _ in xrange(num_sets)]
            disp_2d_split_sec_v = [[] for _ in xrange(num_sets)]
            feature_clicked_v = [[] for _ in xrange(num_sets)]
            news_cnt_short_v = [[] for _ in xrange(num_sets)]
            click_sub_index_2d_v = [[] for _ in xrange(num_sets)]
            for ii in xrange(len(v_user)):
                vali_thread_u[ii % num_sets].append(v_user[ii])
            for ii in xrange(num_sets):
                out=self.data_process_for_placeholder(vali_thread_u[ii])
                # print ("out_val",out['tril_indice'])
                # sys.exit()

                click_2d_v[ii], disp_2d_v[ii], feature_v[ii], sec_cnt_v[ii], tril_ind_v[ii], tril_value_ind_v[ii], \
                disp_2d_split_sec_v[ii], news_cnt_short_v[ii], click_sub_index_2d_v[ii], feature_clicked_v[ii] = out['click_2d_x'], \
                out['disp_2d_x'], \
                out['disp_current_feature_x'], \
                out['sec_cnt_x'], \
                out['tril_indice'], \
                out['tril_value_indice'], \
                out['disp_2d_split_sec'], \
                out['news_cnt_short_x'], \
                out['click_sub_index_2d'], \
                out['feature_clicked_x']

            out2={}
            out2['vali_thread_u']=vali_thread_u 
            out2['click_2d_v']=click_2d_v 
            out2['disp_2d_v']=disp_2d_v 
            out2['feature_v']=feature_v 
            out2['sec_cnt_v']=sec_cnt_v 
            out2['tril_ind_v']=tril_ind_v 
            out2['tril_value_ind_v']=tril_value_ind_v 
            out2['disp_2d_split_sec_v']=disp_2d_split_sec_v 
            out2['news_cnt_short_v']=news_cnt_short_v 
            out2['click_sub_index_2d_v']=click_sub_index_2d_v 
            out2['feature_clicked_v']=feature_clicked_v
            return out2

class UserModelPW(nn.Module):
    """docstring for UserModelPW"""

    def __init__(self, f_dim):
        super(UserModelPW, self).__init__()
        self.f_dim = 3952
        # self.placeholder = {}
        self.hidden_dims = '64-64'
        self.lr = 0.001
        self.pw_dim = 4
        self.band_size = 20
        self.mlp_model = self.mlp(19760, self.hidden_dims, 1, 1000, act_last=False)

    def mlp(self, x_shape, hidden_dims, output_dim, sd, act_last=False):
        hidden_dims = tuple(map(int, hidden_dims.split("-")))
        # print ("hidden_dims",hidden_dims)
        # print ("imp is",x)
        # print (x.shape,x.dtype)
        cur = x_shape
        main_mod = nn.Sequential()
        for i, h in enumerate(hidden_dims):
            main_mod.add_module('Linear-{0}'.format(i), torch.nn.Linear(cur, h))
            main_mod.add_module('act-{0}'.format(i), nn.ELU())
            cur = h

        if act_last:
            main_mod.add_module("Linear_last", torch.nn.Linear(cur, output_dim))
            main_mod.add_module("act_last", nn.ELU())
            return main_mod
        else:
            main_mod.add_module("linear_last", torch.nn.Linear(cur, output_dim))
            return main_mod

    def forward(self, inputs, is_train=False, index=None):
        # input is a dictionaty 
        if is_train == True:

            disp_current_feature = torch.tensor(inputs['disp_current_feature_x'])
            Xs_clicked = torch.tensor(inputs['feature_clicked_x'])
            item_size = torch.tensor(inputs['news_cnt_short_x'])
            section_length = torch.tensor(inputs['sec_cnt_x'])
            click_values = torch.tensor(np.ones(len(inputs['click_2d_x']), dtype=np.float32))
            click_indices = torch.tensor(inputs['click_2d_x'])
            disp_indices = torch.tensor(np.array(np.int64(inputs['disp_2d_x'])))
            disp_2d_split_sec_ind = torch.tensor(inputs['disp_2d_split_sec'])
            cumsum_tril_indices = torch.tensor(np.int64(inputs['tril_indice']))
            cumsum_tril_value_indices = torch.tensor(np.array(inputs['tril_value_indice'], dtype=np.int64))
            click_2d_subindex = torch.tensor(inputs['click_sub_index_2d'])

        else:
            # define the inputs for val/tst here
            # print ("input_val",inputs)

            disp_current_feature = torch.tensor(inputs['feature_v'][index])
            Xs_clicked = torch.tensor(inputs['feature_clicked_v'][index])
            item_size = torch.tensor(inputs['news_cnt_short_v'][index])
            section_length = torch.tensor(inputs['sec_cnt_v'][index])
            click_values = torch.tensor(np.ones(len(inputs['click_2d_v'][index]), dtype=np.float32))
            click_indices = torch.tensor(inputs['click_2d_v'][index])
            disp_indices = torch.tensor(np.array(np.int64(inputs['disp_2d_v'][index])))
            disp_2d_split_sec_ind = torch.tensor(inputs['disp_2d_split_sec_v'][index])
            cumsum_tril_indices = torch.tensor(np.int64(inputs['tril_ind_v'][index]))
            cumsum_tril_value_indices = torch.tensor(np.array(inputs['tril_value_ind_v'][index], dtype=np.int64))
            click_2d_subindex = torch.tensor(inputs['click_sub_index_2d_v'][index])

        denseshape = [section_length, item_size]  # this wont work

        click_history = [[] for _ in xrange(self.pw_dim)]
        # pw
        for ii in xrange(self.pw_dim):
            position_weight = torch.ones(size=[self.band_size]).to(dtype=torch.float64) * 0.0001
            # print (position_weight,cumsum_tril_value_indices)

            cumsum_tril_value = position_weight[cumsum_tril_value_indices]  # tf.gather(position_weight, self.placeholder['cumsum_tril_value_indices'])
          #  print ("cumsum_tril_indices",cumsum_tril_indices)
           # print ("cumsum_tril_value",cumsum_tril_value)

            cumsum_tril_matrix = torch.sparse.FloatTensor(cumsum_tril_indices.t(), cumsum_tril_value,
                                                          [section_length, section_length]).to_dense()
            # print ("cumsum_tril_matrix",cumsum_tril_matrix)
            # print ("Xs_clicked",Xs_clicked.dtype)
            # feature 행렬곱하는 부분
            print(cumsum_tril_matrix.shape)
            print((Xs_clicked.to(dtype=torch.float64)).shape)

            click_history[ii] = torch.matmul(cumsum_tril_matrix,
                                             Xs_clicked.to(dtype=torch.float64))  # Xs_clicked: section by _f_dim

        concat_history = torch.cat(click_history, axis=1)
       # print(concat_history.shape)
        disp_history_feature = concat_history[disp_2d_split_sec_ind]
        #pw_dim 만큼 반복, 3952의 feature벡터가 있고, pw_dim을 4로했음 그런다음 feature를 한줄로, 그다음 추가된 feature를 더해서 한줄로.
        # (4) combine features
        concat_disp_features = torch.reshape(torch.cat([disp_history_feature, disp_current_feature], axis=1),
                                             [-1, self.f_dim * self.pw_dim + self.f_dim])
      #  print(len(concat_disp_features))
       # print(concat_disp_features.shape)
        # (5) compute utility
        # print ("the in put shape s ",concat_disp_features.shape)
        # reward,보상
        u_disp = self.mlp_model(concat_disp_features.float())
        # net.apply(init_weights,sdv)
        # (5)
        exp_u_disp = torch.exp(u_disp)

        sum_exp_disp_ubar_ut = segment_sum(exp_u_disp, disp_2d_split_sec_ind)
        # print ("index",click_2d_subindex)
        sum_click_u_bar_ut = u_disp[click_2d_subindex]

        # (6) loss and precision
        # print ("click_values",click_values)
        # print ("click_indices",click_indices)
        # print ("denseshape",denseshape)
        click_tensor = torch.sparse.FloatTensor(click_indices.t(), click_values, denseshape).to_dense()
        click_cnt = click_tensor.sum(1)
        #유저
        loss_sum = torch.sum(- sum_click_u_bar_ut + torch.log(sum_exp_disp_ubar_ut + 1))
        #클릭의 총합, 평점을 준 횟수의 총합
        event_cnt = torch.sum(click_cnt)
        loss = loss_sum / event_cnt

        exp_disp_ubar_ut = torch.sparse.FloatTensor(disp_indices.t(), torch.reshape(exp_u_disp, (-1,)), denseshape)
        dense_exp_disp_util = exp_disp_ubar_ut.to_dense()
        argmax_click = torch.argmax(click_tensor, dim=1)
        argmax_disp = torch.argmax(dense_exp_disp_util, dim=1)
        # 최고 2개 리턴
        top_2_disp = torch.topk(dense_exp_disp_util, k=2, sorted=False)[1]

        print ("argmax_click",argmax_click.shape)
        #print ("argmax_disp",argmax_disp)
        # for top in top_2_disp:
        #     print ("top_2_disp : " ,top)
        
       # sys.exit()
        #동등 계산
        precision_1_sum = torch.sum((torch.eq(argmax_click, argmax_disp)))
        precision_1 = precision_1_sum / event_cnt

        precision_2_sum = (torch.eq(argmax_click[:, None].to(torch.int64), top_2_disp.to(torch.int64))).sum()
        precision_2 = precision_2_sum / event_cnt

        # self.lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * 0.05  # regularity
        # weight decay can be added in the optimizer for l2 decay
        return loss, precision_1, precision_2, loss_sum, precision_1_sum, precision_2_sum, event_cnt


def segment_sum(data, segment_ids):
    """
    """
    if not all(segment_ids[i] <= segment_ids[i + 1] for i in range(len(segment_ids) - 1)):
        raise AssertionError("elements of segment_ids must be sorted")

    if len(segment_ids.shape) != 1:
        raise AssertionError("segment_ids have be a 1-D tensor")

    if data.shape[0] != segment_ids.shape[0]:
        raise AssertionError("segment_ids should be the same size as dimension 0 of input.")

    num_segments = len(torch.unique(segment_ids))
    return unsorted_segment_sum(data, segment_ids, num_segments)


def unsorted_segment_sum(data, segment_ids, num_segments):
    """
    Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

    :param data: A tensor whose segments are to be summed.
    :param segment_ids: The segment indices tensor.
    :param num_segments: The number of segments.
    :return: A tensor of same data type as the data argument.
    """
    assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

    # segment_ids is a 1-D tensor repeat it to have the same shape as data
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long()
        segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    tensor = torch.zeros(*shape).scatter_add(0, segment_ids, data.float())
    tensor = tensor.type(data.dtype)
    return tensor



def multithread_compute_vali( valid_data, model):
    global vali_sum, vali_cnt

    vali_sum = [0.0, 0.0, 0.0]
    vali_cnt = 0
    threads = []
    for ii in xrange(10):
        # print ("got here")
        # print (dataset.model_type)
        # print (" [dataset.vali_user[ii]]", [dataset.vali_user[ii]])
        # valid_data = dataset.prepare_validation_data(1, [dataset.vali_user[15]]) # is a dict

        # print ("valid_data",valid_data)
        # sys.exit()

        thread = threading.Thread(target=vali_eval, args=(1, ii, valid_data, model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return vali_sum[0] / vali_cnt, vali_sum[1] / vali_cnt, vali_sum[2] / vali_cnt

lock = threading.Lock()

def vali_eval(xx, ii, valid_data, model):
    global vali_sum, vali_cnt
    # print ("dataset.vali_user",dataset.vali_user)

    # valid_data = dataset.prepare_validation_data(1, [dataset.vali_user[ii]]) # is a dict

    # print ("valid_data",valid_data)
    # sys.exit()
    with torch.no_grad():
        _, _, _, loss_sum, precision_1_sum, precision_2_sum, event_cnt = model(valid_data, index=ii)

    lock.acquire()
    vali_sum[0] += loss_sum
    vali_sum[1] += precision_1_sum
    vali_sum[2] += precision_2_sum
    vali_cnt += event_cnt
    lock.release()


lock = threading.Lock()


def multithread_compute_test( test_data, model):
    global test_sum, test_cnt

    num_sets = 1 * 10

    thread_dist = [[] for _ in xrange(10)]
    for ii in xrange(num_sets):
        thread_dist[ii % 10].append(ii)

    test_sum = [0.0, 0.0, 0.0]
    test_cnt = 0
    threads = []
    for ii in xrange(10):
        thread = threading.Thread(target=test_eval, args=(1, thread_dist[ii], test_data, model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return test_sum[0] / test_cnt, test_sum[1] / test_cnt, test_sum[2] / test_cnt


def test_eval(xx, thread_dist, test_data, model):
    global test_sum, test_cnt
    test_thread_eval = [0.0, 0.0, 0.0]
    test_thread_cnt = 0
    for ii in thread_dist:
        with torch.no_grad():
            _, _, _, loss_sum, precision_1_sum, precision_2_sum, event_cnt = model(test_data, index=ii)

        test_thread_eval[0] += loss_sum
        test_thread_eval[1] += precision_1_sum
        test_thread_eval[2] += precision_2_sum
        test_thread_cnt += event_cnt

    lock.acquire()
    test_sum[0] += test_thread_eval[0]
    test_sum[1] += test_thread_eval[1]
    test_sum[2] += test_thread_eval[2]
    test_cnt += test_thread_cnt
    lock.release()


def init_weights(m):
    sd = 1e-3
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight)
        m.weight.data.clamp_(-sd, sd)  # to mimic the normal clmaped weight initilization


def main():

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start" % log_time)

    dataset = Dataset()
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)

    valid_data = dataset.prepare_validation_data(10, dataset.vali_user)
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)
    model = UserModelPW(dataset.f_dim)
    model.apply(init_weights)

    # optimizer = optim.Adam(
    #   [{'params': model.parameters(), 'lr': opts.learning_rate}])

    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

    best_metric = [100000.0, 0.0, 0.0]

    vali_path = "./save_dir/" + '/'
    if not os.path.exists(vali_path):
        os.makedirs(vali_path)

    for i in xrange(10):

        #데이터 학습 usermodel 학습
        # model.train()
        for p in model.parameters():
            p.requires_grad = True
        model.zero_grad()

        training_user_nos = np.random.choice(dataset.train_user, 70, replace=False)

        training_user = dataset.data_process_for_placeholder(training_user_nos)
        for p in model.parameters():
            p.data.clamp_(-1e0, 1e0)

        loss, _, _, _, _, _, _ = model(training_user, is_train=True)
        print ("the loss is",loss)

        loss.backward()
        optimizer.step()

        if np.mod(i, 10) == 0:
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)

        if np.mod(i, 10) == 0:
            if i == 0:
            #여기가 q learning처럼 thread를 모아서 계산하는 곳
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)
            vali_loss_prc = multithread_compute_vali(valid_data, model)
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, first iteration validation complete" % log_time)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: itr%d, vali: %.5f, %.5f, %.5f" %
                  (log_time, i, vali_loss_prc[0], vali_loss_prc[1], vali_loss_prc[2]))

            if vali_loss_prc[0] < best_metric[0]:
                best_metric[0] = vali_loss_prc[0]
                best_save_path = os.path.join(vali_path, 'best-loss')
                torch.save(model.state_dict(), best_save_path)
                # best_save_path = saver.save(sess, best_save_path)
            if vali_loss_prc[1] > best_metric[1]:
                best_metric[1] = vali_loss_prc[1]
                best_save_path = os.path.join(vali_path, 'best-pre1')
                torch.save(model.state_dict(), best_save_path)
            if vali_loss_prc[2] > best_metric[2]:
                best_metric[2] = vali_loss_prc[2]
                best_save_path = os.path.join(vali_path, 'best-pre2')
                torch.save(model.state_dict(), best_save_path)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, iteration %d train complete" % (log_time, i))

    # test
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)

    test_data = dataset.prepare_validation_data(10, dataset.test_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)

    best_save_path = os.path.join(vali_path, 'best-loss')
    model.load_state_dict(torch.load(best_save_path))
    # saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test( test_data, model)
    vali_loss_prc = multithread_compute_vali( valid_data, model)
    print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], vali_loss_prc[0]))

    best_save_path = os.path.join(vali_path, 'best-pre1')
    model.load_state_dict(torch.load(best_save_path))
    # saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test( test_data, model)
    vali_loss_prc = multithread_compute_vali( valid_data, model)
    print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], vali_loss_prc[1]))

    best_save_path = os.path.join(vali_path, 'best-pre2')
    model.load_state_dict(torch.load(best_save_path))
    # saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test( test_data, model)
    vali_loss_prc = multithread_compute_vali( valid_data, model)
    print("test!!!pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], vali_loss_prc[2]))


if __name__ == "__main__":
    print("start!")
    main()
