import numpy as np
from past.builtins import xrange
import pickle
from functools import reduce



class Dataset(object):

    def __init__(self, config):
        self.data_folder = config['data_folder']
        self.dataset = config['dataset']
        self.band_size = config['pw_band_size']

        with open(f"data_behavior.pkl", 'rb') as f:
            data_behavior = pickle.load(f)
            f.close()
        with open(f"new_features.pkl", 'rb') as f:
            item_feature = pickle.load(f)
            f.close()
        self.size_item = item_feature.shape[0] # 46
        self.size_user = data_behavior.shape[0]
        self.f_dim = item_feature.shape[1]
        del item_feature

        item_feature = np.eye(self.size_item)

        # train-valid-test set으로 split
        lst = np.arange(self.size_user)
        np.random.shuffle(lst)
        self.train_user = lst[:int(self.size_user * 0.8)]
        self.valid_user = lst[int(self.size_user * 0.8):int(self.size_user * 0.9)]
        self.test_user = lst[int(self.size_user * 0.9):]
        del lst

        # process data
        k_max = 0
        for db in np.nditer(data_behavior[:, 1], flags=["refs_ok"]):
            d_b = db.tolist()
            for i in xrange(len(d_b)):
                k_max = max(k_max, len(d_b[i]))

        self.data_click = [[] for x in xrange(self.size_user)]
        self.data_disp = [[] for x in xrange(self.size_user)]
        self.data_time = np.zeros(self.size_user, dtype=np.int16)
        self.data_news_cnt = np.zeros(self.size_user, dtype=np.int16)
        self.feature = [[] for x in xrange(self.size_user)]
        self.feature_click = [[] for x in xrange(self.size_user)]

        for user in xrange(self.size_user):
            # (1) count number of clicks
            click_t = 0
            num_events = len(data_behavior[user][1]) # timestamp 수
            click_t += num_events
            self.data_time[user] = click_t
            self.feature_click[user] = np.zeros([click_t, self.f_dim])
        self.feature_click = np.array(self.feature_click, dtype=object)
        del user, click_t, num_events

        for user in xrange(self.size_user):
            # (2)
            if user % 500 == 0:
                print(f"{user}/{self.size_user}")
            news_dict = {}
            num_events = len(data_behavior[user][1])

            click_t = 0
            for event in xrange(num_events):
                disp_list = data_behavior[user][1][event] # event 시점에 user가 본 영화
                pick_id = data_behavior[user][2][event] # event 시점에 user가 클릭한 영화
                for _ in xrange(len(disp_list)):
                    id_ = disp_list[_]
                    if id_ not in news_dict.keys():
                        news_dict[id_] = len(news_dict)  # for each user, news id start from 0

                id_ = pick_id

                for _ in xrange(len(id_)):
                    i = id_[_]
                    self.data_click[user] = [click_t, news_dict[i]] # 유저가 클릭한 시점, 영화아이디
                    self.feature_click[user][click_t] = item_feature[i] # 유저가 클릭한 시점의 아이템 피쳐
                for _ in xrange(len(disp_list)):
                    idd = disp_list[_]
                    self.data_disp[user].append([click_t, news_dict[idd]]) # 유저가 클릭한 시점, 그 시점에 display된 영화 아이디
                click_t += 1  # splitter a event with 2 clickings to 2 events

            self.data_news_cnt[user] = len(news_dict) # 유저가 본 영화 수 (중복X)
            self.feature[user] = np.zeros([self.data_news_cnt[user], self.f_dim])

            for id_ in news_dict:
                self.feature[user][news_dict[id_]] = item_feature[id_]

        del user
        self.max_disp_size = k_max # 최대로 보여진 횟수

    def random_split_user(self):  # resplit용, train, valid, test 셋 다시 나눔
        num_users = len(self.train_user) + len(self.valid_user) + len(self.test_user)
        shuffle_order = np.arange(num_users)
        np.random.shuffle(shuffle_order)
        self.train_user = shuffle_order[0:len(self.train_user)].tolist()
        self.valid_user = shuffle_order[len(self.train_user):len(self.train_user) + len(self.valid_user)].tolist()
        self.test_user = shuffle_order[len(self.train_user) + len(self.valid_user):].tolist()

    def data_process_for_placeholder(self, user_set):  # 매트릭스 연산에 사용되는 부분.

        # print ("user_set",user_set)
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

        for u in user_set:
            t_indice = []
            for kk in xrange(min(self.band_size - 1, self.data_time[u] - 1)):
                t_indice += map(lambda x: [x + kk + 1 + sec_cnt_x, x + sec_cnt_x], np.arange(self.data_time[u] - (kk + 1)))

            tril_indice += t_indice
            tril_value_indice += map(lambda x: (x[0] - x[1] - 1), t_indice)

            click_2d_tmp = list(map(lambda x: [x[0] + sec_cnt_x, x[1]], [self.data_click[u]]))
            click_2d_tmp = reduce(lambda x: x, click_2d_tmp)
            click_2d_tmp = list(click_2d_tmp)
            click_2d_x.append(click_2d_tmp)

            disp_2d_tmp = map(lambda x: [x[0] + sec_cnt_x, x[1]], self.data_disp[u])
            disp_2d_tmp = list(disp_2d_tmp)

            click_sub_index_tmp = map(lambda x: disp_2d_tmp.index(x), [click_2d_tmp])
            click_sub_index_tmp = list(click_sub_index_tmp)

            click_sub_index_2d.append(list(map(lambda x: x + len(disp_2d_x), click_sub_index_tmp)))

            disp_2d_x += disp_2d_tmp

            disp_2d_split_sec += list(map(lambda x: x[0] + sec_cnt_x, self.data_disp[u]))
            sec_cnt_x += self.data_time[u]
            news_cnt_short_x = max(news_cnt_short_x, self.data_news_cnt[u])
            news_cnt_x += self.data_news_cnt[u]
            disp_current_feature_x += map(lambda x: self.feature[u][x], [idd[1] for idd in self.data_disp[u]])
            feature_clicked_x += (self.feature_click[u].tolist())

        out1 = {}
        out1['click_2d_x'] = click_2d_x
        out1['disp_2d_x'] = disp_2d_x
        out1['disp_current_feature_x'] = disp_current_feature_x
        out1['sec_cnt_x'] = sec_cnt_x
        out1['tril_indice'] = tril_indice
        out1['tril_value_indice'] = tril_value_indice
        out1['disp_2d_split_sec'] = disp_2d_split_sec
        out1['news_cnt_short_x'] = news_cnt_short_x
        out1['click_sub_index_2d'] = click_sub_index_2d
        out1['feature_clicked_x'] = feature_clicked_x

        return out1

    def prepare_validation_data(self, num_sets, valid_user):

        valid_thread_u = [[] for _ in xrange(num_sets)]
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
        for ii in xrange(len(valid_user)):
            valid_thread_u[ii % num_sets].append(valid_user[ii])
        for ii in xrange(num_sets):
            out = self.data_process_for_placeholder(valid_thread_u[ii])

            click_2d_v[ii], disp_2d_v[ii], feature_v[ii], sec_cnt_v[ii], tril_ind_v[ii], tril_value_ind_v[ii], \
            disp_2d_split_sec_v[ii], news_cnt_short_v[ii], click_sub_index_2d_v[ii], feature_clicked_v[ii] = out['click_2d_x'], out['disp_2d_x'], \
                                                                                                             out['disp_current_feature_x'], out['sec_cnt_x'], \
                                                                                                             out['tril_indice'], out['tril_value_indice'], \
                                                                                                             out['disp_2d_split_sec'], out['news_cnt_short_x'], \
                                                                                                             out['click_sub_index_2d'], out['feature_clicked_x']

        out2 = {}
        out2['valid_thread_u'] = valid_thread_u
        out2['click_2d_v'] = click_2d_v
        out2['disp_2d_v'] = disp_2d_v
        out2['feature_v'] = feature_v
        out2['sec_cnt_v'] = sec_cnt_v
        out2['tril_ind_v'] = tril_ind_v
        out2['tril_value_ind_v'] = tril_value_ind_v
        out2['disp_2d_split_sec_v'] = disp_2d_split_sec_v
        out2['news_cnt_short_v'] = news_cnt_short_v
        out2['click_sub_index_2d_v'] = click_sub_index_2d_v
        out2['feature_clicked_v'] = feature_clicked_v
        return out2


