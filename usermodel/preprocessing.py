import pandas as pd
import json
import numpy as np
from past.builtins import xrange
import pickle

def preprocess(config):
    filename = config['data_folder'] + config['dataset']
    data = pd.read_csv(filename)

    sizes = data.nunique()
    size_user = sizes['user_new_index']
    size_movie = sizes['movie_new_index']
    data_user = data.groupby(by='user_new_index') # userid별로 grouping
    data_behavior = [[] for _ in xrange(size_user)]
    # xrange 사용 이유: range는 list type으로, range가 커질수록 메모리 사용량이 커지지만, xrange는 xrange type으로 range가 커져도 메모리 사용량이 일정함

    sum_length = 0
    event_cnt = 0

    for user in xrange(size_user):
        data_behavior[user] = [[], [], []]
        data_behavior[user][0] = user
        data_u = data_user.get_group(user)
        data_u_time = data_u.groupby(by='new_timestamp') # user가 본 영화들을 timestamp끼리 묶음
        time_set = np.array(list(set(data_u['new_timestamp'])))
        time_set.sort() #timestamp순으로 정렬

        for t in xrange(len(time_set)):
            display_set = data_u_time.get_group(time_set[t]) # t 시점에서 본 영화들
            event_cnt += 1
            sum_length += len(display_set) # t 시점의 전체 아이템 수
            data_behavior[user][1].append(list(display_set['movie_new_index']))
            data_behavior[user][2].append(display_set[display_set.is_click == 1]['movie_new_index'].to_list()) # t 시점에서 본 영화들 중 클릭한 것
    data_behavior = np.array(data_behavior, dtype=object)
    print(data_behavior)
    to_del = []
    for i in range(len(data_behavior)):
        sum_ = 0
        for j in data_behavior[i][2]:
            sum_ += len(j)
        if sum_ == 0:
            to_del.append(i)

    to_del = sorted(to_del, reverse=True)

    for i in to_del:
        data_behavior = np.delete(data_behavior, i, axis=0)

    with open(f"data_behavior.pkl", 'wb') as f:
         pickle.dump(data_behavior, f)

    del data_behavior
    print("save data_behavior: succeed")
    new_features = np.eye(size_movie) # 영화 아이디 원-핫 인코딩 한 것

    with open(f"new_features.pkl", 'wb') as f:
        pickle.dump(new_features, f)
    del new_features
    print("done")
# process_data.py
# data_behavior format
# data_behavior[user][0]: userId
# data_behavior[user][1][t]: t에서의 displayed list
# data_behavior[user][2][t]: t에서의 선택된 아이템 (is_clicked == 1)


if __name__ == "__main__":
    with open("config.json") as f:
        config = json.load(f)
    f.close()
    preprocess(config)