import torch
from torch import nn
import tqdm
from torch.utils.data import DataLoader
import json
from usermodel import UserModelPW
from data_processing import *
import torch.optim as optim

import datetime
import numpy as np
import os
import threading



'''
@for i in range(pot.iter):
    train_model
    if iter//some_fixed_no==0:
        validate
test_model 
plot_figs
'''


def multithread_compute_vali(config, valid_data, model):
    global valid_sum, valid_cnt

    valid_sum = [0.0, 0.0, 0.0]
    valid_cnt = 0
    threads = []
    for ii in xrange(config['num_thread']):

        thread = threading.Thread(target=valid_eval, args=(1, ii, config, valid_data, model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return valid_sum[0] / valid_cnt, valid_sum[1] / valid_cnt, valid_sum[2] / valid_cnt


lock = threading.Lock() # 해당 쓰레드만 공유 데이터에 접근할 수 있도록


def valid_eval(xx, ii, config, valid_data, model):
    global valid_sum, valid_cnt
    # print ("dataset.valid_user",dataset.valid_user)

    # valid_data = dataset.prepare_validation_data(1, [dataset.valid_user[ii]]) # is a dict

    # print ("valid_data",valid_data)
    # sys.exit()
    with torch.no_grad():
        _, _, _, loss_sum, precision_1_sum, precision_2_sum, event_cnt = model(valid_data, index=ii)

    lock.acquire() # 잠금
    valid_sum[0] += loss_sum
    valid_sum[1] += precision_1_sum
    valid_sum[2] += precision_2_sum
    valid_cnt += event_cnt
    lock.release() # 해제


lock = threading.Lock()


def multithread_compute_test(config, test_data, model):
    global test_sum, test_cnt

    num_sets = 1 * config['num_thread']

    thread_dist = [[] for _ in xrange(config['num_thread'])]
    for ii in xrange(num_sets):
        thread_dist[ii % config['num_thread']].append(ii)

    test_sum = [0.0, 0.0, 0.0]
    test_cnt = 0
    threads = []
    for ii in xrange(config['num_thread']):
        thread = threading.Thread(target=test_eval, args=(1, thread_dist[ii], config, test_data, model))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return test_sum[0] / test_cnt, test_sum[1] / test_cnt, test_sum[2] / test_cnt


def test_eval(xx, thread_dist, config, test_data, model):
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


def main(config):
    # pp.pprint(vars(config))

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start" % log_time)

    dataset = Dataset(config)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, load data completed" % log_time)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare vali data" % log_time)

    valid_data = dataset.prepare_validation_data(config['num_thread'], dataset.valid_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare validation data, completed" % log_time)

    model = UserModelPW(dataset.f_dim, config)

    model.apply(init_weights)

    # optimizer = optim.Adam(
    #   [{'params': model.parameters(), 'lr': config['learning_rate']}])

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], betas=(0.5, 0.999)) # Adam optimizer 사

    best_metric = [100000.0, 0.0, 0.0]

    valid_path = config['save_dir'] + '/'
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)

    # training_dataloader = DataLoader(training_dataset, batch_size=config.['batch_size'], num_workers=1) # need to change the dataloader

    for i in xrange(config['num_itrs']):

        # model.train()
        for p in model.parameters():
            p.requires_grad = True
        model.zero_grad() # 가중치 초기화

        training_user_nos = np.random.choice(dataset.train_user, config['batch_size'], replace=False)

        training_user = dataset.data_process_for_placeholder(training_user_nos)
        for p in model.parameters():
            p.data.clamp_(-1e0, 1e0)

        # for batch_id, batch in enumerate(tqdm(training_dataloader)): # the original code does not iterate over entire batch , so change this one

        loss, _, _, _, _, _, _ = model(training_user, is_train=True)
        # print ("the loss is",loss)

        loss.backward() # 
        optimizer.step()

        if np.mod(i, 10) == 0:
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)

        if np.mod(i, 10) == 0:
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, start first iteration validation" % log_time)
            valid_loss_prc = multithread_compute_vali(config, valid_data, model)
            if i == 0:
                log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print("%s, first iteration validation complete" % log_time)

            log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print("%s: itr%d, vali: %.5f, %.5f, %.5f" %
                  (log_time, i, valid_loss_prc[0], valid_loss_prc[1], valid_loss_prc[2]))

            if valid_loss_prc[0] < best_metric[0]:
                best_metric[0] = valid_loss_prc[0]
                best_save_path = os.path.join(valid_path, 'best-loss')
                torch.save(model.state_dict(), best_save_path)
                # best_save_path = saver.save(sess, best_save_path)
            if valid_loss_prc[1] > best_metric[1]:
                best_metric[1] = valid_loss_prc[1]
                best_save_path = os.path.join(valid_path, 'best-pre1')
                torch.save(model.state_dict(), best_save_path)
            if valid_loss_prc[2] > best_metric[2]:
                best_metric[2] = valid_loss_prc[2]
                best_save_path = os.path.join(valid_path, 'best-pre2')
                torch.save(model.state_dict(), best_save_path)

        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("%s, iteration %d train complete" % (log_time, i))

    # test
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, start prepare test data" % log_time)

    test_data = dataset.prepare_validation_data(config['num_thread'], dataset.test_user)

    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("%s, prepare test data end" % log_time)

    best_save_path = os.path.join(valid_path, 'best-loss')
    model.load_state_dict(torch.load(best_save_path))
    # saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test(config, test_data, model)
    valid_loss_prc = multithread_compute_vali(config, valid_data, model)
    print("test!!!loss!!!, test: %.5f, vali: %.5f" % (test_loss_prc[0], valid_loss_prc[0]))

    best_save_path = os.path.join(valid_path, 'best-pre1')
    model.load_state_dict(torch.load(best_save_path))
    # saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test(config, test_data, model)
    valid_loss_prc = multithread_compute_vali(config, valid_data, model)
    print("test!!!pre1!!!, test: %.5f, vali: %.5f" % (test_loss_prc[1], valid_loss_prc[1]))

    best_save_path = os.path.join(valid_path, 'best-pre2')
    model.load_state_dict(torch.load(best_save_path))
    # saver.restore(sess, best_save_path)
    test_loss_prc = multithread_compute_test(config, test_data, model)
    valid_loss_prc = multithread_compute_vali(config, valid_data, model)
    print("test!!!pre2!!!, test: %.5f, vali: %.5f" % (test_loss_prc[2], valid_loss_prc[2]))


if __name__ == "__main__":
    with open('config.json') as f:
        config = json.load(f)
    f.close()
    main(config)



