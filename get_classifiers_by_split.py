#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import argparse
import pandas as pd
import pickle

from mice_utils import get_split_info, get_labels_subjects

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold,ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


def main(segment_param_string, count_param_string, reg_type='l2', n_segment_to_use=None, task=None, LOO=False,do_pooled=False):

    strains = ('BXD87', 'DBA2', 'C57B6')  # tsc
    tscs = ('Het', 'WT')  # genotype

    segment_info_train = pickle.load(open("data/hpc_segment_info_valid_" + segment_param_string + ".pickle", "rb"))
#    segment_info_train = pickle.load(open("data/hpc_segment_info_train_" + segment_param_string + ".pickle", "rb"))
    segment_info_test = pickle.load(open("data/hpc_segment_info_test_" + segment_param_string + ".pickle", "rb"))

    all_counts = pickle.load(open("data/hpc_count_valid_" + count_param_string + ".pickle", "rb"))
#    all_counts = pickle.load(open("data/hpc_count_train_" + count_param_string + ".pickle", "rb"))
    counts_test = pickle.load(open("data/hpc_count_test_" + count_param_string + ".pickle", "rb"))

    label_train, subject_train, subject_index2name = get_labels_subjects(segment_info_train)
    label_test, subject_test, subject_index2name_test = get_labels_subjects(segment_info_test)
    # print(np.unique(subject_test),np.unique(subject_train))
    # print(subject_index2name.values(),subject_index2name_test.values())
    # print(len(set(subject_index2name.values()).union(set(subject_index2name_test.values()))))
    # print(len(set(subject_index2name.values()))+len(set(subject_index2name_test.values())))

    X_train = np.hstack([np.vstack([all_counts[d, s] for s in segment_info_train]) for d in segment_info_train])
    X_test = np.hstack([np.vstack([counts_test[d, s] for s in segment_info_test]) for d in segment_info_test])

    if task is None:
        task = 'type'
    task_str = '' if task == type else task
    pool_string = 'pooled' if do_pooled else ''
    
    if LOO:
        n_groups = len(np.unique(subject_test))
    else:
        n_groups = 5
        # two fold based on dictionary (done 5 random splits)
    task_str += 'LOO' if LOO else ''

    group_kfold_test = StratifiedGroupKFold(n_splits=n_groups)
    # for hyper-parameter selection, internal k-fold is conducted
    group_kfold = StratifiedGroupKFold(n_splits=3)

    # Classifier architecture
    pipe = Pipeline([
        ('tfidf', TfidfTransformer()),
        ('clf', LogisticRegression())
    ])

    # Classifier hyper-parameters
    C_grid = np.logspace(-1, 4, 15)
    # # Define hyperparameters for grid search
    solver_type = 'saga' if reg_type == 'l1' else 'lbfgs'
    print(solver_type)
    param_grid = {
        'clf__C': C_grid,
        'clf__penalty': [reg_type],
        'clf__max_iter': [1000],
        'clf__solver': [solver_type]
    }

    pipes = []  # list of pipes used
    for k, (train_index, test_index) in enumerate(group_kfold_test.split(X_test, label_test, subject_test)):
        # Data in this training is all of training and some split of test mice
        X = np.vstack((X_train, X_test[train_index]))
        y = np.concatenate((label_train, label_test[train_index]))
        groups = np.concatenate((subject_train, np.max(subject_train)+subject_test[train_index]))
        train_group_max = np.max(subject_train)
        # If groups[i] < np.max(subject_train) then i is from train
        if n_segment_to_use is not None:
            rng = np.random.default_rng(0)
            train_idx = rng.choice(X.shape[0], size=n_segment_to_use * 6, replace=False)
            y = y[train_idx]
            groups = groups[train_idx]
            X = X[train_idx]
            n_segment_to_use_str = str(n_segment_to_use)
        else:
            n_segment_to_use_str = ''

        print(f"Train Fold {k}:")
        # call training function with labels for task
        if task in ['strain','tsc','type']:
            if task == 'strain':
                y = y // 2
            elif task == 'tsc':
                y = y % 2
            print(f"Train Fold {k} for {task_str}")
            pipe_i = clone(pipe)
            pipes.append(get_classifier(pipe_i, X, y, groups, train_group_max, group_kfold, param_grid,do_pooled))

        elif task in ['strain_on_tsc','tsc_on_strain']:
            if task == 'strain_on_tsc':
                cond_labels = y // 2
                given_labels = y % 2
                given_label_names = tscs
            elif task == 'tsc_on_strain':
                cond_labels = y % 2
                given_labels = y // 2
                given_label_names = strains
            cond_pipes = []
            for _,given_label in enumerate(np.unique(given_labels)):
                pipe_i = clone(pipe)
                print(f"Train fold {k} for task {given_label_names[given_label]}")
                cond_pipes.append(get_classifier(pipe_i, X[given_labels == given_label],
                                                 cond_labels[given_labels == given_label],
                                                 groups[given_labels == given_label], train_group_max, group_kfold,
                                                 param_grid,do_pooled))
            pipes.append(cond_pipes)
        else:
            print('task not supported')
            raise

    filename = 'models/classifiers_l2_'+pool_string + count_param_string + n_segment_to_use_str + task_str
    pickle.dump(pipes, open(filename + ".pkl", "wb"))

def get_classifier(pipe_i, X, y, groups, train_group_max, group_kfold,param_grid, do_pooled=False):
    # 5 fold cross validation of the combined training per test fold
    train_acc = np.zeros((group_kfold.get_n_splits(), len(ParameterGrid(param_grid))))
    acc = np.zeros((group_kfold.get_n_splits(), len(ParameterGrid(param_grid))))
    pooled_acc = np.zeros((group_kfold.get_n_splits(), len(ParameterGrid(param_grid))))
    maj_acc = np.zeros((group_kfold.get_n_splits(), len(ParameterGrid(param_grid))))
    train_indices = np.flatnonzero(groups <= train_group_max)
    test_indices = np.flatnonzero(groups > train_group_max)
    print(
    f"  Train:  {np.bincount(y[train_indices], minlength=1+np.max(y))}, {np.bincount(groups[train_indices], minlength=1+np.max(groups))}")
    print(
    f"  Test:  {np.bincount(y[test_indices], minlength=1+np.max(y))}, {np.bincount(groups[test_indices], minlength=1+np.max(groups))}")

    #    for i, (in_index, out_index) in enumerate(group_kfold.split(X, y, groups)):
    for i, (val_in_index, val_out_index) in enumerate(group_kfold.split(X[groups > train_group_max],
                                                                        y[groups > train_group_max],
                                                                        groups[groups > train_group_max])):
        in_index = np.concatenate((train_indices, test_indices[val_in_index]))
        out_index = test_indices[val_out_index]
        print(np.intersect1d(groups[in_index],groups[out_index]))
        subs_in_train, indices_train, inverses_train = np.unique(groups[in_index], return_index=True, return_inverse=True)
        subs_in_test, indices, inverses = np.unique(groups[out_index], return_index=True, return_inverse=True)
        print(
            f"  Held-in: {np.bincount(y[in_index[indices_train]], minlength=6)}, {np.bincount(y[in_index], minlength=1+np.max(y))}, {np.bincount(groups[in_index], minlength=1+np.max(groups))}")
        print(
            f"  Held-out: {np.bincount(y[out_index[indices]], minlength=6)}, {np.bincount(y[out_index], minlength=1+np.max(y))}, {np.bincount(groups[out_index], minlength=1+np.max(groups))}")
        ypooled = y[out_index][indices]
        Xpooled = X[out_index][indices]
        for l, index in enumerate(indices):
            Xpooled[l] = np.sum(X[out_index][inverses == l], axis=0)
        if do_pooled:
            y_train_pooled = y[in_index][indices_train]
            X_train_pooled = X[in_index][indices_train]
            for l, index in enumerate(indices_train):
                X_train_pooled[l] = np.sum(X[in_index][inverses_train == l], axis=0)

        
        for j, params in enumerate(ParameterGrid(param_grid)):
            if do_pooled:
                pipe_i.set_params(**params).fit(X_train_pooled, y_train_pooled)
            else:
                pipe_i.set_params(**params).fit(X[in_index], y[in_index])
                
            train_acc[i, j] = pipe_i.score(X[in_index], y[in_index])
            acc[i, j] = pipe_i.score(X[out_index], y[out_index])
            pooled_acc[i, j] = pipe_i.score(Xpooled, ypooled)
            log_prob = pipe_i.predict_log_proba(X[out_index])
            y_pred_maj = np.zeros_like(ypooled)

            for l, index in enumerate(indices):
                y_pred_maj[l] = np.argmax(np.sum(log_prob[inverses == l], axis=0))
            maj_acc[i, j] = np.mean(y_pred_maj == ypooled)  # m
            print("{},{}: train:{:.2f} valid:{:.2f} pooled:{:.2f} majority:{:.2f}".format(i, j, train_acc[i, j],
                                                                                          acc[i, j], pooled_acc[i, j],
                                                                                          maj_acc[i, j]))

    #        best_param_idx = np.argsort(-np.mean(acc, axis=0))[0]
    best_param_idx = np.lexsort(-np.vstack((np.mean(acc, axis=0), np.mean(pooled_acc, axis=0))))[0]
    best_params = ParameterGrid(param_grid)[best_param_idx]
    pipe_i.set_params(**best_params).fit(X, y)
    best_train_acc = pipe_i.score(X, y)
    print(best_param_idx,best_params, best_train_acc)
    return pipe_i

def consolidate_counts_subset(test_fold, split,train_valid_test,param_string, n_segments):
    strains = ('BXD87', 'DBA2', 'C57B6')  # tsc
    tscs = ('Het', 'WT')  # genotype
    print('Starting consolidate counts', test_fold, split)
    segment_filename = "data/hpc_segment_info_" + train_valid_test + "_" + str(n_segments) + "_" + str(split) + str(test_fold) + param_string + ".pickle"
    count_filename = "data/hpc_count_" + train_valid_test + "_" + str(n_segments) + str(split) + str(test_fold) + param_string + ".pickle"
    try:
        segment_info = pickle.load(open(segment_filename, "rb"))
        all_counts = pickle.load(open(count_filename, "rb"))
        print('Consolidated data ready!')
        return
    except:
        print('Running data consolidation')

    hpc_segment_info = dict()
    hpc_all_counts = dict()
    n_success = 0

    for strain in strains:
        for tsc in tscs:
            key_segment = (split, test_fold, strain, tsc)
            try:
                segment_fname = ("data/segment_info_" + train_valid_test + "_" + str(n_segments) + '_' + str(split) + str(test_fold) + param_string + strain + tsc + ".pickle")
                print(segment_fname)
                hpc_segment_info[key_segment] = pickle.load(open(segment_fname, "rb"))
                count_fname = ("data/count_" + train_valid_test + "_" + str(n_segments) + str(split) + str(test_fold) + param_string + strain + tsc + ".pickle")
                print(count_fname)
                counts = pickle.load(open(count_fname, "rb"))
                for key_dict in [(split, test_fold, strain2, bg2) for strain2 in strains for bg2 in tscs]:
                    hpc_all_counts[key_dict, key_segment] = counts[key_dict]
                n_success += 1
            except:
                print(tsc, strain, split, test_fold, train_valid_test, ' is not finished')

    print(n_success, len(strains) * len(tscs))
    if n_success == len(strains) * len(tscs):
        print('Writing data')
        pickle.dump(hpc_segment_info, open(segment_filename, "wb"))
        pickle.dump(hpc_all_counts, open(count_filename, "wb"))
    else:
        print('Not all types finished')
        raise

def consolidate_counts(test_fold, split,param_string, n_segments):
    strains = ('BXD87', 'DBA2', 'C57B6')  # tsc
    tscs = ('Het', 'WT')  # genotype
    print('Starting consolidate counts', test_fold, split)
    for train_valid_test in ['valid', 'test']:
        segment_filename = "data/hpc_segment_info_" + train_valid_test + "_" + str(n_segments) + "_" + str(split) + str(test_fold) + param_string + ".pickle"
        count_filename = "data/hpc_count_" + train_valid_test + "_" + str(n_segments) + str(split) + str(test_fold) + param_string + ".pickle"
        try:
            segment_info = pickle.load(open(segment_filename, "rb"))
            all_counts = pickle.load(open(count_filename, "rb"))
            print('Consolidated data ready!')
            continue
        except:
            print('Running data consolidation')

        hpc_segment_info = dict()
        hpc_all_counts = dict()
        n_success = 0

        for strain in strains:
            for tsc in tscs:
                key_segment = (split, test_fold, strain, tsc)
                try:
                    segment_fname = ("data/segment_info_" + train_valid_test + "_" + str(n_segments) + '_' + str(split) + str(test_fold) + param_string + strain + tsc + ".pickle")
                    print(segment_fname)
                    hpc_segment_info[key_segment] = pickle.load(open(segment_fname, "rb"))
                    count_fname = ("data/count_" + train_valid_test + "_" + str(n_segments) + str(split) + str(test_fold) + param_string + strain + tsc + ".pickle")
                    print(count_fname)
                    counts = pickle.load(open(count_fname, "rb"))
                    for key_dict in [(split, test_fold, strain2, bg2) for strain2 in strains for bg2 in tscs]:
                        hpc_all_counts[key_dict, key_segment] = counts[key_dict]
                    n_success += 1
                except:
                    print(tsc, strain, split, test_fold, train_valid_test, ' is not finished')

        print(n_success, len(strains) * len(tscs))
        if n_success == len(strains) * len(tscs):
            print('Writing data')
            pickle.dump(hpc_segment_info, open(segment_filename, "wb"))
            pickle.dump(hpc_all_counts, open(count_filename, "wb"))
        else:
            print('Not all types finished')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("type", "tsc", "strain","tsc_on_strain","strain_on_tsc"), default="type",
                        help="which classification task")
    parser.add_argument("--train_segments", type=int,
                        help="how many segments to get from each class for training to be used to subset")
    parser.add_argument("--split", type=int, default=0,
                        help="which random split for dictionary training")
    parser.add_argument("--fold", type=int, default=0,
                        help="which fold in the split for dictionary training")
    parser.add_argument("--class1", type=int, choices=[0,1,2],
                        help="index of class subtype-1")
    parser.add_argument("--class2", type=int, choices=[0,1],
                        help="index of class subtype-2")
    parser.add_argument("--windows", type=int, default=40000,
                        help="how many windows were used in training")
    parser.add_argument("--window_length_s", type=float, default=2,
                        help="how many seconds each window")
    parser.add_argument("-k", "--n_centroids", type=int, default=200,
                        help="how many centroids")
    parser.add_argument("--centroid_length_s", type=float, default=1,
                        help="how many seconds is each waveform/centroid")
    parser.add_argument("--segments", type=int, default=480,
                        help="how many segments to get from each class")
    parser.add_argument("--hours_in_segment", type=float,
                        help="how many hours in each segment for the count vectors")
    parser.add_argument('--spectral', dest='use_spectral', action='store_true')
    parser.set_defaults(use_spectral=False)
    parser.add_argument("--df", type=str, default='Jax_mice_with_splits_df.pickle',
                        help="filename in the meta/data path")
    parser.add_argument('--l1', dest='use_l1', action='store_true')
    parser.add_argument('--l2', dest='use_l1', action='store_false')
    parser.set_defaults(use_l1=False)
    parser.add_argument('--loo', dest='loo', action='store_true')
    parser.set_defaults(loo=False)
    parser.add_argument('--pool', dest='do_pooled', action='store_true')
    parser.set_defaults(do_pooled=False)
    parser.set_defaults(use_sign_invariant=False)
    parser.add_argument('--sign_invariant', dest='use_sign_invariant', action='store_true')
    parser.add_argument('--sphere', dest='use_sphere', action='store_true')
    parser.set_defaults(use_sphere=False)
    parser.add_argument('--svd', dest='use_svd', action='store_true')
    parser.set_defaults(use_svd=False)

    args = parser.parse_args()

    split = args.split
    test_fold = args.fold
    n_centroids = args.n_centroids
    n_windows = args.windows
    n_seg = args.segments

    mice_df = pd.read_pickle('meta-data/'+args.df)
    sfreq = mice_df['sfreq'].iloc[0]  # assume all mice are the same, should be checked on any new dataset
    centroid_len = int(sfreq * args.centroid_length_s)  # in samples
    window_length = int(sfreq * args.window_length_s)  # in samples

    if args.use_sign_invariant:
        str_si = '_si'
    else:
        str_si = '_sv'

    if args.use_sphere:
        str_sph = '_sph'
    else:
        str_sph = ''

    if args.use_svd:
        str_svd = '_svd'
    else:
        str_svd = ''


    if args.use_spectral:
        nfft = args.centroid_length_s
        str_nfft = '' if nfft == 1 else '_'+str(nfft)
        dict_param_string = 'spectral' + '_' + str(n_windows) + '_' + str(window_length) + '_' + str(n_centroids) + str_nfft
    else:
        dict_param_string = str(n_windows) + '_' + str(window_length) + '_' + str(n_centroids) + '_' + str(centroid_len) + str_si+str_sph+ str_svd

    if args.use_l1:
        reg_type = 'l1'
    else:
        reg_type = 'l2'
    segment_param_string = str(n_seg) + '_' + str(split) + str(test_fold) + dict_param_string
    print(f'Going to run consolidation on {split}{test_fold} {dict_param_string} {n_seg}')
    consolidate_counts(test_fold, split, dict_param_string, n_seg)
    count_param_string = str(n_seg) + str(split) + str(test_fold) + dict_param_string
    main(segment_param_string, count_param_string, reg_type, args.train_segments, args.task, args.loo, args.do_pooled)
