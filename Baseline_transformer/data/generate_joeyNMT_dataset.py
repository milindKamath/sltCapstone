import os
from PIL import Image
import pandas as pd
import numpy as np
import torch
import pickle
import gzip
import csv
import pdb
import argparse

def gen_csl(path):
    train_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/old_CSL/train_1792_split_sentences.csv'
    test_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/old_CSL/test_448_split_sentences.csv'

    train_dict = get_csl_caption_detail(train_corpus)
    test_dict = get_csl_caption_detail(test_corpus)

    train_list = []
    test_list = []

    numpyFiles = sorted(os.listdir(path))

    for files in numpyFiles:
        type='train'
        dict = {}
        numpy_dataset_path = (os.path.join(path, files))
        numpy_dataset = np.load(numpy_dataset_path)
        torch_tensor = torch.from_numpy(numpy_dataset).float()
        if files[:-4] in train_dict:
            type = 'train'
        elif files[:-4] in test_dict:
            type = 'test'
        else:
            continue

        if type =='test':
            dict['name'] = files[:-4]
            dict['gloss'] = test_dict[files[:-4]][1]
            dict['text'] = test_dict[files[:-4]][0]
            dict['signer'] = test_dict[files[:-4]][2]
            dict['sign'] = torch_tensor
            test_list.append(dict)
        if type =='train':
            dict['name'] = files[:-4]
            dict['gloss'] = train_dict[files[:-4]][1]
            dict['text'] = train_dict[files[:-4]][0]
            dict['signer'] = train_dict[files[:-4]][2]
            dict['sign'] = torch_tensor
            train_list.append(dict)

    save(train_list, './' + 'CSL_old_dataset.train')

    save(test_list, './' + 'CSL_old_dataset.test')


def gen_asl(path):
    train_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/new_final_filtered_csv/new_asl_2_preprocessed_train.csv'
    test_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase2/Models_scripts/Priyanshu/ASL/new_final_filtered_csv/new_asl_2_preprocessed_test.csv'

    train_dict = get_asl_caption_detail(train_corpus)
    test_dict = get_asl_caption_detail(test_corpus)

    train_list = []
    test_list = []

    numpyFiles = sorted(os.listdir(path))

    for files in numpyFiles:
        type='train'
        dict = {}
        numpy_dataset_path = (os.path.join(path, files))
        numpy_dataset = np.load(numpy_dataset_path)
        torch_tensor = torch.from_numpy(numpy_dataset).float()
        if files[:-4] in train_dict:
            type = 'train'
        elif files[:-4] in test_dict:
            type = 'test'
        else:
            continue

        if type =='test':
            dict['name'] = files[:-4]
            dict['gloss'] = test_dict[files[:-4]][1]
            dict['text'] = test_dict[files[:-4]][0]
            dict['signer'] = test_dict[files[:-4]][2]
            dict['sign'] = torch_tensor
            test_list.append(dict)
        if type =='train':
            dict['name'] = files[:-4]
            dict['gloss'] = train_dict[files[:-4]][1]
            dict['text'] = train_dict[files[:-4]][0]
            dict['signer'] = train_dict[files[:-4]][2]
            dict['sign'] = torch_tensor
            train_list.append(dict)

    save(train_list, './' + 'ASL_new_dataset.train')
    save(test_list, './' + 'ASL_new_dataset.test')


def gen_gsl(path):

    dev_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase1/Dataset_corpus/Correct_corpus/PHOENIX-2014-T.dev.corpus.csv'
    train_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase1/Dataset_corpus/Correct_corpus/PHOENIX-2014-T.train.corpus.csv'
    test_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase1/Dataset_corpus/Correct_corpus/PHOENIX-2014-T.test.corpus.csv'

    dev_dict = get_gsl_caption_detail(dev_corpus)
    train_dict = get_gsl_caption_detail(train_corpus)
    test_dict = get_gsl_caption_detail(test_corpus)

    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            dataset_path = (os.path.join(root, name))
            final_dataset_list = []
            numpy_files = sorted(os.listdir(dataset_path))
            for dataSet in numpy_files:
                dict = {}
                numpy_dataset_path = (os.path.join(dataset_path, dataSet))
                numpy_dataset = np.load(numpy_dataset_path)
                torch_tensor = torch.from_numpy(numpy_dataset).float()
                if name == 'test':
                    dict['name'] = dataSet[:-4]
                    dict['gloss'] = test_dict[dataSet[:-4]][1]
                    dict['text'] = test_dict[dataSet[:-4]][2]
                    dict['signer'] = test_dict[dataSet[:-4]][0]
                    dict['sign'] = torch_tensor
                if name == 'dev':
                    dict['name'] = dataSet[:-4]
                    dict['gloss'] = dev_dict[dataSet[:-4]][1]
                    dict['text'] = dev_dict[dataSet[:-4]][2]
                    dict['signer'] = dev_dict[dataSet[:-4]][0]
                    dict['sign'] = torch_tensor
                if name == 'train':
                    dict['name'] = dataSet[:-4]
                    dict['gloss'] = train_dict[dataSet[:-4]][1]
                    dict['text'] =  train_dict[dataSet[:-4]][2]
                    dict['signer'] = train_dict[dataSet[:-4]][0]
                    dict['sign'] = torch_tensor
                final_dataset_list.append(dict)
            if name == 'test':
               save(final_dataset_list, './'+'GSL_dataset.test')
            if name == 'dev':
                save(final_dataset_list, './' + 'GSL_dataset.dev')
            if name == 'train':
                save(final_dataset_list, './'+'GSL_dataset.train')

def gen_gsl_kmeans(path):
    dev_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase1/Dataset_corpus/Correct_corpus/PHOENIX-2014-T.dev.corpus.csv'
    train_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase1/Dataset_corpus/Correct_corpus/PHOENIX-2014-T.train.corpus.csv'
    test_corpus = '/shared/kgcoe-research/mil/sign_language_review/slt_phase1/Dataset_corpus/Correct_corpus/PHOENIX-2014-T.test.corpus.csv'

    dev_dict = get_gsl_caption_detail(dev_corpus)
    train_dict = get_gsl_caption_detail(train_corpus)
    test_dict = get_gsl_caption_detail(test_corpus)

    dic_dev = {}
    with open('/shared/kgcoe-research/mil/sign_language_review/Datasets/features/KMeans/GSL/kmeans_val.csv',
              newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dic_dev[row['video_name']] = literal_eval(row['cluster_ids'])

    test_dev = {}
    with open('/shared/kgcoe-research/mil/sign_language_review/Datasets/features/KMeans/GSL/kmeans_test.csv',
              newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_dev[row['video_name']] = literal_eval(row['cluster_ids'])

    train_dev = {}
    with open('/shared/kgcoe-research/mil/sign_language_review/Datasets/features/KMeans/GSL/kmeans_train.csv',
              newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            train_dev[row['video_name']] = literal_eval(row['cluster_ids'])

    for root, dirs, files in os.walk(path, topdown=False):
        for name in dirs:
            dataset_path = (os.path.join(root, name))
            final_dataset_list = []
            index = 0
            numpy_files = sorted(os.listdir(dataset_path))
            for dataSet in numpy_files:
                dict = {}
                if name == 'test':
                    test_list = np.asarray(test_dev[dataSet[:-4]])
                    torch_tensor = torch.FloatTensor(test_list)
                    dict['name'] = dataSet[:-4]
                    dict['gloss'] = test_dict[dataSet[:-4]][1]
                    dict['text'] = test_dict[dataSet[:-4]][2]
                    dict['signer'] = test_dict[dataSet[:-4]][0]
                    dict['sign'] = torch_tensor
                if name == 'dev':
                    dev_list = np.asarray(dic_dev[dataSet[:-4]])
                    torch_tensor = torch.FloatTensor(dev_list)
                    dict['name'] = dataSet[:-4]
                    dict['gloss'] = dev_dict[dataSet[:-4]][1]
                    dict['text'] = dev_dict[dataSet[:-4]][2]
                    dict['signer'] = dev_dict[dataSet[:-4]][0]
                    dict['sign'] = torch_tensor
                if name == 'train':
                    train_list = np.asarray(train_dev[dataSet[:-4]])
                    torch_tensor = torch.FloatTensor(train_list)
                    dict['name'] = dataSet[:-4]
                    dict['gloss'] = train_dict[dataSet[:-4]][1]
                    dict['text'] = train_dict[dataSet[:-4]][2]
                    dict['signer'] = train_dict[dataSet[:-4]][0]
                    dict['sign'] = torch_tensor
                final_dataset_list.append(dict)
                index = index + 1
            if name == 'test':
                save(final_dataset_list, './' + 'kmeans_dataset.test')
            if name == 'dev':
                save(final_dataset_list, './' + 'kmeans_dataset.dev')
            if name == 'train':
                save(final_dataset_list, './' + 'kmeans_dataset.train')

def get_file_contents(fileName):
    list_of_contents = []
    with open(fileName, "r", encoding='utf-8') as file:
        for line in file:
            stripped_line = line.strip()
            list_of_contents.append(stripped_line)
    return list_of_contents

def save(object, filename, protocol=0):
    file = gzip.GzipFile(filename, 'wb')
    file.write(pickle.dumps(object, protocol))
    file.close()

def get_gsl_caption_detail(raw_csv):
    dataset = pd.read_csv(raw_csv)
    dict ={}
    for desc in dataset['name|video|start|end|speaker|orth|translation']:
        split_desc = desc.split('|')
        dict[split_desc[0]] = [split_desc[4],split_desc[5], split_desc[6]]

    return  dict

def get_csl_caption_detail(raw_csv):
    dataset = pd.read_csv(raw_csv)
    dict ={}
    for index, row in dataset.iterrows():
        dict[row['VideoID']] = [row['Description'], '', 'Signer01']

    return  dict

def get_asl_caption_detail(raw_csv):
    dataset = pd.read_csv(raw_csv)
    dict ={}
    for index, row in dataset.iterrows():
        dict[row['Video_name']] = [row['Caption'], row['Gloss'], 'Signer01']

    return  dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dataset files in joeyNMT format')
    parser.add_argument('Dataset',
                           metavar='dataset',
                           type=str,
                        help='the dataset type CSL, ASL, GSL or GSL_kmeans')
    parser.add_argument('Path',
                        metavar='path',
                        type=str,
                        help='the path to test,train and dev files')
    args = parser.parse_args()

    if args.Dataset == 'ASL':
        gen_asl(args.Path)
    elif args.Dataset == 'CSL':
        gen_csl(args.Path)
    elif args.Dataset == 'GSL':
        gen_gsl(args.Path)
    elif args.Dataset == 'GSL_kmeans':
        gen_gsl_kmeans(args.Path)

