from __future__ import print_function
import os
import json
import pickle
import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import h5py
import torch
from torch.utils.data import Dataset
import utils
import pandas as pd
import time

load_csv = {'Count':'Count','Trans':'Transition','FrameQA':'FrameQA','Action':'Action'}
mc_task = ['trans','action']

def get_captions(row, task):
    if task.lower() in mc_task:
        columns = ['question', 'a1', 'a2', 'a3', 'a4', 'a5']
    else:
        columns = ['question']
    sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
    return sents


def get_ques_pairs(row, task):
    if task.lower() in mc_task:
        columns = ['question', 'answer', 'key', 'a1', 'a2', 'a3', 'a4', 'a5']
    else:
        columns = ['question','answer','key']
    sents = [row[1][col] for col in columns if not pd.isnull(row[1][col])]
    return sents



class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None, char2idx=None, idx2char=None, char_len=20):
        if word2idx is None:
            word2idx = {}
            word2idx['UNK'] = 0
        if idx2word is None:
            idx2word = []
            idx2word.append('UNK')
        if char2idx is None:
            char2idx = {}
            # 0 means UNK
            char2idx['UNK'] = 0
        if idx2char is None:
            idx2char = []
            idx2char.append('UNK')

        self.word2idx = word2idx
        self.idx2word = idx2word
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.char_len = char_len

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def c_ntoken(self):
        return len(self.char2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    @property
    def charpadding_idx(self):
        return len(self.char2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        char_tokens = []
        if add_word:
            for w in words: # video captioning
                tokens.append(self.add_word(w)) # ['video', 'captioning']
                for c in list(w):
                    c_t = []
                    c_t.append(self.add_char(c))
                char_tokens.append(c_t)
        else:
            for w in words:
                tokens.append(self.word2idx[w])
                c_t = []
                for c in list(w):
                    c_t.append(self.char2idx[c])
                char_tokens.append(c_t)

        return tokens, char_tokens

    def dump_to_file(self, path):
        pickle.dump([self.word2idx, self.idx2word, self.char2idx, self.idx2char], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word, char2idx, idx2char = pickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word, char2idx, idx2char)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def add_char(self, char):
        if char not in self.char2idx:
            self.idx2char.append(char)
            self.char2idx[char] = len(self.idx2char) - 1
        return self.char2idx[char]

    def __len__(self):
        return len(self.idx2word)

def filter_answers(answers_dset):
    """This will change the answer to preprocessed version
    """
    occurence = []

    for ans_entry in answers_dset:
        if ans_entry not in occurence:
            occurence.append(ans_entry)
    return occurence

def create_ans2label(occurence, name, cache_root='data/cache'):
    """Note that this will also create label2ans.pkl at the same time

    occurence: dict {answer -> whatever}
    name: prefix of the output file
    cache_root: str
    """
    ans2label = {}
    label2ans = []
    label = 0
    for answer in occurence:
        label2ans.append(answer)
        ans2label[answer] = label
        label += 1

    utils.create_dir(cache_root)

    cache_file = os.path.join(cache_root, name+'_ans2label.pkl')
    pickle.dump(ans2label, open(cache_file, 'wb'))
    cache_file = os.path.join(cache_root, name+'_label2ans.pkl')
    pickle.dump(label2ans, open(cache_file, 'wb'))
    return ans2label, label2ans

def load_dictionary(load_path, task):

    if os.path.exists('./data/%s_dict.pkl'%task.lower()):
        dictionary = Dictionary.load_from_file(os.path.join('./data/%s_dict.pkl')%task.lower())
        print('loading dictionary done.')
    else:
        print('Creating %s dictionary...'%task)
        file = os.path.join(load_path,'Total_%s_question.csv'%load_csv[task].lower())
        total_q = pd.read_csv(file, sep='\t')
        all_sents = []
        dictionary = Dictionary()
        for row in total_q.iterrows():
            all_sents.extend(get_captions(row, task))
        for q in all_sents:
            dictionary.tokenize(q, True)

        dictionary.dump_to_file('./data/%s_dict.pkl'%task.lower())
    return dictionary

def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.split(' ')
        word = vals[0]
        vals = list(vals[1:])
        word2emb[word] = np.array(vals)
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = np.array(word2emb[word])

    return np.array(weights), word2emb

def get_q_a_v_pair(file, ans2label, mode,task, cache_root='data/cache'):
    name = mode + '_' + task.lower()
    total_q = pd.DataFrame().from_csv(file, sep='\t')

    target = []
    for i, row in enumerate(total_q.iterrows()):

        q_a_v = get_ques_pairs(row, task)
        target.append({
            'id': i,
            'question': q_a_v[0],
            'answer':q_a_v[1],
            'key':q_a_v[2],
            'label':ans2label[q_a_v[1]]
        })
    # print( target)
    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name + '_target.pkl')
    pickle.dump(target, open(cache_file, 'wb'))
    return target

def get_q_cand_a_v_pair(file,ans2label, mode, task, cache_root='data/cache'):
    name = mode+'_'+task.lower()
    total_q = pd.DataFrame().from_csv(file, sep='\t')
    target = []
    # train_errs = [2072, 8036, 11038, 13620, 16515, 16709, 17364, 17393, 17446, 19637, 21176, 34473, 42732]
    print('Constructing %s target...'%name)
    for i, row in enumerate(total_q.iterrows()):
        # if mode=='Train' and i in train_errs:
        #     continue
        q_a_v = get_ques_pairs(row, task)
        # prin
        target.append({
            'id': i,
            'question': q_a_v[0],
            'answer':q_a_v[1],
            'label': q_a_v[1],
            'key': q_a_v[2],
            'a1':q_a_v[3],
            'a2':q_a_v[4],
            'a3':q_a_v[5],
            'a4':q_a_v[6],
            'a5':q_a_v[7],
        })
    print('target length:',len(target))

    utils.create_dir(cache_root)
    cache_file = os.path.join(cache_root, name + '_target.pkl')
    pickle.dump(target, open(cache_file, 'wb'))
    return target


def _load_dataset(dataroot, mode,task, ans2label, sentence_path='./data/'):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val'
    """
    print('Loading %s  %s dataset ...'%(mode, task))
    answer_path = os.path.join(dataroot, 'cache', '%s_%s_target.pkl' % (mode,task.lower()))  #Train_frameqa
    if os.path.exists(answer_path):
        answers = pickle.load(open(answer_path, 'rb'))
        entries = sorted(answers, key=lambda x: x['id'])
        # entries = entries[:2]
    else:
        file = '%s/%s_%s_question.csv'%(sentence_path,mode,load_csv[task].lower())
        if task.lower() in mc_task:
            entries = get_q_cand_a_v_pair(file, ans2label, mode=mode, task=task)
        else:
            entries = get_q_a_v_pair(file, ans2label, mode=mode, task=task)
    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, params, dictionary, dataroot='data', feat_category='resnet', feat_path='./data/',mode='Train', Mul_Choice=5):
        super(VQAFeatureDataset, self).__init__()
        #assert name in ['Train_frameqa', 'Test_frameqa']
        task = params.task
        self.feat_category = feat_category
        self.feat_path = feat_path
        self.task = task
        self.Mul_Choice = Mul_Choice
        f = os.path.join('./data', 'cache', '%s_ans2label.pkl'%task)
        if not os.path.exists(f):
            print('Constructing ans2label ...')
            file = os.path.join(dataroot,'Total_%s_question.csv'%load_csv[task].lower())
            total_q = pd.DataFrame().from_csv(file, sep='\t')['answer']
            occurence = filter_answers(total_q)
            self.ans2label, self.label2ans = create_ans2label(occurence, task)
        else:
            ans2label_path = os.path.join('./data', 'cache', '%s_ans2label.pkl'%task)
            label2ans_path = os.path.join('./data', 'cache', '%s_label2ans.pkl'%task)
            self.ans2label = pickle.load(open(ans2label_path, 'rb'))
            self.label2ans = pickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary

        self.entries = _load_dataset('./data', mode, task, self.ans2label, dataroot)
        self.max_len = params.max_len
        self.char_max_len = params.char_max_len
        self.num_frame = params.num_frame

        self.tokenize()
        self.tensorize()
        if self.feat_category.lower()=='resnet':
            self.v_dim = 2048
        elif self.feat_category.lower()=='c3d':
            self.v_dim = 4096
        else:
            raise ValueError('The feature you used raise error!!!')

    def _load_video(self, index):
        self.features = h5py.File(os.path.join(self.feat_path, 'TGIF_%s_pool5.hdf5'%self.feat_category.upper()), 'r')

        feature = self.features[str(index)][:].astype('float32')
        feature = utils.pad_video(feature, (self.num_frame, self.v_dim)).astype('float32')
        return torch.from_numpy(feature)

    def tokenize(self):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        # if self.task.lower() not in mc_task:

        if self.task.lower() in mc_task:
            for entry in self.entries:
                tokens = []
                char_tokens = []
                for i, candi in enumerate(['a1','a2','a3','a4','a5']):
                    token, char_token = self.dictionary.tokenize(entry['question'], False)
                    token_candi, char_token_candi = self.dictionary.tokenize(entry[candi], False)
                    token = (token + token_candi)[:self.max_len]
                    char_token = (char_token + char_token_candi)

                    if len(tokens) < self.max_len:
                        # Note here we pad in front of the sentence
                        padding = [0] * (self.max_len - len(token))
                        token = token + padding
                    new_char_token = []
                    for char_tok in char_token:
                        char_tok = char_tok[:self.char_max_len]
                        if len(char_tok) < self.char_max_len:
                            char_padding = [0] * (self.char_max_len - len(char_tok))
                            char_tok = char_tok + char_padding
                            new_char_token.append(char_tok)
                        else:
                            new_char_token.append(char_tok)
                    if len(new_char_token) < self.max_len:
                        sub = self.max_len - len(new_char_token)
                        for _ in range(sub):
                            ins = [0] * self.char_max_len
                            new_char_token.append(ins)
                    else:
                        new_char_token = new_char_token[:self.max_len]
                    utils.assert_eq(len(token), self.max_len)
                    tokens.append(np.array(token))
                    char_tokens.append(np.array(new_char_token))
                entry['q_token'] = np.array(tokens)
                entry['char_token'] = np.array(char_tokens)
                assert entry['q_token'].shape == (self.Mul_Choice, self.max_len)
                assert entry['char_token'].shape == (self.Mul_Choice, self.max_len, self.char_max_len)
        else:
            for entry in self.entries:
                tokens, char_tokens = self.dictionary.tokenize(entry['question'], False)
                tokens = tokens[:self.max_len]
                if len(tokens) < self.max_len:
                    # Note here we pad in front of the sentence
                    padding = [0] * (self.max_len - len(tokens))
                    tokens = tokens + padding
                new_char_tokens = []
                for char_tok in char_tokens:
                    char_tok = char_tok[:self.char_max_len]
                    if len(char_tok) < self.char_max_len:
                        char_padding = [0] * (self.char_max_len -len(char_tok))
                        char_tok = char_tok + char_padding
                        new_char_tokens.append(char_tok)
                    else:
                        new_char_tokens.append(char_tok)
                if len(new_char_tokens) < self.max_len:
                    sub = self.max_len - len(new_char_tokens)
                    for _ in range(sub):
                        ins = [0] * self.char_max_len
                        new_char_tokens.append(ins)
                else:
                     new_char_tokens = new_char_tokens[:self.max_len]
                utils.assert_eq(len(tokens), self.max_len)

                entry['q_token'] = np.array(tokens)
                entry['char_token'] = np.array(new_char_tokens)
                assert entry['q_token'].shape == (self.max_len,)
                assert entry['char_token'].shape == (self.max_len, self.char_max_len)


    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            char_question = torch.from_numpy(np.array(entry['char_token']))
            entry['char_token'] = char_question
            labels= []
            if self.task.lower() =='count':
                labels.append(max(entry['answer'],1))
            else:
                labels.append(entry['label'])
            labels = np.array(labels)
            labels = torch.from_numpy(labels)
            entry['labels'] = labels


    def __getitem__(self, index):
        entry = self.entries[index]
        # idx = index
        features = self._load_video(entry['key'])
        question = entry['q_token']
        char_ques = entry['char_token']
        labels = entry['labels']
        ques_eng = entry['question']
        ans_eng = entry['answer']
        idx = entry['id']
        return features, question, char_ques, labels[0], ques_eng, ans_eng, idx

    def __len__(self):
        return len(self.entries)
