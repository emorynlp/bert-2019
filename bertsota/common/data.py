# coding: utf-8

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pickle
from collections import Counter
import numpy as np

from bertsota.common.k_means import KMeans
from bertsota.common.savable import Savable


class ConllWord(object):
    def __init__(self, id, form, lemma=None, cpos=None, pos=None, feats=None, head=None, relation=None, phead=None,
                 pdeprel=None):
        """CoNLL format template, see http://anthology.aclweb.org/W/W06/W06-2920.pdf

        Parameters
        ----------
        id : int
            Token counter, starting at 1 for each new sentence.
        form : str
            Word form or punctuation symbol.
        lemma : str
            Lemma or stem (depending on the particular treebank) of word form, or an underscore if not available.
        cpos : str
            Coarse-grained part-of-speech tag, where the tagset depends on the treebank.
        pos : str
            Fine-grained part-of-speech tag, where the tagset depends on the treebank.
        feats : str
            Unordered set of syntactic and/or morphological features (depending on the particular treebank),
            or an underscore if not available.
        head : int
            Head of the current token, which is either a value of ID,
            or zero (’0’) if the token links to the virtual root node of the sentence.
        relation : str
            Dependency relation to the HEAD.
        phead : int
            Projective head of current token, which is either a value of ID or zero (’0’),
            or an underscore if not available.
        pdeprel : str
            Dependency relation to the PHEAD, or an underscore if not available.
        """
        self.id = id
        self.form = form
        self.cpos = cpos
        self.pos = pos
        self.head = head
        self.relation = relation
        self.lemma = lemma
        self.feats = feats
        self.phead = phead
        self.pdeprel = pdeprel

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos, self.feats, str(self.head), self.relation,
                  self.phead, self.pdeprel]
        return '\t'.join(['_' if v is None else v for v in values])


class ConllSentence(object):
    def __init__(self, words):
        """A list of ConllWord

        Parameters
        ----------
        words : ConllWord
            words of a sentence
        """
        super().__init__()
        self.words = words

    def __str__(self):
        return '\n'.join([word.__str__() for word in self.words])

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return self.words[index]

    def __iter__(self):
        return (line for line in self.words)


class ParserVocabulary(Savable):
    PAD, ROOT, UNK = 0, 1, 2
    """Padding, Root, Unknown"""

    def __init__(self, input_file, pret_file, min_occur_count=2, root='root', shared_vocab=None):
        """Vocabulary, holds word, tag and relation along with their id.
            Load from conll file
            Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications

        Parameters
        ----------
        input_file : str
            conll file
        pret_file : str
            word vector file (plain text)
        min_occur_count : int
            threshold of word frequency, those words with smaller frequency will be replaced by UNK
        """
        super().__init__()
        word_counter = Counter()
        tag_set = set()
        rel_set = set()

        if input_file.endswith('.conllu'):
            with open(input_file) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    cell = line.strip().split()
                    if cell:
                        word, tag = cell[1].lower(), cell[3]
                        word_counter[word] += 1
                        tag_set.add(tag)
                        token = cell[8]
                        if token != '_':
                            token = token.split('|')
                            for edge in token:
                                head, rel = edge.split(':', 1)
                                if rel != root:
                                    rel_set.add(rel)
        else:
            with open(input_file) as f:
                for line in f:
                    info = line.strip().split()
                    if info:
                        if len(info) == 10:
                            arc_offset = 6
                            rel_offset = 7
                        elif len(info) == 8:
                            arc_offset = 5
                            rel_offset = 6
                        # else:
                        #     raise RuntimeError('Illegal line: %s' % line)
                        word, tag, head, rel = info[1].lower(), info[3], int(info[arc_offset]), info[rel_offset]
                        word_counter[word] += 1
                        tag_set.add(tag)
                        if rel != root:
                            rel_set.add(rel)

        self._id2word = ['<pad>', '<root>', '<unk>']
        self._id2tag = ['<pad>', '<root>', '<unk>']
        self._id2rel = ['<pad>', root]
        self.root = root
        reverse = lambda x: dict(list(zip(x, list(range(len(x))))))
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        self._pret_file = pret_file
        self._words_in_train_data = len(self._id2word)
        # print('#words in training set:', self._words_in_train_data)
        if shared_vocab:
            self._id2word = shared_vocab._id2word
            self._id2tag = shared_vocab._id2tag
            self._word2id = shared_vocab._word2id
            self._tag2id = shared_vocab._tag2id
        else:
            if pret_file:
                self._add_pret_words(pret_file)
            self._id2tag += list(sorted(tag_set))
            self._word2id = reverse(self._id2word)
            self._tag2id = reverse(self._id2tag)
        self._id2rel += list(sorted(rel_set))
        self._rel2id = reverse(self._id2rel)
        # print("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def log_info(self, logger):
        """Print statistical information via the provided logger

        Parameters
        ----------
        logger : logging.Logger
            logger created using logging.getLogger()
        """
        logger.info('#words in training set: %d' % self._words_in_train_data)
        logger.info("Vocab info: #words %d, #tags %d #rels %d" % (self.vocab_size, self.tag_size, self.rel_size))

    def _add_pret_words(self, pret_file):
        """Read pre-trained embedding file for extending vocabulary

        Parameters
        ----------
        pret_file : str
            path to pre-trained embedding file
        """
        words_in_train_data = set(self._id2word)
        with open(pret_file) as f:
            for line in f:
                line = line.strip().split()
                if line:
                    word = line[0]
                    if word not in words_in_train_data:
                        self._id2word.append(word)
                        # print 'Total words:', len(self._id2word)

    def has_pret_embs(self):
        """Check whether this vocabulary contains words from pre-trained embeddings

        Returns
        -------
        bool
            Whether this vocabulary contains words from pre-trained embeddings
        """
        return self._pret_file is not None

    def get_pret_embs(self, word_dims=None):
        """Read pre-trained embedding file

        Parameters
        ----------
        word_dims : int or None
            vector size. Use `None` for auto-infer
        Returns
        -------
        numpy.ndarray
            T x C numpy NDArray
        """
        assert (self._pret_file is not None), "No pretrained file provided."
        embs = [[]] * len(self._id2word)
        train = True
        try:
            with open(self._pret_file) as f:
                dim = None
                for line in f:
                    line = line.strip().split()
                    if len(line) > 2:
                        if dim is None:
                            dim = len(line)
                        else:
                            if len(line) != dim:
                                continue
                        word, data = line[0], line[1:]
                        embs[self._word2id[word]] = data
        except FileNotFoundError:
            train = False
        dim -= 1
        if word_dims is None:
            word_dims = dim
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(word_dims)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(pret_embs) if train else pret_embs

    def get_word_embs(self, word_dims):
        """Get randomly initialized embeddings when pre-trained embeddings are used, otherwise zero vectors

        Parameters
        ----------
        word_dims : int
            word vector size
        Returns
        -------
        numpy.ndarray
            T x C numpy NDArray
        """
        if self._pret_file is not None:
            return np.random.randn(self.words_in_train, word_dims).astype(np.float32)
        return np.zeros((self.words_in_train, word_dims), dtype=np.float32)

    def get_tag_embs(self, tag_dims):
        """Randomly initialize embeddings for tag

        Parameters
        ----------
        tag_dims : int
            tag vector size

        Returns
        -------
        numpy.ndarray
            random embeddings
        """
        return np.random.randn(self.tag_size, tag_dims).astype(np.float32)

    def word2id(self, xs):
        """Map word(s) to its id(s)

        Parameters
        ----------
        xs : str or list
            word or a list of words

        Returns
        -------
        int or list
            id or a list of ids
        """
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        """Map id(s) to word(s)

        Parameters
        ----------
        xs : int
            id or a list of ids

        Returns
        -------
        str or list
            word or a list of words
        """
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def rel2id(self, xs):
        """Map relation(s) to id(s)

        Parameters
        ----------
        xs : str or list
            relation

        Returns
        -------
        int or list
            id(s) of relation
        """
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]

    def id2rel(self, xs):
        """Map id(s) to relation(s)

        Parameters
        ----------
        xs : int
            id or a list of ids

        Returns
        -------
        str or list
            relation or a list of relations
        """
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        """Map tag(s) to id(s)

        Parameters
        ----------
        xs : str or list
            tag or tags

        Returns
        -------
        int or list
            id(s) of tag(s)
        """
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    @property
    def words_in_train(self):
        """
        get #words in training set
        Returns
        -------
        int
            #words in training set
        """
        return self._words_in_train_data

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)


class DataLoader(object):
    """
    Load CoNLL data
    Adopted from https://github.com/jcyk/Dynet-Biaffine-dependency-parser with some modifications
    """

    def __init__(self, input_file, n_bkts, vocab, bert=None):
        """Create a data loader for a data set

        Parameters
        ----------
        input_file : str
            path to CoNLL file
        n_bkts : int
            number of buckets
        vocab : ParserVocabulary
            vocabulary object
        """
        self.vocab = vocab
        sents = []
        sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, [0], [ParserVocabulary.ROOT]]]
        with open(input_file) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                info = line.strip().split()
                if info:
                    word, tag = vocab.word2id(info[1].lower()), vocab.tag2id(info[3])
                    token = info[8]
                    hs, rs = [], []
                    if token != '_':
                        token = token.split('|')
                        for edge in token:
                            head, rel = edge.split(':', 1)
                            head = int(head)
                            hs.append(head)
                            # assert rel in vocab._rel2id, 'Relation OOV: %s' % line
                            if rel not in vocab._rel2id:
                                rel = '<pad>'
                            rs.append(vocab.rel2id(rel))
                    sent.append([word, tag, hs, rs])
                else:
                    sents.append(sent)
                    sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, [0], [ParserVocabulary.ROOT]]]

            if len(sent) > 1:  # last sent in file without '\n'
                sents.append(sent)

        self.samples = len(sents)
        len_counter = Counter()
        for sent in sents:
            len_counter[len(sent)] += 1
        self._bucket_lengths = KMeans(n_bkts, len_counter).splits
        self._buckets = [[] for i in range(n_bkts)]
        """bkt_idx x length x sent_idx x 4"""
        len2bkt = {}
        prev_length = -1
        for bkt_idx, length in enumerate(self._bucket_lengths):
            len2bkt.update(list(zip(list(range(prev_length + 1, length + 1)), [bkt_idx] * (length - prev_length))))
            prev_length = length

        self._record = []  # the bucket id of every sent and how many sents are there in that bucket
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            idx = len(self._buckets[bkt_idx])
            self._buckets[bkt_idx].append(sent)
            self._record.append((bkt_idx, idx))

        for bkt_idx, (bucket, length) in enumerate(zip(self._buckets, self._bucket_lengths)):
            self._buckets[bkt_idx] = np.zeros((length, len(bucket), 2 + length * 2), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :2] = np.array([s[:2] for s in sent], dtype=np.int32)
                for wid, word in enumerate(sent):
                    arc, rel = word[-2], word[-1]
                    for a, r in zip(arc, rel):
                        self._buckets[bkt_idx][wid, idx, 2 + a] = 1
                        self._buckets[bkt_idx][wid, idx, 2 + length + a] = r
                    # self._buckets[bkt_idx][wid, idx, 2:length] =

        if bert is not None:
            with open(bert, 'rb') as f:
                self.bert = pickle.load(f)
            self.bert_dim = self.bert[0].shape[1]
        else:
            self.bert = None
            self.bert_dim = 0

    @property
    def idx_sequence(self):
        """Indices of sentences when enumerating data set from batches.
        Useful when retrieving the correct order of sentences

        Returns
        -------
        list
            List of ids ranging from 0 to #sent -1
        """
        return [x[1] for x in sorted(zip(self._record, list(range(len(self._record)))))]

    def get_batches(self, batch_size, shuffle=True):
        """Get batch iterator

        Parameters
        ----------
        batch_size : int
            size of one batch
        shuffle : bool
            whether to shuffle batches. Don't set to True when evaluating on dev or test set.
        Returns
        -------
        tuple
            word_inputs, tag_inputs, arc_targets, rel_targets
        """
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_size = bucket.shape[1]
            n_tokens = bucket_size * self._bucket_lengths[bkt_idx]
            n_splits = min(max(n_tokens // batch_size, 1), bucket_size)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_size), n_splits):
                batches.append((bkt_idx, bkt_batch))

        if shuffle:
            np.random.shuffle(batches)

        sent_idx = 0
        idx_seq = self.idx_sequence
        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # word_id x sent_id
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            arc_targets: np.ndarray = self._buckets[bkt_idx][:, bkt_batch, 2: 2 + word_inputs.shape[0]]
            arc_targets = arc_targets.transpose((2, 0, 1))  # head x dep x batch
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 2 + word_inputs.shape[0]:]
            rel_targets = rel_targets.transpose((2, 0, 1))  # head x dep x batch
            if self.bert:
                seq_len = word_inputs.shape[0]
                bat_len = word_inputs.shape[1]
                batch_bert = np.zeros((seq_len, bat_len, self.bert_dim))
                for i in range(bat_len):
                    bert_sent = self.bert[idx_seq[sent_idx]]
                    batch_bert[1:1 + bert_sent.shape[0], i, :] = bert_sent
                    sent_idx += 1
                yield word_inputs, batch_bert, tag_inputs, arc_targets, rel_targets
            else:
                yield word_inputs, None, tag_inputs, arc_targets, rel_targets


class DepDataLoader(object):
    """
    Load conll data
    """

    def __init__(self, input_file, n_bkts, vocab: ParserVocabulary, bert=None):
        """
        Begin loading
        :param input_file: CoNLL file
        :param n_bkts: number of buckets
        :param vocab: vocabulary object
        """
        self.vocab = vocab
        sents = []
        sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
        with open(input_file) as f:
            for line in f:
                info = line.strip().split()
                if info:
                    arc_offset = 5
                    rel_offset = 6
                    if len(info) == 10:
                        arc_offset = 6
                        rel_offset = 7
                    # else:
                    #     raise RuntimeError('Illegal line: %s' % line)
                    assert info[rel_offset] in vocab._rel2id, 'Relation OOV: %s' % line
                    word, tag, head, rel = vocab.word2id(info[1].lower()), vocab.tag2id(info[3]), int(
                        info[arc_offset]), vocab.rel2id(info[rel_offset])
                    sent.append([word, tag, head, rel])
                else:
                    sents.append(sent)
                    sent = [[ParserVocabulary.ROOT, ParserVocabulary.ROOT, 0, ParserVocabulary.ROOT]]
            if len(sent) > 1:  # last sent in file without '\n'
                sents.append(sent)

        self.samples = len(sents)
        len_counter = Counter()
        for sent in sents:
            len_counter[len(sent)] += 1
        self._bucket_lengths = KMeans(n_bkts, len_counter).splits
        self._buckets = [[] for i in range(n_bkts)]
        """bkt_idx x length x sent_idx x 4"""
        len2bkt = {}
        prev_length = -1
        for bkt_idx, length in enumerate(self._bucket_lengths):
            len2bkt.update(list(zip(list(range(prev_length + 1, length + 1)), [bkt_idx] * (length - prev_length))))
            prev_length = length

        self._record = []
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            idx = len(self._buckets[bkt_idx])
            self._buckets[bkt_idx].append(sent)
            self._record.append((bkt_idx, idx))

        for bkt_idx, (bucket, length) in enumerate(zip(self._buckets, self._bucket_lengths)):
            self._buckets[bkt_idx] = np.zeros((length, len(bucket), 4), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)

        if bert is not None:
            with open(bert, 'rb') as f:
                self.bert = pickle.load(f)
            self.bert_dim = self.bert[0].shape[1]
        else:
            self.bert = None
            self.bert_dim = 0

    @property
    def idx_sequence(self):
        return [x[1] for x in sorted(zip(self._record, list(range(len(self._record)))))]

    def get_batches(self, batch_size, shuffle=True):
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_size = bucket.shape[1]
            n_tokens = bucket_size * self._bucket_lengths[bkt_idx]
            n_splits = min(max(n_tokens // batch_size, 1), bucket_size)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_size), n_splits):
                batches.append((bkt_idx, bkt_batch))

        if shuffle:
            np.random.shuffle(batches)

        sent_idx = 0
        idx_seq = self.idx_sequence
        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]  # word_id x sent_id
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 2]
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 3]
            if self.bert:
                seq_len = word_inputs.shape[0]
                bat_len = word_inputs.shape[1]
                batch_bert = np.zeros((seq_len, bat_len, self.bert_dim))
                for i in range(bat_len):
                    bert_sent = self.bert[idx_seq[sent_idx]]
                    batch_bert[1:1 + bert_sent.shape[0], i, :] = bert_sent
                    sent_idx += 1
                yield word_inputs, batch_bert, tag_inputs, arc_targets, rel_targets
            else:
                yield word_inputs, None, tag_inputs, arc_targets, rel_targets


def split_train_dev(train, output_path):
    with open(train) as src, open(output_path + '.train.conllu', 'w') as train, open(output_path + '.dev.conllu',
                                                                                     'w') as dev:
        sents = src.read().split('\n\n')
        train.write("\n\n".join(s for s in sents if 'sent_id = 220' not in s))
        dev.write("\n\n".join(s for s in sents if 'sent_id = 220' in s))  # section 20 for dev


def slice_train_file(train, ratio):
    with open(train) as src, open(train.replace('.conllu', '.{}.conllu'.format(int(ratio * 100))), 'w') as out:
        sents = src.read().split('\n\n')
        out.write("\n\n".join(s for s in sents[:int(len(sents) * ratio)]))


if __name__ == '__main__':
    # vocab = ParserVocabulary('data/semeval15/en.pas.train.conllu',
    #                          pret_file='data/embedding/glove/glove.6B.100d.debug.txt', root='root')
    # vocab.save('data/model/pas/vocab-local.pkl')
    # print(vocab._id2rel)
    # data_loader = DataLoader(train_file, 2, vocab)
    # for data in 'dm', 'pas', 'psd':
    #     split_train_dev('data/semeval15/en.{}.conllu'.format(data), 'data/semeval15/en.{}'.format(data))
    # for data in 'dm', 'pas', 'psd':
    #     for ratio in range(10, 110, 10):
    #         ratio = ratio / 100
    #         slice_train_file('data/semeval15/en.{}.train.conllu'.format(data), ratio)
    # train_file = 'data/semeval15/en.dm.dev.conllu'
    # vocab = ParserVocabulary(train_file,
    #                          pret_file='data/embedding/glove/glove.6B.100d.debug.txt',
    #                          root='root')
    # # vocab.save('data/model/dm-debug/vocab-local.pkl')
    # data_loader = DataLoader(train_file, 2, vocab, bert='data/semeval15/en.dev.bert')
    # next(data_loader.get_batches(10, shuffle=False))
    with open('data/semeval15/en.dm.train.conllu') as src:
        sents = src.read().split('\n\n')
        print(len(sents))
