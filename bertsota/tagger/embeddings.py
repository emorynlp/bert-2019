# ========================================================================
# Copyright 2018 ELIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
# -*- coding:utf-8 -*-
# Author：ported from PyTorch implementation of flair: https://github.com/zalandoresearch/flair to MXNet
# Date: 2018-09-21 20:58
import pickle
import re
from abc import abstractmethod
from typing import Union, List
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from mxnet.gluon import nn

from bertsota.common.utils import mxnet_prefer_gpu
from bertsota.tagger.contextual_string_model import ContextualStringModel
from bertsota.tagger.corpus import Sentence, read_pretrained_embeddings, Token


class Embeddings(nn.Block):
    """Abstract base class for all embeddings. Every new type of embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    @abstractmethod
    def embedding_type(self) -> str:
        pass

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        """Add embeddings to all words in a list of sentences. If embeddings are already added, updates only if embeddings
        are non-static."""

        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        everything_embedded = True

        if self.embedding_type == 'word-level':
            for sentence in sentences:
                for token in sentence.tokens:
                    if self.name not in token._embeddings.keys(): everything_embedded = False
        else:
            for sentence in sentences:
                if self.name not in sentence._embeddings.keys(): everything_embedded = False

        if not everything_embedded or not self.static_embeddings:
            self._add_embeddings_internal(sentences)

        return sentences

    @abstractmethod
    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        """Private method for adding embeddings to all words in a list of sentences."""
        pass


class TokenEmbeddings(Embeddings):
    """Abstract base class for all token-level embeddings. Ever new type of word embedding must implement these methods."""

    @property
    @abstractmethod
    def embedding_length(self) -> int:
        """Returns the length of the embedding vector."""
        pass

    @property
    def embedding_type(self) -> str:
        return 'word-level'


class WordEmbeddings(TokenEmbeddings):
    """Standard static word embeddings, such as GloVe or FastText."""

    def forward(self, *args):
        pass

    def __init__(self, embedding_file):
        """Init one of: 'glove', 'extvec', 'ft-crawl', 'ft-german'.
        Constructor downloads required files if not there."""

        self.precomputed_word_embeddings, self.__embedding_length = read_pretrained_embeddings(embedding_file)

        # self.name = embeddings
        self.static_embeddings = True

        super().__init__()

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for i, sentence in enumerate(sentences):

            for token, token_idx in zip(sentence.tokens, range(len(sentence.tokens))):
                token = token

                if token.text in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text]
                elif token.text.lower() in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[token.text.lower()]
                elif re.sub(r'\d', '#', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '#', token.text.lower())]
                elif re.sub(r'\d', '0', token.text.lower()) in self.precomputed_word_embeddings:
                    word_embedding = self.precomputed_word_embeddings[re.sub(r'\d', '0', token.text.lower())]
                else:
                    word_embedding = nd.zeros(self.embedding_length, dtype='float32')

                token.set_embedding(self.name, word_embedding)

        return sentences


class BERTEmbeddings(TokenEmbeddings):
    def __init__(self, embedding_file: List[str]):
        super().__init__()
        self.static_embeddings = True
        self.bert = []
        for f in embedding_file:
            with open(f, 'rb') as f:
                bert = pickle.load(f)
                self.bert.append(bert)
            self.bert_dim = self.bert[0][0].shape[1]

    @property
    def embedding_length(self) -> int:
        return self.bert_dim

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for sent in sentences:
            for i, token in enumerate(sent):
                token.set_embedding(self.name, self.bert[sent.portion][sent.idx][i])
        return sentences


class CharLMEmbeddings(TokenEmbeddings):
    """Contextual string embeddings of words, as proposed in Akbik et al., 2018."""

    def forward(self, *args):
        pass

    def __init__(self, model, detach: bool = True, context: mx.Context = None):
        super().__init__()

        """
            Contextual string embeddings of words, as proposed in Akbik et al., 2018.

            Parameters
            ----------
            arg1 : model
                model string, one of 'news-forward', 'news-backward', 'mix-forward', 'mix-backward', 'german-forward',
                'german-backward' depending on which character language model is desired
            arg2 : detach
                if set to false, the gradient will propagate into the language model. this dramatically slows down
                training and often leads to worse results, so not recommended.
        """
        self.static_embeddings = detach
        self.context = context if context else mxnet_prefer_gpu()
        self.lm = ContextualStringModel.load_language_model(model, context=self.context)
        self.detach = detach
        if detach:
            self.lm.freeze()
            self.static_embeddings = True

        self.is_forward_lm = self.lm.is_forward_lm

        with self.context:
            dummy_sentence = Sentence()
            dummy_sentence.add_token(Token('hello'))
            embedded_dummy = self.embed(dummy_sentence)
            self.__embedding_length = len(embedded_dummy[0].get_token(1).get_embedding())

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        # get text sentences
        text_sentences = [sentence.to_tokenized_string() for sentence in sentences]

        longest_character_sequence_in_batch = len(max(text_sentences, key=len))

        # pad strings with whitespaces to longest sentence
        sentences_padded = []
        append_padded_sentence = sentences_padded.append

        end_marker = ' '
        extra_offset = 1
        for sentence_text in text_sentences:
            pad_by = longest_character_sequence_in_batch - len(sentence_text)
            if self.is_forward_lm:
                padded = '\n{}{}{}'.format(sentence_text, end_marker, pad_by * ' ')
                append_padded_sentence(padded)
            else:
                padded = '\n{}{}{}'.format(sentence_text[::-1], end_marker, pad_by * ' ')
                append_padded_sentence(padded)

        # get hidden states from language model
        all_hidden_states_in_lm = self.lm.get_representation(sentences_padded, self.detach)

        # take first or last hidden states from language model as word representation
        for i, sentence in enumerate(sentences):
            sentence_text = sentence.to_tokenized_string()

            offset_forward = extra_offset
            offset_backward = len(sentence_text) + extra_offset

            for token in sentence.tokens:
                token = token

                offset_forward += len(token.text)

                if self.is_forward_lm:
                    offset = offset_forward
                else:
                    offset = offset_backward

                embedding: nd.NDArray = all_hidden_states_in_lm[offset, i, :]

                # if self.tokenized_lm or token.whitespace_after:
                offset_forward += 1
                offset_backward -= 1

                offset_backward -= len(token.text)

                token.set_embedding(self.name, embedding.asnumpy())

        return sentences


class StackedEmbeddings(TokenEmbeddings):
    """A stack of embeddings, used if you need to combine several different embedding types."""

    def forward(self, *args):
        pass

    def __init__(self, embeddings: List[TokenEmbeddings], detach: bool = True):
        """The constructor takes a list of embeddings to be combined."""
        super().__init__()

        self.embeddings = embeddings

        # IMPORTANT: add embeddings as torch modules
        for i, embedding in enumerate(embeddings):
            # self.add_module('list_embedding_%s' % str(i), embedding)
            setattr(self, 'list_embedding_%s' % str(i), embedding)

        self.detach = detach
        # self.name = 'Stack'
        self.static_embeddings = True

        self.__embedding_type = embeddings[0].embedding_type

        self.__embedding_length = 0
        for embedding in embeddings:
            self.__embedding_length += embedding.embedding_length

    def embed(self, sentences: Union[Sentence, List[Sentence]], static_embeddings: bool = True):
        # if only one sentence is passed, convert to list of sentence
        if type(sentences) is Sentence:
            sentences = [sentences]

        for embedding in self.embeddings:
            embedding.embed(sentences)

    @property
    def embedding_type(self) -> str:
        return self.__embedding_type

    @property
    def embedding_length(self) -> int:
        return self.__embedding_length

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:

        for embedding in self.embeddings:
            embedding._add_embeddings_internal(sentences)

        return sentences


class TwoWayEmbeddings(TokenEmbeddings):
    def __init__(self, first_embedding: TokenEmbeddings, second_embedding: TokenEmbeddings, first_size, second_size):
        super().__init__()
        self.first_embedding = first_embedding
        self.second_embedding = second_embedding
        self.first_mlp = nn.Dense(first_size, in_units=first_embedding.embedding_length,
                                  weight_initializer=mx.init.Xavier())
        self.second_mlp = nn.Dense(second_size, in_units=second_embedding.embedding_length,
                                   weight_initializer=mx.init.Xavier())
        self.length = first_size + second_size
        self.first_size = first_size
        self.second_size = second_size
        self.static_embeddings = False

    @property
    def embedding_length(self) -> int:
        return self.length

    def embed(self, sentences: Union[Sentence, List[Sentence]]) -> List[Sentence]:
        return self._add_embeddings_internal(sentences)

    def _add_embeddings_internal(self, sentences: List[Sentence]) -> List[Sentence]:
        for embedding in self.first_embedding, self.second_embedding:
            everything_embedded = True
            for sentence in sentences:
                if embedding.name not in sentence._embeddings.keys() or not embedding.static_embeddings:
                    everything_embedded = False
            if not everything_embedded:
                embedding.embed(sentences)
                if isinstance(embedding, StackedEmbeddings):
                    for sent in sentences:
                        for token in sent:
                            vector: nd.NDArray = token.embedding
                            token.clear_embeddings()
                            token.set_embedding(embedding.name, vector.asnumpy())
                        sent.set_embedding(embedding.name, 'done')

        # first, sort sentences by number of tokens
        sentences.sort(key=lambda x: len(x), reverse=True)
        longest_token_sequence_in_batch = len(sentences[0])
        fe_batch = nd.zeros((len(sentences), longest_token_sequence_in_batch,
                             self.first_embedding.embedding_length))
        se_batch = nd.zeros((len(sentences), longest_token_sequence_in_batch,
                             self.second_embedding.embedding_length))
        for i, sent in enumerate(sentences):
            for j, token in enumerate(sent):
                fe = token.get_embedding(self.first_embedding.name)
                se = token.get_embedding(self.second_embedding.name)
                fe_batch[i][j] = fe
                se_batch[i][j] = se

        fe_batch = fe_batch.reshape(len(sentences) * longest_token_sequence_in_batch,
                                    self.first_embedding.embedding_length)
        se_batch = se_batch.reshape(len(sentences) * longest_token_sequence_in_batch,
                                    self.second_embedding.embedding_length)

        fe_proj = self.first_mlp(fe_batch)
        se_proj = self.second_mlp(se_batch)

        fe_proj = fe_proj.reshape(len(sentences), longest_token_sequence_in_batch, self.first_size)
        se_proj = se_proj.reshape(len(sentences), longest_token_sequence_in_batch, self.second_size)
        for i, sent in enumerate(sentences):
            for j, token in enumerate(sent):
                token.final_embedding = nd.concat(fe_proj[i][j], se_proj[i][j], dim=0)
        return sentences
