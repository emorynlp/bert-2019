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

from bertsota.parser.dep_parser import SDPParser

if __name__ == '__main__':
    parser = SDPParser()
    save_dir = 'data/model/csdg'
    parser.train(train_file='data/SemEval-2016/train/news.train.debug.conllu',
                 dev_file='data/SemEval-2016/train/news.train.debug.conllu',
                 save_dir=save_dir,
                 pretrained_embeddings_file='data/embedding/glove/glove.6B.100d.debug.txt',
                 train_iters=100,
                 num_buckets_train=1,
                 num_buckets_valid=1,
                 validate_every=10,
                 learning_rate=2e-3,
                 root='Root',
                 debug=True)
    parser.load(save_dir)
    parser.evaluate(test_file='data/SemEval-2016/train/news.train.debug.conllu', save_dir=save_dir,
                    num_buckets_test=1)
