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
    parser.train(train_file='data/semeval15/en.psd.conll',
                 dev_file='data/semeval15/en.id.psd.conll',
                 test_file='data/semeval15/en.id.psd.conll',
                 save_dir='data/model/gold-arc-psd-id',
                 pretrained_embeddings_file='data/embedding/glove.6B.100d.txt')
