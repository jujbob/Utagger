"""
1.
Given a pre-processed input text file, this script dumps all of the internal
layers used to compute ELMo representations to a single (potentially large) file.
The input file is previously tokenized, whitespace separated text, one sentence per line.
The output is a hdf5 file (http://docs.h5py.org/en/latest/) where each
sentence is a size (3, num_tokens, 1024) array with the biLM representations.
In the default setting, each sentence is keyed in the output file by the line number
in the original text file.  Optionally, by specifying --use_sentence_key
the first token in each sentence is assumed to be a unique sentence key
used in the output file.

from https://github.com/allenai/allennlp/blob/master/scripts/write_elmo_representations_to_file.py

2.
Creates ELMo word representations from a vocabulary file. These
word representations are _independent_ - they are the result of running
the CNN and Highway layers of the ELMo model, but not the Bidirectional LSTM.
ELMo requires 2 additional tokens: <S> and </S>. The first token
in this file is assumed to be an unknown token.
This script produces two artifacts: A new vocabulary file
with the <S> and </S> tokens inserted and a glove formatted embedding
file containing word : vector pairs, one per line, with all values
separated by a space.

from https://github.com/allenai/allennlp/blob/master/scripts/create_elmo_embeddings_from_vocab.py
"""

import os
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

import numpy
import msgpack
import torch
from torch.autograd import Variable

import logging

from allennlp.modules.elmo import Elmo
from allennlp.data.vocabulary import DEFAULT_OOV_TOKEN
from allennlp.common.checks import ConfigurationError

from allennlp.data.dataset import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

indexer = ELMoTokenCharactersIndexer()

def batch_to_ids(batch):
    """
    Given a batch (as list of tokenized sentences), return a batch
    of padded character ids.
    """
    instances = []
    for sentence in batch:
        tokens = [Token(token) for token in sentence]
        field = TextField(tokens, {'character_ids': indexer})
        instance = Instance({"elmo": field})
        instances.append(instance)

    dataset = Batch(instances)
    vocab = Vocabulary()
    dataset.index_instances(vocab)
    return dataset.as_tensor_dict()['elmo']['character_ids']

def main():
    # Load the vocabulary words and convert to char ids
    #with open(vocab_path, 'r') as vocab_file:
        #tokens = vocab_file.read().strip().split('\n')

    # Insert the sentence boundary tokens which elmo uses at positions 1 and 2.
    #if tokens[0] != DEFAULT_OOV_TOKEN and not use_custom_oov_token:
        #raise ConfigurationError("ELMo embeddings require the use of a OOV token.")

    #tokens = [tokens[0]] + ["<S>", "</S>"] + tokens[1:]
    device = 0
    batch_size = 64

    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG,
                    datefmt='%m/%d/%Y %I:%M:%S')
    log = logging.getLogger(__name__)

    log.info('start ELMo data preparing...')

    log.info('load Elmo ...')
    if device != -1:
        elmo_token_embedder = Elmo('options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json', 'squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5', 2).cuda(device)
    else:
        elmo_token_embedder = Elmo('options_squad_lm_2x4096_512_2048cnn_2xhighway_skip.json', 'squad_context_concat_lm_2x4096_512_2048cnn_2xhighway_skip.hdf5', 2)

    log.info('done.')
    indexer = ELMoTokenCharactersIndexer()

    log.info('load data.msgpack ...')
    with open('../SQuAD/data.msgpack', 'rb') as f:
        data = msgpack.load(f, encoding='utf8')
    log.info('done.')

    trn_docs = data['trn_context_tokens']
    dev_docs = data['dev_context_tokens']
    trn_ques = data['trn_question_tokens']
    dev_ques = data['dev_question_tokens']

    '''
    log.info('make TRN DOCUMENT elmo weight ...')
    trn_doc_embed_weight = [sentence_to_embedding(elmo_token_embedder, indexer, tokens, batch_size, device) for tokens in trn_docs]
    log.info('done. hidden1: {}, hidden2: {}'.format(len(trn_doc_embed_weight[0]), len(trn_doc_embed_weight[1])))
    log.info('make DEV DOCUMENT elmo weight ...')
    dev_doc_embed_weight = [sentence_to_embedding(elmo_token_embedder, indexer, tokens, batch_size, device) for tokens in dev_docs]
    log.info('done. hidden1: {}, hidden2: {}'.format(len(dev_doc_embed_weight[0]), len(dev_doc_embed_weight[1])))
    log.info('make TRN QUESTION elmo weight ...')
    trn_que_embed_weight = [sentence_to_embedding(elmo_token_embedder, indexer, tokens, batch_size, device) for tokens in trn_ques]
    log.info('done. hidden1: {}, hidden2: {}'.format(len(trn_que_embed_weight[0]), len(trn_que_embed_weight[1])))
    log.info('make DEV QUESTION elmo weight ...')
    '''
    #dev_que_embed_weight = [sentence_to_embedding(elmo_token_embedder, indexer, tokens, batch_size, device) for tokens in dev_ques]
    dev_que_embed_weight = [sentence_to_embedding(elmo_token_embedder, tokens, batch_size, device) for tokens in dev_ques]
    log.info('done. hidden1: {}, hidden2: {}'.format(len(dev_que_embed_weight[0]), len(dev_que_embed_weight[1])))

    result = {
        'trn_doc_elmo1': trn_doc_embed_weight[0],
        'trn_doc_elmo2': trn_doc_embed_weight[1],
        'dev_doc_elmo1': dev_doc_embed_weight[0],
        'dev_doc_elmo2': dev_doc_embed_weight[1],
        'trn_que_elmo1': trn_que_embed_weight[0],
        'trn_que_elmo2': trn_que_embed_weight[1],
        'dev_que_elmo1': dev_que_embed_weight[0],
        'dev_que_elmo2': dev_que_embed_weight[1]
    }

    log.info('save disk ...')
    with open('../SQuAD/elmo_embeddings.msgpack', 'wb') as f:
        msgpack.dump(result, f)
    log.info('done.')

#def sentence_to_embedding(elmo_token_embedder, indexer, tokens, batch_size, device):
def sentence_to_embedding(elmo_token_embedder, tokens, batch_size, device=-1):
    indices = [indexer.token_to_indices(Token(token), Vocabulary()) for token in tokens]
    sentences = []
    for k in range((len(indices) // 50) + 1):
        sentences.append(indexer.pad_token_sequence(indices[(k * 50):((k + 1) * 50)],
                                                    desired_num_tokens=50,
                                                    padding_lengths={}))

    last_batch_remainder = 50 - (len(indices) % 50)

    all_embeddings1 = []; all_embeddings2 = []
    for i in range((len(sentences) // batch_size) + 1):
        array = numpy.array(sentences[i * batch_size: (i + 1) * batch_size])
        try:
            if device != -1:
                batch = Variable(torch.from_numpy(array).cuda(device))
            else:
                batch = Variable(torch.from_numpy(array))
        except: import pdb; pdb.set_trace()

        #token_embedding = elmo_token_embedder(batch)['token_embedding'].data
        token_embedding1, token_embedding2 = elmo_token_embedder(batch)['elmo_representations']

        # Reshape back to a list of words of shape (batch_size * 50, encoding_dim)
        # We also need to remove the <S>, </S> tokens appended by the encoder.
        #per_word_embeddings = token_embedding[:, 1:-1, :].contiguous().view(-1, token_embedding.size(-1))
        per_word_embeddings1 = token_embedding1.contiguous().view(-1, token_embedding1.size(-1))
        per_word_embeddings2 = token_embedding2.contiguous().view(-1, token_embedding2.size(-1))

        all_embeddings1.append(per_word_embeddings1)
        all_embeddings2.append(per_word_embeddings2)

    # Remove the embeddings associated with padding in the last batch.
    all_embeddings1[-1] = all_embeddings1[-1][:-last_batch_remainder, :]
    all_embeddings2[-1] = all_embeddings2[-1][:-last_batch_remainder, :]

    save=True
    if save:
        embedding_weight1 = torch.cat(all_embeddings1, 0).cpu().data.numpy()
        embedding_weight2 = torch.cat(all_embeddings2, 0).cpu().data.numpy()
    else:
        embedding_weight1 = torch.cat(all_embeddings1, 0)
        embedding_weight2 = torch.cat(all_embeddings2, 0)

    return [embedding_weight1, embedding_weight2]


if __name__ == "__main__":
    main()
