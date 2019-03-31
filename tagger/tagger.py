
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
from collections import defaultdict
from collections import Counter
import re
import ftfy
import random
import pickle
import os, os.path
import copy


# In[2]:


# 1. Vocaburary builder


# In[3]:



ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)

def normalize(word):
    return re.sub(r"\d", "0", word).lower()


def strong_normalize(word):
    w = ftfy.fix_text(word.lower())
    w = re.sub(r".+@.+", "*EMAIL*", w)
    w = re.sub(r"@\w+", "*AT*", w)
    w = re.sub(r"(https?://|www\.).*", "*url*", w)
    w = re.sub(r"([^\d])\1{2,}", r"\1\1", w)
    w = re.sub(r"([^\d][^\d])\1{2,}", r"\1\1", w)
    w = re.sub(r"``", '"', w)
    w = re.sub(r"''", '"', w)
    w = re.sub(r"\d", "0", w)
    return w


def buildVocab(graphs, cutoff=1):
    wordsCount = Counter()
    charsCount = Counter()
    uposCount = Counter()
    xposCount = Counter()
    relCount = Counter()
    featCount = Counter()
    langCount = Counter()

    for graph in graphs:
        wordsCount.update([node.norm for node in graph.nodes[1:]])
        for node in graph.nodes[1:]:
            charsCount.update(list(node.word))
            featCount.update(node.feats_set)
            #  charsCount.update(list(node.norm))
        uposCount.update([node.upos for node in graph.nodes[1:]])
        xposCount.update([node.xupos for node in graph.nodes[1:]])
        relCount.update([rel for rel in graph.rels[1:]])
        langCount.update([node.lang for node in graph.nodes[1:]])
        

    wordsCount = Counter({w: i for w, i in wordsCount.items() if i >= cutoff})
    print("Vocab containing {} words".format(len(wordsCount)))
    print("Charset containing {} chars".format(len(charsCount)))
    print("UPOS containing {} tags".format(len(uposCount)), uposCount)
    #print("XPOS containing {} tags".format(len(xposCount)), xposCount)
    print("Rels containing {} tags".format(len(relCount)), relCount)
    print("Feats containing {} tags".format(len(featCount)))
    print("lang containing {} tags".format(len(langCount)), langCount)

    ret = {
        "vocab": list(wordsCount.keys()),
        "wordfreq": wordsCount,
        "charset": list(charsCount.keys()),
        "charfreq": charsCount,
        "upos": list(uposCount.keys()),
        "xpos": list(xposCount.keys()),
        "rels": list(relCount.keys()),
        "feats": list(featCount.keys()),
        "lang": list(langCount.keys()),
    }

    return ret

def shuffled_stream(data):
    len_data = len(data)
    while True:
        for d in random.sample(data, len_data):
            yield d

def shuffled_balanced_stream(data):
    for ds in zip(*[shuffled_stream(s) for s in data]):
        ds = list(ds)
        random.shuffle(ds)
        for d in ds:
            yield d


# In[4]:


#2. Classes about data structures


# In[5]:



def parse_dict(features):
    if features is None or features == "_":
        return {}

    ret = {}
    lst = features.split("|")
    for l in lst:
        k, v = l.split("=")
        ret[k] = v
    return ret


def parse_features(features):
    if features is None or features == "_":
        return set()

    return features.lower().split("|")


class Word:

    def __init__(self, word, upos, lemma=None, xpos=None, feats=None, misc=None, lang=None):
        self.word = word
        self.norm = normalize(word) #strong_normalize(word)
        self.lemma = lemma if lemma else "_"
        self.upos = upos
        self.xpos = xpos if xpos else "_"
        self.xupos = self.upos + "|" + self.xpos
        self.feats = feats if feats else "_"
        self.feats_set = parse_features(self.feats)
        self.misc = misc if misc else "_"
        self.lang = lang if lang else "_"

    def cleaned(self):
        return Word(self.word, "_")

    def clone(self):
        return Word(self.word, self.upos, self.lemma, self.xpos, self.feats, self.misc)

    def __repr__(self):
        return "{}_{}".format(self.word, self.upos)


class DependencyGraph(object):

    def __init__(self, words, tokens=None):
        #  Token is a tuple (start, end, form)
        if tokens is None:
            tokens = []
        self.nodes = np.array([Word("*root*", "*root*")] + list(words))
        self.tokens = tokens
        self.heads = np.array([-1] * len(self.nodes))
        self.rels = np.array(["_"] * len(self.nodes), dtype=object)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.nodes = self.nodes
        result.tokens = self.tokens
        result.heads = self.heads.copy()
        result.rels = self.rels.copy()
        return result

    def cleaned(self, node_level=True):
        if node_level:
            return DependencyGraph([node.cleaned() for node in self.nodes[1:]], self.tokens)
        else:
            return DependencyGraph([node.clone() for node in self.nodes[1:]], self.tokens)

    def attach(self, head, tail, rel):
        self.heads[tail] = head
        self.rels[tail] = rel

    def __repr__(self):
        return "\n".join(["{} ->({})  {} ({})".format(str(self.nodes[i]), self.rels[i], self.heads[i], self.nodes[self.heads[i]]) for i in range(len(self.nodes))])


# In[6]:


#3. IO and CoNLL file reader


# In[7]:


# take it from https://github.com/chantera/biaffineparser/blob/master/utils.py
def read_conll(filename, lang_code=None):
    
    print("read_conll with", lang_code)
    def get_word(columns):
        return Word(columns[FORM], columns[UPOS], lemma=columns[LEMMA], xpos=columns[XPOS], feats=columns[FEATS], misc=columns[MISC], lang=lang_code)

    def get_graph(graphs, words, tokens, edges):
        graph = DependencyGraph(words, tokens)
        for (h, d, r) in edges:
            graph.attach(h, d, r)
        graphs.append(graph)

    file = open(filename, "r", encoding="UTF-8")

    graphs = []
    words = []
    tokens = []
    edges = []

    sentence_start = False
    while True:
        line = file.readline()
        if not line:
            if len(words) > 0:
                get_graph(graphs, words, tokens, edges)
                words, tokens, edges = [], [], []
            break
        line = line.rstrip("\r\n")

        # Handle sentence start boundaries
        if not sentence_start:
            # Skip comments
            if line.startswith("#"):
                continue
            # Start a new sentence
            sentence_start = True
        if not line:
            sentence_start = False
            if len(words) > 0:
                get_graph(graphs, words, tokens, edges)
                words, tokens, edges = [], [], []
            continue

        # Read next token/word
        columns = line.split("\t")

        # Skip empty nodes
        if "." in columns[ID]:
            continue

        # Handle multi-word tokens to save word(s)
        if "-" in columns[ID]:
            start, end = map(int, columns[ID].split("-"))
            tokens.append((start, end + 1, columns[FORM]))

            for _ in range(start, end + 1):
                word_line = file.readline().rstrip("\r\n")
                word_columns = word_line.split("\t")
                words.append(get_word(word_columns))
                if word_columns[HEAD].isdigit():
                    head = int(word_columns[HEAD])
                else:
                    head = -1
                edges.append((head, int(word_columns[ID]), word_columns[DEPREL].split(":")[0]))
        # Basic tokens/words
        else:
            words.append(get_word(columns))
            if columns[HEAD].isdigit():
                head = int(columns[HEAD])
            else:
                head = -1
            edges.append((head, int(columns[ID]), columns[DEPREL].split(":")[0]))

    file.close()

    return graphs


def write_conll(filename, graphs, append=False):
    if append:
        file = open(filename, "a", encoding="UTF-8")
    else:
        file = open(filename, "w", encoding="UTF-8")

    for j in range(len(graphs)):
        graph = graphs[j]
        curtoken = 0
        for i in range(1, len(graph.nodes)):
            if curtoken < len(graph.tokens) and i == graph.tokens[curtoken][0]:
                file.write("{}-{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_\n".format(graph.tokens[curtoken][0], graph.tokens[curtoken][1] - 1, graph.tokens[curtoken][2]))
                curtoken += 1

            file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t_\t{}\n".format(
                i, graph.nodes[i].word, graph.nodes[i].lemma, graph.nodes[i].upos, graph.nodes[i].xpos,
                graph.nodes[i].feats, graph.heads[i], graph.rels[i], graph.nodes[i].misc))

        file.write("\n")

    file.close()


def read_text(filename):
    file = open(filename, "r")
    text = file.read()
    file.close()

    documents = text.split("\n\n")
    ret = [" ".join(x.split("\n")).strip() for x in documents]

    return ret


# In[8]:


# 4. Build Pytorch Model Class
# import packages
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
    
def _model_var(model, x):
    p = next(model.parameters())
    if p.is_cuda:
        x = x.cuda(p.get_device())
    return torch.autograd.Variable(x)

def pad_sequence(xs, length=None, padding=0, dtype=np.int64):
    lengths = [len(x) for x in xs]
    if length is None:
        length = max(lengths)
    y = np.array([np.pad(x.astype(dtype), (0, length - l),
                         mode="constant", constant_values=padding)
                  for x, l in zip(xs, lengths)])
    return y


# In[9]:


# take it from https://github.com/chantera/teras/blob/master/teras/framework/pytorch/model.py
class MLP(nn.ModuleList):

    def __init__(self, layers):
        assert all(type(layer) == MLP.Layer for layer in layers)
        super(MLP, self).__init__(layers)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    class Layer(nn.Linear):

        def __init__(self, in_features, out_features,
                     activation=None, dropout=0.0, bias=True):
            super(MLP.Layer, self).__init__(in_features, out_features, bias)
            if activation is None:
                self._activate = lambda x: x
            else:
                if not callable(activation):
                    raise ValueError("activation must be callable: type={}"
                                     .format(type(activation)))
                self._activate = activation
            assert dropout == 0 or type(dropout) == float
            self._dropout_ratio = dropout
            if dropout > 0:
                self._dropout = nn.Dropout(p=self._dropout_ratio)
            else:
                self._dropout = lambda x: x

        def forward(self, x):
            size = x.size()
            if len(size) > 2:
                y = super(MLP.Layer, self).forward(
                    x.contiguous().view(-1, size[-1]))
                y = y.view(size[0:-1] + (-1,))
            else:
                y = super(MLP.Layer, self).forward(x)
            return self._dropout(self._activate(y))


# In[ ]:





# In[10]:


class MetaLSTM(nn.Module):

    def __init__(self, input_dim, meta_hidden_size, meta_layer_size,
                 meta_directions, meta_mlp_hidden, num_class, dropout_ratio):
        super(MetaLSTM, self).__init__()
        
        self.meta_hidden_size = 300
        self.meta_layer_size = 1
        self.meta_directions = 2
        self.meta_mlp_hidden = 300
        self.meta_input_dim = input_dim
        
        self.meta_LSTM = nn.LSTM(self.meta_input_dim, self.meta_hidden_size, self.meta_layer_size, dropout=dropout_ratio, batch_first=False, bidirectional=True)        
        layers = [MLP.Layer(self.meta_hidden_size * self.meta_directions, self.meta_mlp_hidden, F.elu, dropout_ratio) for i in range(1)]
        self.meta_MLP = MLP(layers)
        self.meta_linear = nn.Linear(self.meta_mlp_hidden, num_class, bias=True)
        
    def forward(self, input_vector, batch_lengths):
        # input_vec = b,s,d
        
        input_vecs = input_vector
        lengths = batch_lengths
        
        indices = np.argsort(-np.array(lengths)).astype(np.int64) #sorting based on seq_len
        lengths = lengths[indices]
        input_vecs = torch.stack([input_vecs[idx] for idx in indices]) #In order to put batched-LSTM: ordering inputs and stacking input_vecs = 3 x 30 x 9        
        input_vecs = nn.utils.rnn.pack_padded_sequence(input_vecs, lengths, batch_first=True) #padded -> pack to eliminate processes for paddings  input_vecs = lengths*batch x feature_dim = 58 x 9
        encode_out = self.meta_LSTM(input_vecs)[0] #encode_out = batch x lengths*batch x hidden_size*2 =  58x16
        encode_out = nn.utils.rnn.pad_packed_sequence(encode_out, batch_first=True)[0] #packed -> pad #3x30x16
        encode_out = encode_out.index_select(dim=0, index=_model_var(self, torch.from_numpy(np.argsort(indices).astype(np.int64))))

        #MLP-based shape resizing
        mlp_logits = self.meta_MLP(encode_out)
        meta_linear = self.meta_linear(mlp_logits)
        pred = meta_linear.data.max(2)[1].cpu()
        
        return mlp_logits, meta_linear, pred
        


# In[11]:



class Encode_model(nn.Module):
    def __init__(self, 
                 num_word, 
                 dim_word,
                 num_char,
                 dim_char,
                 num_pos, 
                 dim_pos, 
                 num_rel, 
                 dim_rel,
                 num_lang,
                 dim_lang,
                 ext_word_emb,
                 ext_word_size,
                 dim_ext_word,
                 enc_hidden_size, 
                 enc_layer_size, 
                 mlp_hidden_size,
                 char_active,
                 char_global_active,
                 elmo_active,
                 postagger_active,
                 elmo_weight_file,
                 elmo_option_file,
                 cuda_device
                ):
        super(Encode_model, self).__init__()
        
        self.cuda_device = cuda_device
        
        #Step0: init variables
        self.enc_hidden_size = enc_hidden_size
        self.enc_layer_size = enc_layer_size
        self.num_directions = 2 #static
        self.arc_mlp_hidden = mlp_hidden_size
        self.dep_mlp_hidden = 100 #static
        self.dropout_ratio = 0.33 #static
        
        self.ext_word_dim = 0 #static
        
        self.char_active = char_active
        self.char_global_active = char_global_active
        self.char_cnn_active = False# 

        self.char_hidden_size = 300 #static ####KKL
        self.char_layer_size = 3 #static
        self.char_directions = 2 #static
        self.char_mlp_size = 1 #static
        self.char_global_mlp_size = 1 #static
        self.char_feature_dim = 0 #static
        self.char_global_feature_dim = 0
        
        self.num_pos = num_pos
        self.pos_hidden_size = enc_hidden_size
        self.pos_layer_size = enc_layer_size
        self.pos_directions = 2 #static
        self.pos_mlp_hidden = 300
        
        self.dim_lang = dim_lang
        self.num_lang = num_lang
        
        self.postagger_active = postagger_active
        self.elmo = None #static
        self.elmo_active = elmo_active
        self.elmo_weight_file = elmo_weight_file
        self.elmo_option_file = elmo_option_file
        self.elmo_dim = 0 #static
        self.elmo_hidden_size = 0
        

        #Step1: Preparing external word embeddings
        if ext_word_emb is not None:
            num_ext_emb, dim_ext_emb = ext_word_emb.shape
            self.ext_emb = nn.Embedding(*ext_word_emb.shape, padding_idx=0)
            self.ext_emb.cpu() # load embeddings on cpu
            self.ext_emb.weight = nn.Parameter(torch.from_numpy(ext_word_emb))
            #self.ext_emb.weight.data._copy(torch.from_numpy(ext_word_emb))
            self.ext_emb.weight.requires_grad = False
            self.ext_word_dim = dim_ext_emb
            print("The dimension of pre-trained word embedding: ", self.ext_word_dim)
        elif ext_word_size > 0 and dim_ext_word > 0:
            self.ext_emb = nn.Embedding(ext_word_size + 3, dim_ext_word, padding_idx=0)
            self.ext_emb.cpu()
            self.ext_emb.weight.requires_grad = False
            self.ext_word_dim = dim_ext_word
        else:
            print("Init without external embeding")

                
        #Step2: Preparing ELMo embeddings
        if self.elmo_active:
            from allennlp.modules.elmo import Elmo
            self.elmo_dim = 1024
            if torch.cuda.is_available() and self.cuda_device != -1: self.elmo = Elmo(options_file=self.elmo_option_file , weight_file=self.elmo_weight_file, num_output_representations=2).cuda(self.cuda_device)
            else: self.elmo = Elmo(options_file=self.elmo_option_file , weight_file=self.elmo_weight_file, num_output_representations=2)

            # Elmo,
            from allennlp.modules.scalar_mix import ScalarMix
            self.elmo_hidden_size = 1024 
            self.scalar_mix_x_in = ScalarMix(2, False) # scalar vec dim, layer_normalization
            #self.x_elmo_layer_in = nn.Linear(self.elmo_dim, self.elmo_hidden_size)	# dimension reduction

        #Step3: Initializing char embeddings
        if self.char_active: ## With Structured-self attentive embedding 
            self.char_feature_dim = 100   ####KKL
            self.char_emb = nn.Embedding(num_char + 3, dim_char, padding_idx=0)
            self.char_LSTM = nn.LSTM(dim_char, self.char_hidden_size, self.char_layer_size, dropout=self.dropout_ratio, bidirectional=True)
            layers = [MLP.Layer(self.char_hidden_size * self.char_directions, self.char_mlp_size, F.elu, 0) for i in range(1)]
            self.char_mlp = MLP(layers)
            self.att_score = nn.Softmax(dim=2)
            self.char_linear = nn.Linear(self.char_mlp_size * self.char_hidden_size * self.char_directions, self.char_feature_dim, bias=False)
        elif self.char_cnn_active: ## With CNN based embedding
            self.char_feature_dim = 100
            self.embedding4char = nn.Embedding(num_char + 3, dim_char, padding_idx=0)
            self.char_cnn = CNN_Text(
                    input_size=int(self.char_feature_dim/2),
                    output_size=int(self.char_feature_dim/2),
                    filter_sizes=[2,3,4,5],
                    num_filters=30,
                    dropout_rate=self.dropout_ratio,
            )
            
            #Step3-1: Initializing char global embeddings    
        if self.char_global_active:
            self.char_global_feature_dim = 100   ####KKL
            self.char_global_emb = nn.Embedding(num_char + 3, dim_char, padding_idx=0)
            self.char_global_LSTM = nn.LSTM(dim_char, self.char_hidden_size, self.char_layer_size, dropout=self.dropout_ratio, bidirectional=True)
            layers = [MLP.Layer(self.char_hidden_size * self.char_directions, self.char_global_mlp_size, F.elu, 0) for i in range(1)]
            self.char_global_mlp = MLP(layers)
            self.att_global_score = nn.Softmax()
            self.char_global_linear = nn.Linear(self.char_global_mlp_size * self.char_hidden_size * self.char_directions, num_pos, bias=True)
        
        #Step4: Initialzing Encoder
            #4-0. init embeddings
        self.word_emb = nn.Embedding(num_word + 3, dim_word, padding_idx=0)
        self.dropout = nn.Dropout(p=self.dropout_ratio)
        self.pos_emb = nn.Embedding(num_pos, dim_pos, padding_idx=0)
        
        if self.dim_lang > 0:
            self.lang_emb = nn.Embedding(self.num_lang + 2, self.dim_lang, padding_idx=0)

            #4-1. init Pos Encoder and bilinear Classifier
        if self.postagger_active:
            self.pos_input_dim = self.char_hidden_size * self.char_directions + self.ext_word_dim + self.dim_lang 
            self.pos_LSTM = nn.LSTM(self.pos_input_dim, self.pos_hidden_size, self.pos_layer_size, dropout=self.dropout_ratio, batch_first=False, bidirectional=True)        
            layers = [MLP.Layer(self.pos_hidden_size * self.pos_directions, self.pos_mlp_hidden, F.elu, self.dropout_ratio) for i in range(1)]
            self.pos_MLP = MLP(layers)
            self.pos_linear = nn.Linear(self.pos_mlp_hidden, num_pos, bias=True)

            #4-2. init Encoder
        self.input_dim = dim_word + self.ext_word_dim + self.char_feature_dim + self.elmo_hidden_size + self.dim_lang 
        self.enc_LSTM = nn.LSTM(self.input_dim, enc_hidden_size, enc_layer_size, dropout=self.dropout_ratio, batch_first=False, bidirectional=True)        
        
            #4-3. init Decoder with MLPs for arc and dep
        if False: layers_input = self.enc_hidden_size * self.num_directions + self.elmo_hidden_size
        else: layers_input = self.enc_hidden_size * self.num_directions
        layers = [MLP.Layer(layers_input, self.arc_mlp_hidden, F.elu, self.dropout_ratio) for i in range(1)]
        self.arc_MLP_rel = MLP(layers)
        self.pos2_linear = nn.Linear(self.arc_mlp_hidden, num_pos, bias=True)
        
        self.meta_input_dim = self.arc_mlp_hidden + dim_pos + self.dim_lang #+ self.char_hidden_size * self.char_directions
        self.metaLSTM = MetaLSTM(self.meta_input_dim,300,1,2,300,num_pos, self.dropout_ratio)
        

    def forward(self, word_seqs, pos_seqs, ext_word_seqs, char_seqs, token_seqs, lang_seqs, seq_lenths=None, train=True):
        
        # #print(token_seqs)
        #Set batch and lenth for padding
        batch_size = len(word_seqs)
        lengths = seq_lenths
        
        #Getting word embeddings
        word_seqs = self.cuda_variable(torch.from_numpy(word_seqs)) # word_seq = batch x seq_IDs
        words_vecs = self.dropout(self.word_emb(word_seqs)) #words_vecs = batch x len(seq) x hidden_size = 3 x 30 x 5
        
        #Processing ELMo embeddings
        if self.elmo_active:
            import make_elmo_cids
            elmo_cids = make_elmo_cids.batch_to_ids(token_seqs)

            if torch.cuda.is_available() and self.cuda_device != -1:
                elmo_cids = elmo_cids.cuda()

            elmo_emb = self.elmo(elmo_cids)['elmo_representations']
                
            
            x_ELMo_in = self.scalar_mix_x_in(torch.stack([elmo_emb[0], elmo_emb[1]]))
            x_ELMo_in = nn.functional.dropout(x_ELMo_in, p=0.5, training=self.training)

            if not self.char_active and not self.char_cnn_active:
                words_vecs = torch.cat((words_vecs, x_ELMo_in), -1)

        #Processing self attention representation for char-level word embedding
        if self.char_active:
            """
            char_seqs: list(list(sentence))
             - B: batch_size = len(char_seqs)
            """
            batch_size, word_lenth, word_dim = words_vecs.size()
            words_chars_vecs = self.cuda_variable(torch.zeros(batch_size, word_lenth, word_dim + self.char_feature_dim + self.elmo_hidden_size))            
            for sent_idx, sentence in enumerate(char_seqs):
                """
                 - sentence: list(tokens)
                 - B: batch_size = len(sentence) = size of tokens
                 - S: seq_len = len(token) = size of chars
                """
                char_batch_size = len(sentence) #number of tokens == batch_size
                char_len_list = np.array([len(token) for token in sentence])

                h0 = self.cuda_variable(torch.zeros(self.char_layer_size*self.char_directions, char_batch_size, self.char_hidden_size))
                c0 = self.cuda_variable(torch.zeros(self.char_layer_size*self.char_directions, char_batch_size, self.char_hidden_size))
                
                pad_sentence = pad_sequence(sentence)
                char_local_seqs = self.cuda_variable(torch.from_numpy(pad_sentence))
                char_vecs = self.char_emb(char_local_seqs)

                indices = np.argsort(-np.array(char_len_list)).astype(np.int64) #Searching "char_len_list"'s dec order
                char_len_list_ordered = char_len_list[indices] #Sorting "char_len_list" based on the indices
                char_vecs_ordered = torch.stack([char_vecs[order] for order in indices]) #Stacking batches as an embedding based on dec order                
                char_sentence_packed = nn.utils.rnn.pack_padded_sequence(char_vecs_ordered, char_len_list_ordered, batch_first=True)
                char_lstm_batch_out = self.char_LSTM(char_sentence_packed)[0] # (Batch * seq_len x  2*LSTM_hidden)
                char_lstm_batch_out = nn.utils.rnn.pad_packed_sequence(char_lstm_batch_out, batch_first=True)[0] # (Batch x seq_len x 2*LSTM_hidden)
                char_lstm_batch_out = char_lstm_batch_out.index_select(dim=0, index=_model_var(self, torch.from_numpy(np.argsort(indices).astype(np.int64)))) #make it back unorder
                
                #print(char_lstm_batch_out)
                
                char_decode_out = self.char_mlp(char_lstm_batch_out)
                
                #print(char_decode_out)
                char_att_score = self.att_score(char_decode_out.transpose(1, 2))
                char_att_features = char_att_score.bmm(char_lstm_batch_out)
                char_att_features = char_att_features.view(char_att_features.size(0), -1) # B x seq_len * 2 * mlp_hidden
                char_features = self.char_linear(char_att_features)
                
                if self.elmo_active:
                    words_chars_vecs[sent_idx, :char_features.size(0)] = torch.cat((words_vecs[sent_idx, :char_features.size(0)], x_ELMo_in[sent_idx, :char_features.size(0)], char_features), -1)
                else:
                    words_chars_vecs[sent_idx, :char_features.size(0)] = torch.cat((words_vecs[sent_idx, :char_features.size(0)], char_features), -1)
            words_vecs = words_chars_vecs  
            
        
        elif self.char_cnn_active:
            import pdb; pdb.set_trace()
            xc1_emb = nn.functional.dropout(char_vecs, p=0.2, training=self.training)
            xc1_emb2 = self.char_cnn(xc1_emb.view(-1, xc1_emb.size(2), xc1_emb.size(3)))
            words_vecs = torch.cat((words_vecs, xc1_emb2.view(xc1.size(0), xc1.size(1), -1)), -1)
            
            
        pos_global_vecs = None
        if self.char_global_active:
            #char_seqs = [[np.array([self._charset.get(ch, 0) for ch in token.word]) for token in graph.nodes] for graph in graphs]
            char_global_seqs = [np.concatenate([np.append(token , np.array([1])) for token in sentence]) for sentence in char_seqs] 
            
            pad_global_seqs = pad_sequence(char_global_seqs)
            char_global_seqs = self.cuda_variable(torch.from_numpy(pad_global_seqs))
            char_global_vecs = self.char_global_emb(char_global_seqs)
            
            #batched and char global LSTM
            char_global_len_list = np.array([len(sentence) for sentence in char_global_seqs])
            
            global_indices = np.argsort(-np.array(char_global_len_list)).astype(np.int64) #sorting based on seq_len
            char_global_len_list_ordered = char_global_len_list[global_indices] #Sorting "char_len_list" based on the indices
            char_global_vecs_ordered = torch.stack([char_global_vecs[order] for order in global_indices]) #Stacking batches as an embedding based on dec order                
            char_global_sentence_packed = nn.utils.rnn.pack_padded_sequence(char_global_vecs_ordered, char_global_len_list_ordered, batch_first=True)
            char_global_lstm_batch_out = self.char_global_LSTM(char_global_sentence_packed)[0] # (Batch * seq_len x  2*LSTM_hidden)
            char_global_lstm_batch_out = nn.utils.rnn.pad_packed_sequence(char_global_lstm_batch_out, batch_first=True)[0] # (Batch x seq_len x 2*LSTM_hidden)
            char_global_lstm_batch_out = char_global_lstm_batch_out.index_select(dim=0, index=_model_var(self, torch.from_numpy(np.argsort(global_indices).astype(np.int64)))) #make it back unorder
            
            batch_size, word_lenth, __ = words_vecs.size()
            char_global_feature = self.cuda_variable(torch.zeros(batch_size, word_lenth, self.char_hidden_size * self.char_directions))#self.num_pos))
            for batch_idx in range(batch_size):
                ch_start_idx, ch_end_idx = 0,0
                for token_idx, token in enumerate(token_seqs[batch_idx]):
                    token_lenth = len(token)
                    ch_end_idx = ch_start_idx + token_lenth 
                    char_global_token = char_global_lstm_batch_out[batch_idx, ch_start_idx:ch_end_idx, :] #char_global_lstm_batch_out[0, 0:6, :] ==> *root*
                    char_global_token_mlp = self.char_global_mlp(char_global_token)
                    
                    char_global_att_score = self.att_global_score(char_global_token_mlp.transpose(0, 1))
                    char_global_att_vec = char_global_att_score.mm(char_global_token)
                    char_global_feature[batch_idx,token_idx,:] = char_global_att_vec #char_global_att_vec_lin
                    ch_start_idx = ch_end_idx + 1 # +1 == white space KKL
                    
            #char_global_att_feature = self.char_global_linear(char_global_feature)
            pos_global_vecs = char_global_feature
            
        #Processing Additional embeddings (External embedding, corpus embedding, POS)
        if ext_word_seqs is not None:
            ext_word_seqs = self.cuda_variable(torch.from_numpy(ext_word_seqs))
            ext_words_vecs = self.dropout(self.ext_emb(ext_word_seqs))
            words_vecs = torch.cat((words_vecs, ext_words_vecs.float()), -1)
            pos_global_vecs = torch.cat((char_global_feature, ext_words_vecs.float()), -1)
            
        if self.dim_lang > 0:
            lang_vecs = self.lang_emb(self.cuda_variable(torch.from_numpy(lang_seqs))) 
            words_vecs = torch.cat((words_vecs, lang_vecs), -1)
            pos_global_vecs = torch.cat((pos_global_vecs, lang_vecs), -1)
              
        if self.postagger_active:
            pos_indices = np.argsort(-np.array(lengths)).astype(np.int64) #sorting based on seq_len
            pos_lengths = lengths[pos_indices]
            pos_input_vecs = torch.stack([pos_global_vecs[idx] for idx in pos_indices]) #In order to put batched-LSTM: ordering inputs and stacking input_vecs = 3 x 30 x 9        
            pos_input_vecs = nn.utils.rnn.pack_padded_sequence(pos_input_vecs, pos_lengths, batch_first=True) #padded -> pack to eliminate processes for paddings  input_vecs = lengths*batch x feature_dim = 58 x 9
            pos_out = self.pos_LSTM(pos_input_vecs)[0] #encode_out = batch x lengths*batch x hidden_size*2 =  58x16
            pos_out = nn.utils.rnn.pad_packed_sequence(pos_out, batch_first=True)[0] #packed -> pad #3x30x16
            pos_out = pos_out.index_select(dim=0, index=_model_var(self, torch.from_numpy(np.argsort(pos_indices).astype(np.int64))))

            pos_mlp_out = self.pos_MLP(pos_out)
            pos_linear = self.pos_linear(pos_mlp_out)
            pred_poses = pos_linear.data.max(2)[1].cpu()       
            pos_seqs = self.cuda_variable(pred_poses)
        else:
            pos_linear = None
            pos_seqs = self.cuda_variable(torch.from_numpy(pos_seqs)) # pos_seq = batch x seq_IDs

        #pos_seqs = self.cuda_variable(pred_poses)
        pos_vecs = self.dropout(self.pos_emb(pos_seqs)) #pos_vecs = batch x len(seq) x hidden_size = 3 x 30 x 4        

        input_vecs = words_vecs
        indices = np.argsort(-np.array(lengths)).astype(np.int64) #sorting based on seq_len
        lengths = lengths[indices]
        input_vecs = torch.stack([input_vecs[idx] for idx in indices]) #In order to put batched-LSTM: ordering inputs and stacking input_vecs = 3 x 30 x 9        
        input_vecs = nn.utils.rnn.pack_padded_sequence(input_vecs, lengths, batch_first=True) #padded -> pack to eliminate processes for paddings  input_vecs = lengths*batch x feature_dim = 58 x 9
        encode_out = self.enc_LSTM(input_vecs)[0] #encode_out = batch x lengths*batch x hidden_size*2 =  58x16
        encode_out = nn.utils.rnn.pad_packed_sequence(encode_out, batch_first=True)[0] #packed -> pad #3x30x16
        encode_out = encode_out.index_select(dim=0, index=_model_var(self, torch.from_numpy(np.argsort(indices).astype(np.int64))))

        #MLP-based shape resizing
        mlp_arc_rel = self.arc_MLP_rel(encode_out)
        pos2_linear = self.pos2_linear(mlp_arc_rel)
        pred_poses2 = pos2_linear.data.max(2)[1].cpu()

        input_meta_vec = torch.cat((pos_vecs, mlp_arc_rel),-1)
        if self.dim_lang > 0:
            input_meta_vec = torch.cat((input_meta_vec, lang_vecs), -1)

        meta_logits, meta_linear, meta_pred = self.metaLSTM(input_meta_vec, seq_lenths)

        return pos_linear, pos2_linear, meta_linear #, rel_logits

    def cuda_variable(self, tensor):
        # Do cuda() before wrapping with variable
        if torch.cuda.is_available() and self.cuda_device >= 0:
            return Variable(tensor.cuda(self.cuda_device))
        else:
            return Variable(tensor)


# In[12]:


# 5. Build the parser class


# In[13]:


class MLparser:
    def __init__(self):
        pass
    
    def create_parser(self, **kwargs):
        
        self._verbose = kwargs.get("verbose", True)
        if self._verbose:
            print("Parameters (others default):")
            for k in sorted(kwargs):
                print(k, kwargs[k])
        self._args = kwargs
        
        self._seed = kwargs.get("seed", 0)
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)
        self._cuda_device = kwargs.get("cuda_device", -1)
        self._save_model_folder = kwargs.get("save_model_folder", "../result")
        self._save_model_name = kwargs.get("save_model_name", "en")
        self._save_UDfile_folder = kwargs.get("save_UDfile_folder", "../result")
        self._save_UDfile_name = kwargs.get("save_UDfile_name", "en")
        self._model_file = kwargs.get("model_file", None)
        self._train_file = kwargs.get("train_file", None)
        self._train_lang_code = kwargs.get("train_lang_code", None)
        self._dev_file = kwargs.get("dev_file", None)
        self._dev_lang_code = kwargs.get("dev_lang_code", None)
        self._learning_rate = kwargs.get("learning_rate", 0.0015)
        self._dim_word = kwargs.get("dim_word", 100)
        self._dim_char = kwargs.get("dim_char", 100) #####KKL
        self._dim_pos = kwargs.get("dim_pos", 100)
        self._dim_rel = kwargs.get("dim_rel", 30)
        self._dim_lang = kwargs.get("dim_lang", 12)
        
        
        self._enc_hidden_size = kwargs.get("enc_hidden_size", 400)
        self._enc_layer_size = kwargs.get("enc_layer_size", 3)
        self._mlp_hidden_size = kwargs.get("mlp_hidden_size", 300) ##before 500 KKL
        
        #For external-word-embedding
        self._ext_emb_file = kwargs.get("ext_emb_file", None)
        self._ext_word_size = kwargs.get("ext_word_size", 0)
        self._dim_ext_word = kwargs.get("dim_ext_word", 0)
        self._ext_limit = kwargs.get("ext_limit", 1000000)
        self._multi_emb = kwargs.get("multi_emb", False)
        self._ext_emb = None
        
        self._postagger_active = kwargs.get("postagger_active", True)
        self._char_active = kwargs.get("char_active", False)
        self._char_global_active = kwargs.get("char_global_active", True)
        self._elmo_active = kwargs.get("elmo_active", False)
        self._elmo_weight_file = kwargs.get("elmo_weight_file", None)
        self._elmo_option_file = kwargs.get("elmo_option_file", None)

        if self._model_file is None:
            print("########## create a new tagger ##########")
            # Init external embedding       
            self._ext_vocab = None
            self._init_ext_embeddings(self._ext_emb_file)
            # Init Voca
            self.build_vocab(self._train_file, self._train_lang_code) # Init Vocaburaries and set dictionaries
            self._load_dataset(self._train_file, self._train_lang_code, self._dev_file, self._dev_lang_code)
            
        self._init_model()        # Init generate parser_model

        return self
        
    def _init_ext_embeddings(self, ext_emb_file=None):
        # set external word embeddings
        
        if ext_emb_file is not None:
            external_embedding_fp = open(ext_emb_file, 'rb')#, encoding="ISO-8859-1")
            external_embedding_fp.readline() #Caution!! trow out first line of the embedding
            #self.external_embedding = {line.decode('utf8').split(' ')[0]: [float(f) for f in line.decode('utf8').strip().split(' ')[1:]] for line, idx in zip(external_embedding_fp, range(self._ext_limit))}

            error_lines =0
            self.external_embedding = {}
            for l, idx in zip(external_embedding_fp, range(self._ext_limit)):
                try:
                    line = l.decode('utf8')
                    line_split = line.strip().split(' ')
                    self.external_embedding.update({line_split[0]: [float(f) for f in line_split[1:]]})
                except:
                    error_lines += 1
                    continue

            external_embedding_fp.close()
            print("# of embedding reading error: ", error_lines)
            
            self._dim_ext_word  = len(list(self.external_embedding.values())[0])
            self._args['dim_ext_word'] = self._dim_ext_word
            
            self._ext_vocab = {word: i + 3 for i, word in enumerate(self.external_embedding)}
            self._ext_word_size = len(self._ext_vocab)
            self._args['ext_word_size'] = self._ext_word_size
            
            self._ext_emb = np.zeros((len(self.external_embedding) + 3, self._dim_ext_word))
            for word, i in self._ext_vocab.items():
                if len(self.external_embedding[word]) == self._dim_ext_word:
                    self._ext_emb[i] = self.external_embedding[word]
                else:
                    print("word embedding is not matched dimensions, an error [voca]:", word)
            
            self._ext_vocab['*pad*'] = 0
            self._ext_vocab['*root*'] = 1
            print('Loaded external embedding. Vector dimensions ', self._dim_ext_word, "number of external words", self._ext_word_size)
        
        return self

    
    def _init_model(self):

        self.encode_model = Encode_model(len(self._vocab),
                                         self._dim_word,
                                         len(self._charset),
                                         self._dim_char,
                                         len(self._upos),
                                         self._dim_pos,
                                         len(self._rels),
                                         self._dim_rel,
                                         len(self._langs),
                                         self._dim_lang,
                                         self._ext_emb,
                                         self._ext_word_size,
                                         self._dim_ext_word,
                                         self._enc_hidden_size,
                                         self._enc_layer_size,
                                         self._mlp_hidden_size,
                                         self._char_active,
                                         self._char_global_active,
                                         self._elmo_active,
                                         self._postagger_active,
                                         self._elmo_weight_file,
                                         self._elmo_option_file,
                                         self._cuda_device
                                        )
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encode_model.parameters()),
                                     lr=self._learning_rate, betas=(0.9, 0.99), eps=1e-12)
        
        torch.nn.utils.clip_grad_norm(self.encode_model.parameters(), max_norm=5.0)

        class Annealing(object):

            def __init__(self, optimizer, lr):                
                self.step = 0
                self.lr = lr
                self.optimizer = optimizer

            def __call__(self):
                self.step = self.step + 1
                decay, decay_step = 0.75, 5000
                decay_rate = decay ** (self.step / decay_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * decay_rate

        self.annealing = Annealing(optimizer, self._learning_rate)
        
        if torch.cuda.is_available() and self._cuda_device >= 0:
            self.encode_model.cuda(self._cuda_device)
        
        return self

       
    def _load_dataset(self, train_UD=None, train_lang_code=None, dev_UD=None, dev_lang_code=None):

        if train_UD is not None:
            if isinstance(train_UD, str):
                self.train_graphs_list = read_conll(train_UD, train_lang_code)
            elif isinstance(train_UD, list):
                self.train_graphs_list = []
                for file, lang_code in zip(train_UD, train_lang_code):
                    self.train_graphs_list.extend(read_conll(file, lang_code))
        else:
            self.train_graphs_list = None


        if dev_UD is not None:
            self.dev_graphs_list = read_conll(dev_UD, dev_lang_code)
        else:
            self.dev_graphs_list = None

        return self
    
    def _load_vocab(self, vocab):
        self._fullvocab = vocab
        self._upos = {p: i for i, p in enumerate(vocab["upos"])}
        self._iupos = vocab["upos"]
        self._xpos = {p: i for i, p in enumerate(vocab["xpos"])}
        self._ixpos = vocab["xpos"]
        self._vocab = {w: i + 3 for i, w in enumerate(vocab["vocab"])}
        self._wordfreq = vocab["wordfreq"]
        self._charset = {c: i + 3 for i, c in enumerate(vocab["charset"])}
        self._charfreq = vocab["charfreq"]
        self._rels = {r: i for i, r in enumerate(vocab["rels"])}
        self._irels = vocab["rels"]
        self._feats = {f: i + 1 for i, f in enumerate(vocab["feats"])}
        
        self._langs = {r: i+2 for i, r in enumerate(vocab["lang"])}
        self._ilangs = vocab["lang"]
        
        self._vocab['*pad*'] = 0
        self._charset['*pad*'] = 0
        self._langs['*pad*'] = 0

        self._vocab['*root*'] = 1
        self._charset['*root*'] = 1
        
        return self
        

    def load_vocab(self, filename):
        with open(filename, "rb") as f:
            vocab = pickle.load(f)
        self._load_vocab(vocab)
        return self

    def save_vocab(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self._fullvocab, f)
        return self
    

    def build_vocab(self, filename=None, lang_code=None, savefile=None, cutoff=1):
        if filename is None:
            filename = self._train_file
        if isinstance(filename, str):
            graphs = read_conll(filename, lang_code)
        elif isinstance(filename, list):
            graphs = []
            for f,l in zip(filename, lang_code):
                graphs.extend(read_conll(f, l))

        self._fullvocab= buildVocab(graphs, cutoff)

        if savefile:
            self.save_vocab(savefile)
        self._load_vocab(self._fullvocab)
        return self
       
    
    def save(self, filename=None):
        if filename is None:
            filename = self._save_model_folder+"/"+self._save_model_name
        self.save_vocab(filename + ".vocab")
        with open(filename + ".params", "wb") as f:
            pickle.dump((self._args, self._ext_vocab), f)
            
        tmp = filename + '.model'
        torch.save(self.encode_model.state_dict(), tmp)
        #shutil.move(tmp, filename)
        return self
        
    
    def load(self, model_name, **kwargs):
        self.load_vocab(model_name + ".vocab")
        kwargs["model_file"] = model_name+".model"
        with open(model_name + ".params", "rb") as f:
            (args, self._ext_vocab)= pickle.load(f)
            args.update(kwargs)
            self.create_parser(**args)
        
        if kwargs["cuda_device"] == -1:  
            self.encode_model.load_state_dict(torch.load(model_name + ".model", map_location='cpu'))
        else:
            self.encode_model.load_state_dict(torch.load(model_name + ".model"))
        return self
    
    
    def compute_loss(self, y, t):
        
        arc_logits = y
        true_arcs = t
        
        b, l1, l2 = arc_logits.size()
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = self.cuda_variable(torch.from_numpy(true_arcs))
        
        arc_loss = F.cross_entropy(arc_logits.view(b * l1, l2), true_arcs.view(b * l1), ignore_index=-1)
        loss = arc_loss
        
        return loss
    
    def cuda_variable(self, tensor):
        # Do cuda() before wrapping with variable
        if torch.cuda.is_available() and self._cuda_device >= 0:
            return Variable(tensor.cuda(self._cuda_device))
        else:
            return Variable(tensor)
    
    
    def compute_accuracy(self, y, t):
        
        arc_logits = y
        true_arcs = t

        if type(arc_logits) == np.ndarray: 
            arc_logits = torch.from_numpy(arc_logits)
            pred_arcs = arc_logits.max(2)[1].cpu()
        else: pred_arcs = arc_logits.data.max(2)[1].cpu()
        b, l1, l2 = arc_logits.size()
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = torch.from_numpy(true_arcs)
        
        correct = pred_arcs[:,1:].eq(true_arcs[:,1:]).cpu().sum()
        num_tokens = (b * l1 - np.sum(true_arcs.cpu().numpy() == -1))  # -b for excluding ROOT
        
        return correct, num_tokens
    
    def compute_LAS(self, arc_y, arc_t, rel_y, rel_t):
        
        arc_logits = arc_y
        true_arcs = arc_t
        rel_logits = rel_y
        true_rels = rel_t
        
        pred_arcs = arc_logits.data.max(2)[1].cpu()
        b, l1, l2 = arc_logits.size()
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        true_arcs = torch.from_numpy(true_arcs)
        
        arc_correct_idx = pred_arcs[:,1:].eq(true_arcs[:,1:])
        
        pred_rels = rel_logits.data.max(2)[1].cpu()
        b, l1, l2 = rel_logits.size()
        true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        true_rels = torch.from_numpy(true_rels)
        
        rel_correct_idx = pred_rels[:,1:].eq(true_rels[:,1:])
        arc_rel_match = arc_correct_idx + rel_correct_idx #arc_correct_idx.eq(rel_correct_idx)
        
        las_correct = arc_rel_match.eq(2).cpu().sum()
        uas_correct = pred_arcs[:,1:].eq(true_arcs[:,1:]).cpu().sum()
        num_tokens = (b * l1 - np.sum(true_arcs.cpu().numpy() == -1))  # -b for excluding ROOT
        
        return las_correct, uas_correct, num_tokens
    
    def compute_test_LAS(self, arc_y, arc_t, rel_y, rel_t):
        
        pred_arcs = np.array(arc_y)
        true_arcs = arc_t
        pred_rels = np.array(rel_y)
        true_rels = rel_t
        
        pred_arcs = pad_sequence(pred_arcs, padding=-2, dtype=np.int64)
        true_arcs = pad_sequence(true_arcs, padding=-1, dtype=np.int64)
        pred_rels = pad_sequence(pred_rels, padding=-2, dtype=np.int64)
        true_rels = pad_sequence(true_rels, padding=-1, dtype=np.int64)
        
        pred_arcs = torch.from_numpy(pred_arcs)
        true_arcs = torch.from_numpy(true_arcs)
        pred_rels = torch.from_numpy(pred_rels)
        true_rels = torch.from_numpy(true_rels)
        
        b, l1 = pred_arcs.size()
        
        arc_correct_idx = pred_arcs[:,1:].eq(true_arcs[:,1:])
        rel_correct_idx = pred_rels[:,1:].eq(true_rels[:,1:])
        arc_rel_match = arc_correct_idx + rel_correct_idx

        las_correct = arc_rel_match.eq(2).cpu().sum()
        uas_correct = pred_arcs[:,1:].eq(true_arcs[:,1:]).cpu().sum()
        num_tokens = (b * l1 - np.sum(true_arcs.cpu().numpy() == -1))  # -b for excluding ROOT
        
        return las_correct, uas_correct, num_tokens
    
    
    def train(self, lang_code=None, train_file=None, batch_size=32):
        
        self.encode_model.train() # set train mode for dropout on
        total_sentence = 0 # of sentence
        total_loss = 0 # loss for a ephoc
        total_token = 0 # of tokens
        total_uas_correct = 0   
        total_las_correct = 0
        total_pos_correct = 0
        
        graphs_list = self.train_graphs_list if train_file is None else read_conll(train_file, lang_code)
        shuffledTrain = graphs_list
        random.shuffle(shuffledTrain) ###KKL
        num_sentence = len(shuffledTrain)
        num_batch = int(np.ceil(num_sentence/batch_size))
        
        for idx in range(num_batch):  
            graphs = shuffledTrain[idx*batch_size : idx*batch_size+batch_size] #takes a training corpus with batch_size
            
            self.annealing.optimizer.zero_grad()

            seq_lenths = np.array([len(graph.nodes) for graph in graphs])
            word_seqs = [np.array([ self._vocab.get(token.norm, 0) if self._dim_word > 2 else 0 for token in graph.nodes ]) for graph in graphs]
            pos_seqs = [np.array([ self._upos.get(token.upos, 0) for token in graph.nodes ]) for graph in graphs]
            if self._ext_emb_file is not None:
                if self._multi_emb:
                    ext_word_seqs = [np.array([ self._ext_vocab.get(token.lang.split('_')[0]+":"+token.norm, 0) for token in graph.nodes ]) for graph in graphs]
                else:
                    ext_word_seqs = [np.array([ self._ext_vocab.get(token.norm, 0) for token in graph.nodes ]) for graph in graphs]
            else:
                ext_word_seqs = None
                    
            char_seqs = [[np.array([self._charset.get(ch, 0) for ch in token.word]) for token in graph.nodes] for graph in graphs] if (self._char_active or self._char_global_active) else None
            token_seqs = [np.array([ token.word for token in graph.nodes ]) for graph in graphs]
            lang_seqs = [np.array([ self._langs.get(token.lang, 0) for token in graph.nodes ]) for graph in graphs]
            
            #padding
            word_seqs_pad = pad_sequence(word_seqs)
            pos_seqs_pad = pad_sequence(pos_seqs)
            ext_word_seqs_pad = pad_sequence(ext_word_seqs) if self._ext_emb_file is not None else None
            lang_seqs_pad = pad_sequence(lang_seqs)
            
            poses, poses2, meta_poses = self.encode_model(word_seqs_pad, pos_seqs_pad, ext_word_seqs_pad, char_seqs, token_seqs, lang_seqs_pad, seq_lenths, train=True)
            pos_logits = poses
            pos2_logits = poses2
            
            true_poses = np.array([np.array([self._upos.get(token.upos, -1) for token in graph.nodes]) for graph in graphs])    
            loss_poses = self.compute_loss(pos_logits, true_poses) if self._postagger_active else 0
            loss_poses2 = self.compute_loss(pos2_logits, true_poses) if self._postagger_active else 0
            loss_meta_poses = self.compute_loss(meta_poses, true_poses) if self._postagger_active else 0

            loss = loss_poses + loss_poses2 + loss_meta_poses
            total_loss += loss.item()
            
            pos_correct, pos_num_token = self.compute_accuracy(meta_poses, true_poses) if self._postagger_active else (0,0)           
            total_pos_correct += pos_correct
            total_token += pos_num_token
            
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.encode_model.parameters(), 5.0)
            self.annealing.optimizer.step()
            self.annealing.__call__()
            total_sentence += len(graphs)
            
        pos_accuracy = round(float(total_pos_correct)/float(total_token)*100, 2) if self._postagger_active else None
  
        print(total_sentence,"trained", "###### total Loss: ", total_loss ,"POS: ", pos_accuracy)      
        return pos_accuracy

        
    def test(self, lang_code=None, test_file=None, gold_file=None, batch_size=32, write_file=False, epoch=0, file_location=None):
        
        self.encode_model.eval()
        total_sentence = 0 # of sentence
        total_token = 0 # of tokens
        total_uas_correct = 0  
        total_las_correct = 0
        total_pos_correct = 0
        total_pos_char_correct = 0
        total_pos_meta_correct = 0
        
        graphs_list = copy.deepcopy(self.dev_graphs_list) if test_file is None else read_conll(test_file, lang_code=lang_code)
        num_sentence = len(graphs_list)
        num_batch = int(np.ceil(num_sentence/batch_size))
    
        for idx in range(num_batch):
            graphs = graphs_list[idx*batch_size:idx*batch_size+batch_size] #takes a training corpus with batch_size
            
            #print(graphs)
            #converts as a list of vectors and pads based on batch_size
            seq_lenths = np.array([len(graph.nodes) for graph in graphs])
            word_seqs = [np.array([ self._vocab.get(token.norm, 0) if self._dim_word > 2 else 0 for token in graph.nodes ]) for graph in graphs]
            pos_seqs = [np.array([ self._upos.get(token.upos, 0) for token in graph.nodes ]) for graph in graphs]
            if self._ext_emb_file is not None:
                if self._multi_emb:
                    ext_word_seqs = [np.array([ self._ext_vocab.get(token.lang.split('_')[0]+":"+token.norm, 0) for token in graph.nodes ]) for graph in graphs]
                else:
                    ext_word_seqs = [np.array([ self._ext_vocab.get(token.norm, 0) for token in graph.nodes ]) for graph in graphs]
            else:
                ext_word_seqs = None
                
            char_seqs = [[np.array([self._charset.get(ch, 0) for ch in token.word]) for token in graph.nodes] for graph in graphs] if (self._char_active or self._char_global_active) else None
            token_seqs = [np.array([ token.word for token in graph.nodes ]) for graph in graphs]
            lang_seqs = [np.array([ self._langs.get(token.lang, 0) for token in graph.nodes ]) for graph in graphs]

            word_seqs_pad = pad_sequence(word_seqs)
            pos_seqs_pad = pad_sequence(pos_seqs)
            ext_word_seqs_pad = pad_sequence(ext_word_seqs) if self._ext_emb_file is not None else None
            lang_seqs_pad = pad_sequence(lang_seqs)
                
            poses, poses2, meta_poses = self.encode_model(word_seqs_pad, pos_seqs_pad, ext_word_seqs_pad, char_seqs, token_seqs, lang_seqs_pad, seq_lenths, train=True)
            pos_logits = poses
            pos2_logits = poses2
            
            true_poses = np.array([np.array([self._upos.get(token.upos, -1) for token in graph.nodes]) for graph in graphs])    
            pos_correct, pos_num_token = self.compute_accuracy(pos2_logits, true_poses) if self._postagger_active else (0,0)           
            pos_char_correct, pos_num_token = self.compute_accuracy(pos_logits, true_poses) if self._postagger_active else (0,0)           
            pos_meta_correct, pos_num_token = self.compute_accuracy(meta_poses, true_poses) if self._postagger_active else (0,0)           

            total_pos_correct += pos_correct
            total_pos_char_correct += pos_char_correct
            total_pos_meta_correct += pos_meta_correct
            total_token += pos_num_token
            total_sentence += len(graphs)
            
            if write_file:
            
                #set the predicted posses to each sentence
                if self._postagger_active:
                    pred_poses = meta_poses.data.max(2)[1].cpu()
                    batch_idx=0
                    for graph in graphs:              
                        token_idx=0
                        for token in graph.nodes:
                            token.upos = self._iupos[pred_poses[batch_idx,token_idx]]
                            token_idx +=1
                        batch_idx += 1

        pos_accuracy = round(float(total_pos_correct)/float(total_token)*100, 2) if self._postagger_active else None
        pos_char_accuracy = round(float(total_pos_char_correct)/float(total_token)*100, 2) if self._postagger_active else None
        pos_meta_accuracy = round(float(total_pos_meta_correct)/float(total_token)*100, 2) if self._postagger_active else None

        print(total_sentence,"Tested", "POS_meta",pos_meta_accuracy, "POS: ", pos_accuracy, "POS_char:", pos_char_accuracy)      
        
        if write_file:
            
            write_loca = file_location if file_location is not None else self._save_UDfile_folder+self._save_UDfile_name+str(epoch)+"conllu"
            write_conll(write_loca, graphs_list)
            if gold_file is not None:
                os.system('python ./evaluation/conll18_ud_eval.py -v '+gold_file+' '+write_loca+' > '+write_loca+'.txt')
        
        return pos_meta_accuracy
        
    


# In[14]:


################ Finish line of MLparser (class) ###################


# In[15]:


def read_corpus_list(filename="./train_list.csv"):
    line_counter = 0
    data_header = []
    train_list = []

    with open(filename) as train_data:
        while True:
            data = train_data.readline()
            if not data: break
            if line_counter == 0:
                data_header = data.split(",")
            else:
                train_list.append(data.split(","))
            line_counter +=1
    print("Header: \t", data_header)
    
    return train_list

    
def test_model(model, test_file, test_lang, save_folder, save_name, batch_size, elmo_weight=None, elmo_option=None, ensemble=False, cuda_device=-1, file_location=None):
    
    if ensemble:
        print("Ensemble is not implimented yet")
        
    else:
        mtparser = MLparser()
        mtparser.load(model_name=model,
                      save_UDfile_folder=save_folder,
                      save_UDfile_name=save_name,
                      elmo_weight_file = elmo_weight,
                      elmo_option_file = elmo_option,
                      cuda_device=cuda_device)

        accuracy = mtparser.test(lang_code=test_lang,
                          test_file=test_file,
                          batch_size=batch_size,
                          file_location=file_location,
                          write_file=True)
        


# In[16]:


# Generate a Parser


# In[17]:


import time

if __name__== "__main__":
    
    import argparse
    import sys
    
    #sys.argv = ['-f'] + ["train"] + ["--home_dir=/home/parser/"]

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default="train", type=str, help="train? or test?")
    parser.add_argument("--cuda_device", default=-1, type=int, help="-1: without GPU, 0: with GPU,     e.g) CUDA_VISIBLE_DEVICES=4 python tagger.py --cuda_device=0")
    parser.add_argument("--home_dir", default="/home/test/tagger/", type=str, help="homedirectory. i.g.) /home/abc/tagger/")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    
    ##For Training options
    parser.add_argument("--ud_name", default=None, type=str, help="UD language name    e.g)UD_English-EWT")
    parser.add_argument("--min_epochs", default=80, type=int, help="# of epochs to start saving model")
    parser.add_argument("--epochs", default=300, type=int, help="Total number of epochs")
    parser.add_argument("--char_tok_active", default=True, help="Activate token-based character emb.")
    parser.add_argument("--elmo_active", default=False, help="Activate ELMO emb")

    ##For Testing options
    parser.add_argument("--test_lang_code", default=None, type=str, help="Language_code    e.g)en_ewt")
    parser.add_argument("--model_file", default=None, type=str, help="Location of the model. Do not add .model    e.g)./result/UD_English-EWT/English07_95.17")
    parser.add_argument("--test_file", default=None, type=str, help="Location of the test corpus.     e.g)./corpus/en-dev.conllu")
    parser.add_argument("--gold_file", default=None, type=str, help="Location of the gold corpus.     e.g)./corpus/en-test.conllu")
    parser.add_argument("--out_file", default=None, type=str, help="Location of the tagging output.   e.g)./corpus/en.out")
    parser.add_argument("--elmo_weight", default=None, type=str, help="Location of the ELMo weight.   e.g)./ELMO/English/weights.hdf5")
    parser.add_argument("--elmo_option", default=None, type=str, help="Location of the ELMo option.   e.g)./ELMO/English/options.json")
    
    # Load defaults
    args = parser.parse_args()
    home_dir = args.home_dir 
    
    if args.mode=="predict":    

        test_lang = "UD_English-EWT" if args.ud_name is None else args.ud_name
        test_lang_code = "en_ewt" if args.test_lang_code is None else args.test_lang_code
        model_file = home_dir+"/result/UD_English/English1_17.65" if args.model_file is None else args.model_file
        test_file = home_dir+"/corpus/official-submissions/conll18-baseline/en_ewt.conllu" if args.test_file is None else args.test_file
        gold_file = home_dir+"/corpus/official-submissions/00-gold-standard/en_ewt.conllu" if args.gold_file is None else args.gold_file
        elmo_weight = home_dir+"/ELMO/"+test_lang+"/weights.hdf5" if args.elmo_weight is None else args.elmo_weight
        elmo_option = home_dir+"/ELMO/"+test_lang+"/options.json" if args.elmo_option is None else args.elmo_option
        file_location= "./"+test_lang+".eval2" if args.out_file is None else args.out_file
        cuda_device = args.cuda_device


        acc = test_model(model=model_file,
                         test_file=test_file,
                         test_lang=test_lang_code,
                         save_folder="../../result/",
                         save_name="test",
                         file_location=file_location,
                         elmo_option = elmo_option,
                         elmo_weight = elmo_weight,
                         batch_size=args.batch_size,
                         cuda_device=cuda_device,
                         ensemble=False)
        os.system('python ./evaluation/conll18_ud_eval.py -v '+gold_file+' '+"./"+test_lang+".eval2 > "+ test_lang+'.txt')


    elif args.mode=="train":
        
        corpus_list = read_corpus_list("./train_list.csv")
        for corpus in corpus_list:
            if corpus[6] == "no": continue
            else:
                print("A corpus loaded:", corpus)
    
                language = corpus[2] 
                train_lang_code = corpus[3] 
                dev_lang_code = corpus[3]
                test_lang_code = corpus[3]
                
                elmo_active = args.elmo_active
                char_tok_active = False if elmo_active else args.char_tok_active 

                home_dir = home_dir
                save_home = home_dir+"result/"+corpus[0]+"/" 
                corpus_home = home_dir+"corpus/"+"release-2.2-st-train-dev-data/ud-treebanks-v2.2/" 
                emb_home = home_dir+"embeddings/"+language+"/"
                emb_file = train_lang_code.split('_')[0] +".vectors"

                test_file = home_dir+"corpus/official-submissions/Uppsala-18/"+test_lang_code+".conllu" ##KKL
                gold_file = home_dir+"corpus/official-submissions/00-gold-standard/"+test_lang_code+".conllu"

                train_file = corpus[8]
                dev_file = corpus[9]

                #for training and testing
                mlparser = MLparser()
                mlparser.create_parser(#train_file=[corpus_home+corpus[0]+"/"+train_file, corpus_home+"/UD_English-GUM/en_gum-ud-train.conllu", corpus_home+"/UD_English-LinES/en_lines-ud-train.conllu"],
                                       #train_lang_code =[train_lang_code, "en_gum", "en_lines"],
                                       train_file=corpus_home+corpus[0]+"/"+train_file,
                                       train_lang_code = train_lang_code,
                                       dev_file=corpus_home+corpus[0]+"/"+dev_file,
                                       dev_lang_code = dev_lang_code,
                                       save_model_folder=save_home,
                                       save_model_name=language,
                                       save_UDfile_folder=save_home,
                                       save_UDfile_name=language,
                                       ext_emb_file=emb_home+emb_file,
                                       char_active=char_tok_active,
                                       char_global_active=True,
                                       elmo_active=elmo_active,
                                       postagger_active=True,
                                       elmo_weight_file=home_dir+"ELMO/"+corpus[2]+"/weights.hdf5",
                                       elmo_option_file=home_dir+"ELMO/"+corpus[2]+"/options.json",
                                       seed=random.randrange(100),
                                       dim_lang=0,
                                       ext_limit=1500000,
                                       dim_word=100,
                                       cuda_device=args.cuda_device,
                                       learning_rate=0.002,
                                       multi_emb=False
                                      )


                batch_size = args.batch_size
                num_train_epoch = int(corpus[7]) # epoch num
                best_accuracy = 0
                best_epoch = 0
                min_test_epoch = round(num_train_epoch/7) ## minimun training epoch without testing
                force_save_per = 20 ## force to save a model per
                req_save_epoch = 1  ## save model req_save_epoch times before the end of training, it should be bigger than 0!!
                save_epoch = copy.deepcopy(req_save_epoch) ##it should be bigger than 0!!
                stop_run = 300   ## stop run when non_update_count is over
                non_update_count = 0  ## count on the number of non-update

                print("batch_size:", batch_size)

                for epoch in range(num_train_epoch):

                    start = time.time()
                    if non_update_count > stop_run or save_epoch < 1:
                        break

                    pos_acc = mlparser.train(batch_size=batch_size)
                    if pos_acc > 101:  ## if traning accuracy are higher than 99.8 then save next save_epoc times and break
                        save_epoch-=1

                    if epoch > min_test_epoch: 
                        non_update_count+=1
                        pos_acc = mlparser.test(batch_size=batch_size, write_file=False, epoch=epoch)
                        if pos_acc >= best_accuracy or epoch > (num_train_epoch - req_save_epoch) or epoch%force_save_per==0 or save_epoch < req_save_epoch:
                            mlparser.save(filename=mlparser._save_model_folder+mlparser._save_model_name+str(epoch)+"_"+str(pos_acc))
                            _ = mlparser.test(lang_code=test_lang_code, gold_file=gold_file, test_file=test_file, batch_size=batch_size, epoch=epoch, write_file=True)
                            if pos_acc > best_accuracy:
                                best_accuracy = pos_acc
                                best_ephoc = epoch
                                non_update_count = 0

                    print("epoch", epoch, "time",time.time()-start, "best POS accuracy:", best_accuracy, best_epoch)


    else:
        print("python tagger.py (train or predict) --test_lang")



# # 
