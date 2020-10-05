#!/usr/bin/env python
# coding: utf-8

# # Textual Triple tagger based on Bert-like models.

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


# In[2]:


import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import re
from tqdm import tqdm
from collections import defaultdict
from collections import Counter
import torch.nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup, get_constant_schedule, XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig


DEVICE=0
ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC = range(10)


# In[3]:


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


# In[4]:


# 2. Data Loader
class CoNLLDataset:
    def __init__(self, graphs, tokenizer, max_len, fullvocab=None):
        self.conll_graphs = graphs
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        self._fullvocab = fullvocab if fullvocab else buildVocab(self.conll_graphs, cutoff=1)
            
        self._upos = {p: i for i, p in enumerate(self._fullvocab["upos"])}
        self._iupos = self._fullvocab["upos"]
        self._xpos = {p: i for i, p in enumerate(self._fullvocab["xpos"])}
        self._ixpos = self._fullvocab["xpos"]
        self._vocab = {w: i+3 for i, w in enumerate(self._fullvocab["vocab"])}
        self._wordfreq = self._fullvocab["wordfreq"]
        self._charset = {c: i+3 for i, c in enumerate(self._fullvocab["charset"])}
        self._charfreq = self._fullvocab["charfreq"]
        self._rels = {r: i for i, r in enumerate(self._fullvocab["rels"])}
        self._irels = self._fullvocab["rels"]
        self._feats = {f: i for i, f in enumerate(self._fullvocab["feats"])}
        self._langs = {r: i+2 for i, r in enumerate(self._fullvocab["lang"])}
        self._ilangs = self._fullvocab["lang"]
        
        #self._posRels = {r: i for i, r in enumerate(self._fullvocab["posRel"])}
        #self._iposRels = self._fullvocab["posRel"]
        
        self._vocab['*pad*'] = 0
        self._charset['*pad*'] = 0
        self._langs['*pad*'] = 0
        
        self._vocab['*root*'] = 1
        self._charset['*whitespace*'] = 1
        
        self._vocab['*unknown*'] = 2
        self._charset['*unknown*'] = 2
        
        
    
    def __len__(self):
        return len(self.conll_graphs)
        
        
    def __getitem__(self, item):
        
        graph = self.conll_graphs[item]
        word_list = [node.word for node in graph.nodes]
        upos_list = [node.upos for node in graph.nodes]
        
        encoded = self.tokenizer.encode_plus(' '.join(word_list[1:]),
                                             None,
                                             add_special_tokens=True,
                                             truncation=True,
                                             max_length = self.max_len,
                                             pad_to_max_length = True)
        
        ids, mask = encoded['input_ids'], encoded['attention_mask']
        
        bpe_head_mask = [0]; upos_ids = [-1] # --> CLS token
        
        for word, upos in zip(word_list[1:], upos_list[1:]):
            bpe_len = len(self.tokenizer.tokenize(word))
            head_mask = [1] + [0]*(bpe_len-1)
            bpe_head_mask.extend(head_mask)
            upos_mask = [self._upos.get(upos)] + [-1]*(bpe_len-1)
            upos_ids.extend(upos_mask)
            #print("head_mask", head_mask)
        
        bpe_head_mask.append(0); upos_ids.append(-1) # --> END token
        bpe_head_mask.extend([0] * (self.max_len - len(bpe_head_mask))); upos_ids.extend([-1] * (self.max_len - len(upos_ids))) ## --> padding by max_len

        
        return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'bpe_head_mask': torch.tensor(bpe_head_mask, dtype=torch.long),
                'upos_ids': torch.tensor(upos_ids, dtype=torch.long)
               }
    
    

    


# In[5]:


class XLMRobertaEncoder(nn.Module):
    def __init__(self, num_upos):
        super(XLMRobertaEncoder, self).__init__()
        self.xlm_roberta = transformers.XLMRobertaModel.from_pretrained('xlm-roberta-base')
        self.dropout = nn.Dropout(0.33)
        self.linear = nn.Linear(768, num_upos)
            
    def forward(self, ids, mask):
        o1, o2 = self.xlm_roberta(ids, mask)
        
        #apool = torch.mean(o1, 1)
        #mpool, _ = torch.max(o1, 1)
        #cat = torch.cat((apool, mpool), 1)
        #bo = self.dropout(cat)
        logits = self.linear(o1)       
        outputs = logits
        return outputs
        


# In[6]:




# 2. Set tokenizer for BERT
tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')


# In[7]:


# 1. Read file
TRAIN_BATCH_SIZE = 40
EPOCHS = 200
train_graphs = read_conll('/home/work/code/ktlim/TTtagger/corpus/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu')
train_dataset = CoNLLDataset(train_graphs, tokenizer, 220)
train_loader = DataLoader(dataset=train_dataset, shuffle=True, num_workers=8, batch_size=TRAIN_BATCH_SIZE)

dev_graphs = read_conll('/home/work/code/ktlim/TTtagger/corpus/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-test.conllu')
dev_dataset = CoNLLDataset(dev_graphs, tokenizer, 310, fullvocab=train_dataset._fullvocab)
dev_loader = DataLoader(dataset=dev_dataset, shuffle=False, num_workers=4, batch_size=8)


# In[8]:


num_upos = len(train_dataset._upos)
model = XLMRobertaEncoder(num_upos)
model = nn.DataParallel(model)
model = model.cuda()


# In[9]:


loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
lr = 5e-5
#optimizer = AdamW(model.parameters(), lr=lr)

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE / 1 * EPOCHS)
print(f'num_train_steps = {num_train_steps}')
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=num_train_steps
)



# In[10]:





def train_loop_fn(train_loader, model, optimizer, DEVICE, scheduler=None):
    model.train()
    
    total_pred = []
    total_targ = []
    total_loss = []
    
    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        optimizer.zero_grad()
        #print(batch['ids'], len(batch['ids']), batch['ids'].size() )
        #print(batch['mask'], len(batch['mask']))
        #print(batch['bpe_head_mask'], len(batch['bpe_head_mask']))
        #print(batch['upos_ids'], len(batch['upos_ids']))

        logists = model(batch['ids'].cuda(), batch['mask'].cuda())
        #print(logists, logists.size())
        #print(batch['upos_ids'], batch['upos_ids'].size())
        #print(logists.view(45,9), logists.view(45,9).size())
        #print(batch['upos_ids'].view(45), batch['upos_ids'].view(45).size())
        b,s,l = logists.size()
        loss = loss_fn(logists.view(b*s,l), batch['upos_ids'].cuda().view(b*s))
        total_loss.append(loss.item())
        total_pred.extend(torch.argmax(logists.view(b*s,l), 1).cpu().tolist())
        total_targ.extend(batch['upos_ids'].cuda().view(b*s).cpu().tolist())
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
    #print(total_pred, len(total_pred))
    #print(total_targ, len(total_targ))
    count_active_tokens = np.count_nonzero(np.array(total_targ) > -1)
    count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    print("TRAINING ACCURACY:", count_correct/count_active_tokens)
    #print(count_active_tokens)
    #print(count_correct)


# In[11]:


def dev_loop_fn(dev_loader, model, optimizer, DEVICE, scheduler=None):
    model.eval()
    
    total_pred = []
    total_targ = []
    total_loss = []
    for idx, batch in tqdm(enumerate(dev_loader), total=len(dev_loader)):

        logists = model(batch['ids'].cuda(), batch['mask'].cuda())
        b,s,l = logists.size()
        loss = loss_fn(logists.view(b*s,l), batch['upos_ids'].cuda().view(b*s))
        total_loss.append(loss.item())
        total_pred.extend(torch.argmax(logists.view(b*s,l), 1).cpu().tolist())
        total_targ.extend(batch['upos_ids'].cuda().view(b*s).cpu().tolist())
        
        '''
        print("ids", batch['ids'].size(), batch['ids'])
        print("pred", len(total_pred), total_pred)
        print("targ", len(total_targ), total_targ)
        break
        '''
        
    count_active_tokens = np.count_nonzero(np.array(total_targ) > -1)
    count_correct = np.count_nonzero((np.array(total_pred)==np.array(total_targ)) == True)
    print("TESTING ACC:", count_correct/count_active_tokens)
    
    return round(count_correct/count_active_tokens, 4), total_pred, total_targ


# In[12]:


for idx in range(200):
    train_loop_fn(train_loader, model, optimizer, DEVICE)
    acc, pred, targ = dev_loop_fn(dev_loader, model, optimizer, DEVICE)

    filename=""+str(acc)
    
    pred_np = np.array(pred)
    targ_np = np.array(targ)
    ids_pad = np.where(targ_np != -1)
    pred_np_noPad = pred_np[ids_pad].tolist()
    targ_np_noPad = targ_np[ids_pad].tolist()
    
    with open("./results/predicted_"+filename, "w", encoding="UTF-8") as write_file:            
        pred_list = [ train_dataset._iupos[p] for p in pred_np_noPad]
        write_file.writelines("\n".join(pred_list))

    with open("./results/gold_"+filename, "w", encoding="UTF-8") as write_file:
        gold_list = [ train_dataset._iupos[p] for p in targ_np_noPad]
        write_file.writelines("\n".join(gold_list))
        
    break


# In[ ]:


len(dev_loader.dataset)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
print(input, input.size())
print(target, target.size())

output = loss(input, target)
output.backward()


# In[ ]:


'''
from transformers import RobertaTokenizer, RobertaForTokenClassification

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForTokenClassification.from_pretrained('roberta-base')

input_ids = torch.tensor(tokenizer.encode("Hello, my dog is delexicalized", add_special_tokens=True)).unsqueeze(0)
tokens = tokenizer.tokenize("Hello, my dog is delexicalized", add_special_tokens=True)

labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
outputs = model(input_ids, labels=labels)
loss, scores = outputs[:2]

print(tokenizer.tokenize("Hello, my dog is delexicalized", add_special_tokens=True))
print(tokenizer.encode("Hello, my dog is delexicalized", add_special_tokens=True))
print(tokens)
print(input_ids)
print(labels)

tokenized = tokenizer.tokenize("*root* UPDATE", add_special_tokens=False)
encoded = tokenizer.encode("UPDATE", add_special_tokens=False)

print(tokenized)
print(encoded)

'''

