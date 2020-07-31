# -*- coding: utf-8 -*-

import torch
from transformers import *


model_name_or_path = "pretrained_models/ToD-BERT-jnt"
# model_name_or_path = "pretrained_models/ToD-BERT-mlm"
model_class, tokenizer_class, config_class = BertModel, BertTokenizer, BertConfig
# using local pretrained models
tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
tod_bert = model_class.from_pretrained(model_name_or_path)
# using online model
# tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
# tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1")

# Encode text
input_text = "[CLS] [SYS] Hello, what can I help with you today? [USR] Find me a cheap restaurant nearby the north town."
input_tokens = tokenizer.tokenize(input_text)
story = torch.Tensor(tokenizer.convert_tokens_to_ids(input_tokens)).long()

if len(story.size()) == 1:
    story = story.unsqueeze(0) # batch size dimension

if torch.cuda.is_available():
    tod_bert = tod_bert.cuda()
    story = story.cuda()

with torch.no_grad():
    input_context = {"input_ids": story, "attention_mask": (story > 0).long()}
    hiddens = tod_bert(**input_context)[0]
    print(hiddens)