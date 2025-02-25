# model.py

import torch
import torch.nn as nn
import config
from transformers import BertModel, BertConfig

class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        # Optionally load the HF config for output_hidden_states, etc.
        self.bert_config = BertConfig.from_pretrained(
            config.CONFIG.BERT_MODEL_NAME,
            output_hidden_states=True
        )
        self.bert = BertModel.from_pretrained(
            config.CONFIG.BERT_MODEL_NAME,
            config=self.bert_config
        )

        self.dropout = nn.Dropout(0.1)
        # For extraction approach: We predict start + end
        hidden_size = 768
        self.out = nn.Linear(hidden_size, 2)  # (start_logits, end_logits)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.bert(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        # last_hidden_state shape: (batch_size, seq_len, hidden_size)
        last_hidden_state = outputs.last_hidden_state

        x = self.dropout(last_hidden_state)
        logits = self.out(x)  # shape: (batch_size, seq_len, 2)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)  # (batch_size, seq_len)
        end_logits = end_logits.squeeze(-1)      # (batch_size, seq_len)

        return start_logits, end_logits
