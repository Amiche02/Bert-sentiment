import torch
import config

class TweetDataset:
    """
    Example dataset for tweet-sentiment-extraction style tasks.
    If you only need classification, remove 'targets_start', etc.
    """

    def __init__(self, df):
        self.df = df
        self.tokenizer = config.CONFIG.TOKENIZER
        self.max_len = config.CONFIG.MAX_LEN

        # e.g. your DF has columns: text, selected_text, sentiment
        self.tweets = df["text"].values
        self.selected_texts = df["selected_text"].values
        self.sentiments = df["sentiment"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = str(self.tweets[index])
        selected_text = str(self.selected_texts[index])
        sentiment = str(self.sentiments[index])
        
        encoding = self.tokenizer(
            sentiment,
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_offsets_mapping=True
        )

        # Suppose you want the "start/end" to point to selected_text:
        offsets = encoding["offset_mapping"]
        input_ids = encoding["input_ids"]
        token_type_ids = encoding["token_type_ids"]
        attention_mask = encoding["attention_mask"]

        # Dummy approach (replace with your logic to find exact start/end):
        start_idx = 0
        end_idx = 0

        # Convert everything to torch tensors:
        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "mask": torch.tensor(attention_mask, dtype=torch.long),
            "start_idx": torch.tensor(start_idx, dtype=torch.long),
            "end_idx": torch.tensor(end_idx, dtype=torch.long),
            "tweet": tweet,
            "selected_text": selected_text,
            "sentiment": sentiment,
            "offsets": offsets,  # sometimes needed for post-processing
        }
