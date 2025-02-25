import torch
import config
import numpy as np
from model import TweetModel
from dataset import TweetDataset

def load_model(model_path=None):
    device = torch.device(config.CONFIG.DEVICE)
    model = TweetModel()
    if model_path is None:
        model_path = config.CONFIG.MODEL_PATH
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict(model, tweet, sentiment="positive"):
    """
    Example: predict the extraction for a single tweet.
    For classification, you'd do something simpler.
    """
    tokenizer = config.CONFIG.TOKENIZER
    encoded = tokenizer(
        sentiment,
        tweet,
        add_special_tokens=True,
        max_length=config.CONFIG.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True
    )

    ids = encoded["input_ids"]
    mask = encoded["attention_mask"]
    token_type_ids = encoded["token_type_ids"]
    offsets = encoded["offset_mapping"].numpy()[0]

    device = torch.device(config.CONFIG.DEVICE)
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)

    with torch.no_grad():
        start_logits, end_logits = model(ids, mask, token_type_ids)
        start_idx = torch.argmax(start_logits, dim=1).cpu().numpy()[0]
        end_idx = torch.argmax(end_logits, dim=1).cpu().numpy()[0]

    # Post-process if end < start, etc.
    if end_idx < start_idx:
        end_idx = start_idx

    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        start_offset, end_offset = offsets[ix]
        selected_text += tweet[start_offset:end_offset]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
            selected_text += " "

    return selected_text.strip()

if __name__ == "__main__":
    model = load_model()
    example_tweet = "I love the new design of your website!"
    example_sentiment = "positive"
    output = predict(model, example_tweet, example_sentiment)
    print("Predicted extracted text:", output)
