import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def loss_fn(start_logits, end_logits, start_positions, end_positions):
    """
    Basic cross-entropy over start/end positions.
    """
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    return (start_loss + end_loss) / 2.0


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    total_loss = 0.0

    for batch in tqdm(data_loader, desc="Training", leave=False):
        ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
        start_idx = batch["start_idx"].to(device, dtype=torch.long)
        end_idx = batch["end_idx"].to(device, dtype=torch.long)

        optimizer.zero_grad()
        start_logits, end_logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(start_logits, end_logits, start_idx, end_idx)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validating", leave=False):
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
            start_idx = batch["start_idx"].to(device, dtype=torch.long)
            end_idx = batch["end_idx"].to(device, dtype=torch.long)

            start_logits, end_logits = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(start_logits, end_logits, start_idx, end_idx)
            total_loss += loss.item()

    return total_loss / len(data_loader)
