# functions.py
import os
import torch
from tqdm import tqdm

def calculate_topk_accuracy(y_pred, y, k):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, top_pred = y_pred.topk(k, 1)
        top_pred = top_pred.t()
        correct = top_pred.eq(y.view(1, -1).expand_as(top_pred))
        correct_1 = correct[:1].reshape(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        acc_1 = correct_1 / batch_size
        acc_k = correct_k / batch_size
    return acc_1, acc_k

def train(model, iterator, optimizer, criterion, device, k=2):
    epoch_loss = 0.0
    epoch_acc_1 = 0.0
    epoch_acc_k = 0.0

    model.train()
    for (x, y) in tqdm(iterator, desc="Train", position=1, leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = criterion(logits, y)
        acc_1, acc_k = calculate_topk_accuracy(logits, y, k)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_k += acc_k.item()

    n = len(iterator)
    return epoch_loss / n, epoch_acc_1 / n, epoch_acc_k / n

@torch.no_grad()
def evaluate(model, loader, criterion, device, k: int = 1):
    was_training = model.training
    model.eval()

    total_loss = 0.0
    correct1   = 0
    correctk   = 0
    n_samples  = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        logits = out[0] if isinstance(out, (tuple, list)) else out

        loss = criterion(logits, y)
        bs   = y.size(0)
        total_loss += loss.item() * bs
        n_samples  += bs

        pred1 = logits.argmax(dim=1)
        correct1 += pred1.eq(y).sum().item()

        if k > 1:
            _, predk = logits.topk(k, dim=1, largest=True, sorted=True)
            correctk += predk.eq(y.view(-1, 1)).any(dim=1).sum().item()
        else:
            correctk = correct1

    if was_training:
        model.train()

    epoch_loss = total_loss / n_samples
    acc1 = correct1 / n_samples
    acck = correctk / n_samples
    return epoch_loss, acc1, acck

def get_num_workers():
    total_cores = os.cpu_count() or 1
    optimal_workers = max(1, total_cores // 2)
    min_workers = 4
    num_workers = max(optimal_workers, min_workers)
    print(f"Optimal number of workers: {num_workers} (Total cores: {total_cores})")
    return num_workers
