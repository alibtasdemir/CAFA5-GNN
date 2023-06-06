import torch
from torchmetrics import AUROC, F1Score, ConfusionMatrix
from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report


def get_report(outputs, labels):
    pred = ((outputs > 0.5).float() * 1).cpu().numpy()
    print(classification_report(labels.cpu(), pred))


def confusionMatrix(outputs, labels):
    cm = ConfusionMatrix(task="binary").to(torch.device("cuda"))
    return cm(outputs, labels)


def f1Score(outputs, labels):
    f1 = F1Score(task="binary").to(torch.device("cuda"))
    return f1(outputs, labels)


def auroc(outputs, labels):
    ar = AUROC(task="binary")
    return ar(outputs, labels)


@torch.no_grad()
def evaluate(model, val_loader, device=torch.device('cuda')):
    model.eval()
    outputs = []
    gtruths = []
    losses = []
    for batch in tqdm(val_loader, leave=True, desc="Validation"):
        batch.to(device)
        loss, out, gtruth = model.validation_step(batch)
        outputs.append(out)
        gtruths.append(gtruth)
        losses.append({'Validation_loss': loss})

    outputs = torch.cat(outputs, dim=0)
    gtruths = torch.cat(gtruths, dim=0)

    metrics = {
        "auc": auroc(outputs, gtruths),
        "f1": f1Score(outputs, gtruths)
    }

    get_report(outputs, gtruths)

    return model.validation_epoch_end(metrics, losses)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam, device=torch.device('cuda')):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in tqdm(range(epochs), leave=False):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, leave=True, desc="Training"):
            batch.to(device)
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        result = evaluate(model, val_loader, device=device)
        result['Train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)

    return history


@torch.no_grad()
def predict(model, loader, prot_term_df, prot_map, term_map, max=None, device=torch.device("cuda")):
    model.eval()
    pred_edges = []
    for batch in tqdm(loader, leave=True, desc="Test"):
        batch.to(device)
        out = model(batch).view(-1)
        pred = ((out > 0.5).float() * 1).cpu().numpy()
        found = np.argwhere(pred == 1)
        if found.size > 0:
            edge_tuples = batch['protein', 'function', 'term'].edge_index.t().cpu().numpy()
            select_index = found.reshape(1, found.size)[0]
            edges = edge_tuples[select_index]
            pred_edges += edges.tolist()
            if max:
                if len(pred_edges) >= max:
                    break

    df = pd.DataFrame.from_dict([{'source': a, 'target': b} for a, b in pred_edges])
    df["source"] = df["source"].apply(lambda x: prot_map[prot_map["mappedId"] == x]["proteinId"].values[0])
    df["target"] = df["target"].apply(lambda x: term_map[term_map["mappedId"] == x]["termId"].values[0])
    df["correct"] = df[["source", "target"]].apply(tuple, axis=1).isin(
        prot_term_df[["EntryID", "term"]].apply(tuple, axis=1)
    )

    pos = df[df["correct"] == True]
    neg = df[df["correct"] == False]
    return pos, neg
