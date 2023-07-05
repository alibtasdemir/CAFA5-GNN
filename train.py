import os
import warnings

import engine
from model import Model
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore')
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

from preprocess import readProtTerm, read_embeddings, read_embeddings_new
from helpers import get_mappings, getTrainLoader, getValidationLoader
plt.style.use('ggplot')

TRAIN_DIR = "data/Train"
GO_DIR = "data/GO_embeds"

go_path = os.path.join(TRAIN_DIR, "go-basic.obo")
GO_edgelist_path = "GO.edgelist"

GO_features_path = "data/go_embeddings.csv"
train_terms_path = os.path.join(TRAIN_DIR, "train_terms_reduced.tsv")


def create_dataset(prot_ids, annot_ids, prot_term_edge_index, prot_feature=None, annot_feature=None):
    data = HeteroData()
    data["protein"].node_id = torch.arange(len(prot_ids))
    data["term"].node_id = torch.arange(len(annot_ids))
    if prot_feature is not None:
        data["protein"].x = prot_feature
    if annot_feature is not None:
        data["term"].x = annot_feature
    data["protein", "function", "term"].edge_index = prot_term_edge_index

    return T.ToUndirected()(data)

"""
train_terms_df = readProtTerm(train_terms_path)
train_prot_ids, protein_features = read_embeddings(TRAIN_DIR)
term_ids = train_terms_df.term.unique()
unique_protein_id, unique_annot_id, edge_index_protein_to_term = get_mappings(
    train_terms_df,
    train_prot_ids,
    term_ids
)
"""
train_terms_df = readProtTerm(train_terms_path)
train_prot_ids, protein_features, term_ids, term_features = read_embeddings_new(TRAIN_DIR)
train_terms_df_new = train_terms_df.query('term in @term_ids and EntryID in @train_prot_ids')

unique_protein_id, unique_annot_id, edge_index_protein_to_term = get_mappings(
    train_terms_df_new,
    train_prot_ids,
    term_ids
)

data = create_dataset(
    unique_protein_id,
    unique_annot_id,
    edge_index_protein_to_term,
    prot_feature=protein_features,
    annot_feature=term_features
)


transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=2.0,
    add_negative_train_samples=False,
    edge_types=("protein", "function", "term"),
    rev_edge_types=("term", "rev_function", "protein"),
)
train_data, val_data, test_data = transform(data)

EPOCHS = 20
BATCH_SIZE = 1024
LR = 0.001
OPT_FUNC = torch.optim.Adam
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = getTrainLoader(train_data, batch_size=BATCH_SIZE)
val_loader = getValidationLoader(val_data, batch_size=BATCH_SIZE)

model = Model(
    hidden_channels=128,
    protein_input=data["protein"].num_nodes,
    term_input=data["term"].num_nodes,
    metadata=data.metadata()
)

model.to(device)

history = engine.fit(EPOCHS, LR, model, train_loader, val_loader, opt_func=OPT_FUNC, device=device)
print(history)
torch.save(model.state_dict(), "full-heteroGNN.torch")

pos, neg = engine.predict(model, val_loader, train_terms_df, unique_protein_id, unique_annot_id, device=device)
print(pos.shape)
print(pos.head())
print(neg.shape)
print(neg.head())

def plot_accuracies(history):
    """ Plot the history of accuracies"""
    auc = [x['Validation_AUC'] for x in history]
    f1 = [x['Validation_F1'] for x in history]
    plt.plot(auc, '-bx')
    plt.plot(f1, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('F1 and AUROC scores')
    plt.legend(['AUC', 'F1'])
    plt.xticks(np.arange(1, EPOCHS+1, step=1))
    plt.title('Scores vs. No. of epochs')
    plt.savefig("acc.png")
    plt.show()


def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('Train_loss') for x in history]
    val_losses = [x['Validation_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.xticks(np.arange(1, EPOCHS + 1, step=1))
    plt.savefig("losses.png")
    plt.show()


plot_losses(history)
plot_accuracies(history)
"""

print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1, 6):
    total_loss = total_examples = 0
    for sampled_data in tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["protein", "function", "term"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
"""