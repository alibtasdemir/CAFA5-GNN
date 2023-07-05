from os import path

import networkx as nx
import numpy as np
import obonet
import pandas as pd
import torch


def create_GO_edgelist(gopath, outpath):
    G = obonet.read_obo(gopath)
    nx.write_edgelist(G, outpath, data=False)


def create_GO_df(edgelist):
    return pd.read_csv(edgelist, sep=" ", header=None, names=["source", "target"])


def readProtTerm(pairPath):
    return pd.read_csv(pairPath, sep="\t")


def read_embeddings(t5path):
    train_prot_ids = np.load(path.join(t5path, "train_ids_reduced.npy"), allow_pickle=True)
    train_embeddings = np.load(path.join(t5path, "train_embeds_reduced.npy"), allow_pickle=True)

    num_cols = train_embeddings.shape[1]
    train_df = pd.DataFrame(train_embeddings, columns=["Feature_" + str(i) for i in range(1, num_cols + 1)])
    train_df["EntryID"] = train_prot_ids
    train_df.set_index("EntryID", inplace=True)

    return train_prot_ids, torch.from_numpy(train_df.values).to(torch.float)


def read_embeddings_new(main_dir):
    protein_embs_df = pd.read_csv(path.join(main_dir, "train_protein_embs_reduced.csv"), index_col=0)
    term_embs_df = pd.read_csv(path.join(main_dir, "go_embeddings_reduced.csv"), index_col=0)

    protein_ids, protein_embs = protein_embs_df.index.values, torch.from_numpy(protein_embs_df.values).to(torch.float)
    term_ids, term_embs = term_embs_df.index.values, torch.from_numpy(term_embs_df.values).to(torch.float)
    return protein_ids, protein_embs, term_ids, term_embs
