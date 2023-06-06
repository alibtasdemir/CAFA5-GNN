import pandas as pd
import torch
from torch_geometric.loader import LinkNeighborLoader


def get_unique_ids(prot_ids, term_ids):
    unique_protein_id = pd.DataFrame(
        data={
            'proteinId': prot_ids,
            'mappedId': pd.RangeIndex(len(prot_ids))
        }
    )
    unique_annot_id = pd.DataFrame(
        data={
            'termId': term_ids,
            'mappedId': pd.RangeIndex(len(term_ids))
        }
    )

    return unique_protein_id, unique_annot_id


def get_mappings(prot_terms_df, prot_ids, term_ids):
    unique_protein_id, unique_annot_id = \
        get_unique_ids(
            prot_ids,
            term_ids
        )

    merged_protein_id = pd.merge(
        prot_terms_df['EntryID'], unique_protein_id,
        left_on='EntryID', right_on='proteinId', how='left'
    )

    merged_protein_id = torch.from_numpy(merged_protein_id['mappedId'].values)
    merged_term_id = pd.merge(
        prot_terms_df['term'], unique_annot_id,
        left_on='term', right_on='termId', how='left'
    )
    merged_term_id = torch.from_numpy(merged_term_id['mappedId'].values)

    edge_index_protein_to_term = torch.stack([merged_protein_id, merged_term_id], dim=0)

    return unique_protein_id, unique_annot_id, edge_index_protein_to_term


def getTrainLoader(data, neg_sampling_ratio=2.0, batch_size=128):
    edge_label_index = data["protein", "function", "term"].edge_label_index
    edge_label = data["protein", "function", "term"].edge_label
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        neg_sampling_ratio=neg_sampling_ratio,
        edge_label_index=(("protein", "function", "term"), edge_label_index),
        edge_label=edge_label,
        batch_size=batch_size,
        shuffle=True
    )
    return loader


def getValidationLoader(data, batch_size=128):
    edge_label_index = data["protein", "function", "term"].edge_label_index
    edge_label = data["protein", "function", "term"].edge_label
    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=[20, 10],
        edge_label_index=(("protein", "function", "term"), edge_label_index),
        edge_label=edge_label,
        batch_size=3 * batch_size,
        shuffle=False,
    )
    return loader
