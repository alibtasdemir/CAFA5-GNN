import torch
from torch.nn import Module
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from engine import auroc


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_prot: Tensor, x_term: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_prot = x_prot[edge_label_index[0]]
        edge_feat_term = x_term[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        out = (edge_feat_prot * edge_feat_term).sum(dim=-1)

        return F.sigmoid(out)


class Model(Module):
    def __init__(self, hidden_channels, protein_input, term_input, metadata):
        super().__init__()

        # PROTEIN FEATURE INPUT
        self.prot_lin = torch.nn.Linear(1024, hidden_channels)

        # EMBEDDING LAYERS
        self.prot_emb = torch.nn.Embedding(protein_input, hidden_channels)
        self.term_emb = torch.nn.Embedding(term_input, hidden_channels)

        # Homogenous GNN
        self.gnn = GNN(hidden_channels)

        # To heterogeneous
        self.gnn = to_hetero(self.gnn, metadata=metadata)

        # Classifier
        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
            "term": self.term_emb(data["term"].node_id),
            "protein": self.prot_lin(data["protein"].x) + self.prot_emb(data["protein"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["protein"],
            x_dict["term"],
            data["protein", "function", "term"].edge_label_index,
        )

        return pred

    def training_step(self, batch):
        out = self(batch)
        labels = batch["protein", "function", "term"].edge_label
        loss = F.binary_cross_entropy(out, labels)
        return loss

    """
    def validation_step(self, batch):
        X, labels = batch, batch["protein", "function", "term"].edge_label
        out = self(X)  # Generate predictions
        loss = F.binary_cross_entropy(out, labels)  # Calculate loss
        acc = auroc(out, labels)  # Calculate accuracy
        return {'Validation_loss': loss.detach(), 'Validation_acc': acc}
    """
    def validation_step(self, batch):
        X, labels = batch, batch["protein", "function", "term"].edge_label
        out = self(X)  # Generate predictions
        loss = F.binary_cross_entropy(out, labels)  # Calculate loss
        return loss.detach(), out, labels

    def epoch_end(self, epoch, result):
        if epoch % 1 == 0:
            print(
                "Epoch [{}], Train_loss: {:.4f}, Validation_loss: {:.4f}, Validation_AUC: {:.4f}, Validation_F1: {:.4f}".format(
                    epoch, result['Train_loss'], result['Validation_loss'], result['Validation_AUC'], result['Validation_F1']
                )
            )

    def validation_epoch_end(self, metrics, losses):
        batch_losses = [x['Validation_loss'] for x in losses]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses

        epoch_auc = metrics["auc"]
        epoch_f1 = metrics["f1"]
        return {'Validation_loss': epoch_loss.item(), 'Validation_AUC': epoch_auc.item(), 'Validation_F1': epoch_f1.item()}
