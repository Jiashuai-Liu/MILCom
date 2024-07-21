import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typeguard import typechecked
from typing import Tuple, Optional, Sequence

class StableSoftmax(torch.nn.Module):
    @typechecked
    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.nn.LogSoftmax(dim=self.dim)(inputs).exp()

class Sum:
    def pool(self, features, **kwargs) -> int:
        return torch.sum(features, **kwargs)

class AdditiveClassifier(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
    ):

        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.additive_function = Sum()
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = torch.nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dims):
                layers.append(self.hidden_activation)
                layers.append(nn.Dropout(0.25))
        model = torch.nn.Sequential(*layers)
        return model

    def forward(self, features, attention):
        attended_features = attention * features
        patch_logits = self.model(attended_features)
        logits = self.additive_function.pool(patch_logits, dim=1, keepdim=False)
        classifier_out_dict = {}
        classifier_out_dict['logits'] = logits
        classifier_out_dict['patch_logits'] = patch_logits
        return classifier_out_dict

class DefaultAttentionModule(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        input_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = nn.ReLU(),
        output_activation: torch.nn.Module = StableSoftmax(dim=1),
        use_batch_norm: bool = True,
        track_bn_stats: bool = True,
    ):

        super().__init__()

        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats

        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [1]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dims):
                if self.use_batch_norm:
                    layers.append(nn.BatchNorm1d(nodes_out, track_running_stats=self.track_bn_stats))
                layers.append(self.hidden_activation)
                layers.append(nn.Dropout(0.25))
        model = nn.Sequential(*layers)
        return model

    def forward(self, features, bag_size):
        out = self.model(features)
        out = out.view([-1, bag_size])
        attention = self.output_activation(out)
        return attention.unsqueeze(-1)
    
class DefaultMILGraph(torch.nn.Module):
    @typechecked
    def __init__(
        self,
        classifier: torch.nn.Module,
        pointer: torch.nn.Module,
    ):
        super().__init__()
        self.classifier = classifier
        self.pointer = pointer
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = self.classifier.to(device)
        self.pointer = self.pointer.to(device)

    def forward(self, h):
        
        batch_size = 1
        bag_size = h.shape[0]
        
        features = h
        attention = self.pointer(features, bag_size)
        
        if not torch.all(attention >= 0):
            raise ValueError("{}: Attention weights cannot be negative".format(attention))

        features = features.view([batch_size, bag_size] + list(features.shape[1:]))  # separate batch and bag dim
        
        classifier_out_dict = self.classifier(features, attention)
        
        bag_logits = classifier_out_dict['logits']

        patch_logits = classifier_out_dict['patch_logits'] if 'patch_logits' in classifier_out_dict else None
        
        bag_prob = F.softmax(bag_logits, dim = 1)
        
        patch_prob = F.softmax(patch_logits.squeeze(0), dim = 1)
        
        Y_hat = torch.topk(bag_prob.squeeze(0), 1, dim = 0)[1]
        
        return bag_logits, bag_prob, Y_hat, patch_prob.T, {}