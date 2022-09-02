"""
main model 
"""
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange
import os

from .utils import length_to_mask, load_model_and_alphabet_core


class EsmModel(nn.Module):
    def __init__(self, hidden_size=64, num_labels=2, projection_size=24, head=12):
        super().__init__()

        basedir = os.path.abspath(os.path.dirname(__file__))
        self.esm, self.alphabet = load_model_and_alphabet_core(os.path.join(basedir, 'args.pt'))
        self.num_labels = num_labels
        self.head = head
        self.hidden_size = hidden_size
        self.projection = nn.Linear(hidden_size, projection_size)
        self.cov_1 = nn.Conv1d(projection_size, projection_size, kernel_size=3, padding='same')
        self.cov_2 = nn.Conv1d(projection_size, int(projection_size/2), kernel_size=1, padding='same')
        # self.gating = nn.Linear(projection_size, projection_size)
        self.W = nn.Parameter(torch.randn((head, int(projection_size/2))))
        # self.mu = nn.Parameter(torch.randn((1, 768)))
        self.fcn = nn.Sequential(nn.Linear(int(projection_size/2)*head, int(projection_size/2)),
                                nn.ReLU(), nn.Linear(int(projection_size/2), num_labels))      
    

    def forward(self, peptide_list, device='cpu'):
        peptide_length = [len(i[1]) for i in peptide_list]
        batch_converter = self.alphabet.get_batch_converter()
        _, _, batch_tokens = batch_converter(peptide_list)
        batch_tokens = batch_tokens.to(device)
        protein_dict = self.esm(batch_tokens, repr_layers=[12], return_contacts=False)
        protein_embeddings = protein_dict["representations"][12][:, 1:, :]
        protein_embed = rearrange(protein_embeddings, 'b l (h d)-> (b h) l d', h=self.head)
        representations = self.projection(protein_embed)
        representations = rearrange(representations, 'b l d -> b d l')
        representation_cov = F.relu(self.cov_1(representations))
        representation_cov =  F.relu(self.cov_2(representation_cov))
        representations = rearrange(representation_cov, '(b h) d l -> b h l d', h=self.head)
        att = torch.einsum('bhld,hd->bhl', representations, self.W)
        mask = length_to_mask(torch.tensor(peptide_length)).to(device)
        att = att.masked_fill(mask.unsqueeze(1)==0, -np.inf)
        att= F.softmax(att, dim=-1)
        # print(att)
        representations = rearrange(representations * att.unsqueeze(-1), 'b h l d -> b l (h d)')
        representations = torch.sum(representations, dim=1)
        return self.fcn(representations), att


