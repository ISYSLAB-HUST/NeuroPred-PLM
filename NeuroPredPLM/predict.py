from .model import EsmModel
from .utils import load_hub_workaround
import torch

MODEL_URL = "https://zenodo.org/record/7042286/files/model.pth"

def predict(peptide_list, device='cpu'):
    with torch.no_grad():
        neuroPred_model = EsmModel()
        neuroPred_model.eval()
        state_dict = load_hub_workaround(MODEL_URL)
        # state_dict = torch.load("/mnt/d/protein-net/Neuropep-ESM/model.pth", map_location="cpu")
        neuroPred_model.load_state_dict(state_dict)
        neuroPred_model = neuroPred_model.to(device)
        prob, att = neuroPred_model(peptide_list, device)
        pred = torch.argmax(prob, dim=-1).cpu().tolist()
        att = att.cpu().numpy()
        out = {i[0]:[j,m[:, :len(i[1])]] for i, j, m in zip(peptide_list, pred, att)}
    return out
    