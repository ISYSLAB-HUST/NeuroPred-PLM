import torch
import esm
from argparse import Namespace
import pathlib
import urllib

def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device,
                        dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def load_model_and_alphabet_core(args_dict, regression_data=None):
    args_dict = torch.load(args_dict)
    alphabet = esm.Alphabet.from_architecture(args_dict["args"].arch)

    # upgrade state dict
    pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
    prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
    model_args = {pra(arg[0]): arg[1] for arg in vars(args_dict["args"]).items()}
    model_type = esm.ProteinBertModel

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )
    return model, alphabet


def load_hub_workaround(url):
    try:
        data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    except RuntimeError:
        # Pytorch version issue - see https://github.com/pytorch/pytorch/issues/43106
        fn = pathlib.Path(url).name
        data = torch.load(
            f"{torch.hub.get_dir()}/checkpoints/{fn}",
            map_location="cpu",
        )
    except urllib.error.HTTPError as e:
        raise Exception(f"Could not load {url}, check your network!")
    return data