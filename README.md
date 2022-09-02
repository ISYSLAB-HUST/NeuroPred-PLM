## NeuroPred-PLM: an interpretable and robust model for prediction of neuropeptides by protein language model

### Requirements
To install requirements:

```
pip install git+https://github.com/ISYSLAB-HUST/NeuroPred-PLM.git
```
### Usage [<img src="https://colab.research.google.com/assets/colab-badge.svg">](https://colab.research.google.com/github/ISYSLAB-HUST/NeuroPred-PLM/blob/master/notebook/NeuroPred_PLM_test.ipynb)


```
import torch
from NeuroPredPLM.predict import predict
data = [
    ("peptide_1", "IGLRLPNMLKF"),
    ("peptide_2", "QAAQFKVWSASELVD"),
    ("peptide_3","LRSPKMMHKSGCFGRRLDRIGSLSGLGCNVLRKY")
]

device = "cuda" if torch.cuda.is_available() else "cpu" 
neuropeptide_pred = predict(data,device)
# {peptide_id:[Type:int(1->neuropeptide,0->non-neuropeptide), attention score:nd.array]}
```

### Contact
If you have any questions, comments, or would like to report a bug, please file a Github issue or contact me at wanglei94@hust.edu.cn.