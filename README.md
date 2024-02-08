# AntiFraud
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-amazon-fraud)](https://paperswithcode.com/sota/fraud-detection-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-amazon-fraud)](https://paperswithcode.com/sota/node-classification-on-amazon-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/fraud-detection-on-yelp-fraud)](https://paperswithcode.com/sota/fraud-detection-on-yelp-fraud?p=semi-supervised-credit-card-fraud-detection)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/semi-supervised-credit-card-fraud-detection/node-classification-on-yelpchi)](https://paperswithcode.com/sota/node-classification-on-yelpchi?p=semi-supervised-credit-card-fraud-detection)

A Financial Fraud Detection Framework.

Source codes implementation of papers:
- `MCNN`: Credit card fraud detection using convolutional neural networks, in ICONIP 2016. 
- `STAN`: Spatio-temporal attention-based neural network for credit card fraud detection, in AAAI2020
- `STAGN`: Graph Neural Network for Fraud Detection via Spatial-temporal Attention, in TKDE2020
- `GTAN`: Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation, in AAAI2023
- `RGTAN`: Enhancing Attribute-driven Fraud Detection with Risk-aware Graph Representation
- `GGTAN`: GGTAN: A Novel GAT-Enhanced Gated Temporal Attention Network for Advanced Fraud Detection in Financial Transactions
	- [Project Link](https://drive.google.com/file/d/1P_loGSXID4GVcWVcSwu6gA3n2Bqn8ElX/view?usp=sharing)



## Usage

### Data processing
1. Run `unzip /data/Amazon.zip`, `unzip /data/YelpChi.zip`, `unzip /data/S-FFSD.zip` and `unzip /data/IBM.zip`to unzip the datasets; 
2. Run `python feature_engineering/data_process.py` to pre-process all datasets needed in this repo.
	* If you just want to use FFSD and IBM dataset with `GTAN` and `GGTAN` method, then run `python feature_engineering/data_process_ggtan.py`



### Training & Evalutaion

To test implementations of `MCNN`, `STAN` and `STAGN`, run
```
python main.py --method mcnn
python main.py --method stan
python main.py --method stagn
```
Configuration files can be found in `config/mcnn_cfg.yaml`, `config/stan_cfg.yaml` and `config/stagn_cfg.yaml`, respectively.

Models in `GTAN`, `RGTAN` and `GGTAN` can be run via:
```
python main.py --method gtan
python main.py --method rgtan
python main.py --method ggtan
```
For specification of hyperparameters, please refer to `config/gtan_cfg.yaml`, `config/rgtan_cfg.yaml`, and `congif/ggtan_cfg.yaml`.


### Data Description

This repository utilizes four datasets for model experiments: YelpChi, Amazon, S-FFSD, and IBM.

#### YelpChi and Amazon Datasets
These datasets are sourced from [CARE-GNN](https://dl.acm.org/doi/abs/10.1145/3340531.3411903) and the original data can be found in [this repository](https://github.com/YingtongDou/CARE-GNN/tree/master/data).

#### S-FFSD Dataset
S-FFSD is a simulated, smaller version of a financial fraud semi-supervised dataset. Its description is as follows:

| Name     | Type       | Range                  | Note                                    |
|----------|------------|------------------------|-----------------------------------------|
| Time     | np.int32   | 0 to N                 | N: Number of transactions.              |
| Source   | string     | S_0 to S_ns            | ns: Number of transaction senders.      |
| Target   | string     | T_0 to T_nt            | nt: Number of transaction receivers.    |
| Amount   | np.float32 | 0.00 to np.inf         | Transaction amount.                     |
| Location | string     | L_0 to L_nl            | nl: Number of transaction locations.    |
| Type     | string     | TP_0 to TP_np          | np: Number of different transaction types. |
| Labels   | np.int32   | 0 to 2                 | 2 denotes 'unlabeled'.                  |

#### IBM Credit Card Transaction Dataset
This is a publicly available synthetic dataset for fraud detection research, containing simulated transaction data provided by IBM.

##### Dataset Highlights
- **Total Transactions**: 24 million unique transactions.
- **Unique Merchants**: 6,000.
- **Unique Cards**: 100,000.
- **Fraudulent Transactions**: 30,000 samples (0.1% of total transactions).

##### Key Characteristics
- **Class Imbalance**: More non-fraudulent transactions, reflecting real-world scenarios.
- **Fraud Labels**: Indicates whether a transaction is fraudulent.
- **Data Nature**: Synthetic, not linked to real customers or financial institutions.

##### Data Accessibility
- **Local Download**: [IBM Dataset Link](https://ibm.ent.box.com/v/tabformer-data)
- **Kaggle**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/ealtman2019/credit-card-transactions)

##### Usage
The dataset is used in our experiments on a sample of approximately 100,000 transactions.

> Seeking more public datasets for interesting studies! Suggestions are welcome.

<!--## Test Result

The performance of five models tested on three datasets are listed as follows:
| |YelpChi| | |Amazon| | |S-FFSD| | |
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
| |AUC|F1|AP|AUC|F1|AP|AUC|F1|AP|
|MCNN||- | -| -| -| -|0.7129|0.6861|0.3309|
|STAN|- |- | -| -| -| -|0.7422|0.6698|0.3324|
|STAGN|- |- | -| -| -| -|0.7659|0.6852|0.3599|
|GTAN|0.9241|0.7988|0.7513|0.9630|0.9213|0.8838|0.8286|0.7336|0.6585|
|RGTAN|0.9498|0.8492|0.8241|0.9750|0.9200|0.8926|0.8461|0.7513|0.6939|
-->

## Model Performance Summary on S-FFSD and IBM Datasets

| Model | S-FFSD     |       |       | IBM         |       |       |
|-------|------------|-------|-------|-------------|-------|-------|
|       | AUC        | F1    | AP    | AUC         | F1    | AP    |
| XGB   | 0.7931     | 0.6512| 0.4830| 0.9272      | 0.8941| 0.8111|
| MCNN  | 0.7129     | 0.6861| 0.3309| 0.8771±0.001| 0.7814±0.004| 0.4084±0.007|
| GAT   | 0.7302±0.005| 0.6147±0.006| - | 0.9256±0.001| 0.8325±0.025| - |
| GTAN  | 0.8286     | 0.7336| 0.6585| 0.9140±0.010| 0.6959±0.059| 0.5424±0.040|
| GGTAN | 0.8951±0.003| 0.7853±0.006| 0.7530±0.006| 0.9952±0.000| 0.9496±0.002| 0.9646±0.002|


<!--
> `MCNN`, `STAN` and `STAGN` are presently not applicable to YelpChi and Amazon datasets.
-->
## Repo Structure
The repository is organized as follows:
- `models/`: the pre-trained models for each method. The readers could either train the models by themselves or directly use our pre-trained models;
- `data/`: dataset files;
- `config/`: configuration files for different models;
- `feature_engineering/`: data processing;
- `methods/`: implementations of models;
- `main.py`: organize all models;
- `requirements.txt`: package dependencies;

    
## Requirements
```
python           3.7
scikit-learn     1.0.2
pandas           1.3.5
numpy            1.21.6
networkx         2.6.3
scipy            1.7.3
torch            1.12.1+cu113
dgl-cu113        0.8.1
torch_geometric  2.4.0
```

## Citing

If you find *Antifraud* is useful for your research, please consider citing the following papers:

    @inproceedings{Xiang2023SemiSupervisedCC,
        title={Semi-supervised Credit Card Fraud Detection via Attribute-driven Graph Representation},
        author={Sheng Xiang and Mingzhi Zhu and Dawei Cheng and Enxia Li and Ruihui Zhao and Yi Ouyang and Ling Chen and Yefeng Zheng},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        year={2023}
    }
    @article{cheng2020graph,
        title={Graph Neural Network for Fraud Detection via Spatial-temporal Attention},
        author={Cheng, Dawei and Wang, Xiaoyang and Zhang, Ying and Zhang, Liqing},
        journal={IEEE Transactions on Knowledge and Data Engineering},
        year={2020},
        publisher={IEEE}
    }
    @inproceedings{cheng2020spatio,
        title={Spatio-temporal attention-based neural network for credit card fraud detection},
        author={Cheng, Dawei and Xiang, Sheng and Shang, Chencheng and Zhang, Yiyi and Yang, Fangzhou and Zhang, Liqing},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={34},
        number={01},
        pages={362--369},
        year={2020}
    }
    @inproceedings{fu2016credit,
        title={Credit card fraud detection using convolutional neural networks},
        author={Fu, Kang and Cheng, Dawei and Tu, Yi and Zhang, Liqing},
        booktitle={International Conference on Neural Information Processing},
        pages={483--490},
        year={2016},
        organization={Springer}
    }
