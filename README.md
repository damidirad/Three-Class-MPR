# mcMPR

**Based on the paper and reproduction package by Zhang et al. (2025): https://github.com/jizhi-zhang/MPR/tree/main**.

## Overview

**Three-Class-MPR** implements methods for Multiple Prior-Guided Robust Optimization (MPR) (Zhang et al., 2025) in recommender systems using matrix factorization (MF) and sensitive attribute prediction. The repository is focused on expanding binary sensitive-attribute MPR to handle multi-class scenarios.

---

## Features

- **MF Models**: Standard and fairness-regularized collaborative filtering, enabling sensitive attribute-aware training.
- **Fairness Training**: Implements multi-class demographic parity, handling cases with only partial knowledge of user-sensitive attributes.
- **Sensitive Attribute Prediction (SST)**: Neural models for inferring sensitive attributes when they are partially disclosed.
- **Flexible Experimental Setup**: Configurable for different datasets, numbers of sensitive attribute classes, and disclosure ratios.
- **Evaluation**: Metrics for both model accuracy and fairness (demographic parity gaps) across user groups.
- **Configurable via Python Dataclasses**: Easily adjust hyperparameters and dataset settings in `config.py`.

---

## Key Files & Structure

| File / Dir                         | Purpose                                                                                   |
|:-----------------------------------|:------------------------------------------------------------------------------------------|
| `MF.py`                            | Implements standard MF model.                                                             |
| `MPR.py`                           | Main script for running MPR experiments.                                                  |
| `MPR_fairness_training.py`         | Core method for fairness-constrained training with partial sensitive attribute info.      |
| `SST.py`                           | Neural network for predicting sensitive user attributes from MF embeddings.               |
| `pretrain_baseline.py`             | Script for baseline MF model pretraining.                                                 |
| `predict_sensitive_labels.py`      | Script to generate predicted sensitive attribute distributions.                           |
| `evaluation.py`                    | Functions to evaluate models on accuracy and fairness metrics.                            |
| `helpers.py`                       | Utility functions to manage data splits, prior setups, RMSE calculation, etc.             |
| `config.py`                        | Central hyperparameter and experiment config.                                             |
| `datasets/`                        | Contains data splits, including sensitive attribute CSVs (e.g. `sensitive_attribute.csv`).|
| `pretrained_models/`               | Directory for pretrained MF model weights.                                                |
| `deliverables/`                    | Directory for predicted sensitive-attribute distributions and MPR experiment results.     |

---

## Example Use

1. **Edit Configuration**: Edit `config.py` or pass arguments in `MPR.py` for your task, dataset, or ratios of disclosed sensitive info.

2. **Pretrain MF Baseline**:
   ```sh
   python pretrain_baseline.py --task_type <dataset>
   ```

3. **Predict sensitive attribute distributions under a range of prior distributions.**
   ```sh
   python predict_sensitive_labels.py --task_type <dataset> --prior_resample_idx <prior idx in resample range> --unfair_model <path to unfair MF model>
   ```

3. **Train with MPR**
   ```sh
   python MPR.py --task_type <dataset> --s_attr <attribute> --s_ratios <r1> <r2> ... --fair_reg <fval>
   ```

4. **Evaluate Fairness and Accuracy**:
   - View saved logs and metrics for RMSE and demographic parity gap in output.

---

## Datasets

- The repository is designed for real-world datasets (e.g., MovieLens or Lastfm) with (binary or multi-class) sensitive attributes.
- Example: `datasets/ml-1m/train.csv` has user IDs, item recommendations, and an implicit feedback label.
- The datasets labeled as 'synthetic' contain MovieLens and Lastfm datasets augmented with an extra gender class for multi-class fairness trials.
- Example: `datasets/Lastfm-360K-synthetic/sensitive_attribute.csv` has user IDs and multi-class gender labels. 

---

## Requirements

- Python 3.8+
- PyTorch
- pandas, numpy, tqdm
- (optional for notebooks): Jupyter

_See individual script files for additional dependencies as needed._

---

## Reference

Zhang, J., Shen, H., Shi, T., Bao, K., Chen, X., Zhang, Y., & Feng, F. (2025). Fair recommendation with biased-limited sensitive attribute. \
In *Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)*, 1717–1727.

---

## License

**Private research repository. For personal or academic use only.**

---

## Quick Start

```sh
# (Recommended) Set up a virtual environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure your experiment in config.py or pass CLI arguments
```

---

## Contact

Maintained by:
[@damidirad](https://github.com/damidirad), 
[@k-sert](https://github.com/k-sert), 
[@Gerrit499](https://github.com/Gerrit499),
[@Papa-Beer](https://github.com/Papa-Beer). 

For questions, open an issue or contact directly.
