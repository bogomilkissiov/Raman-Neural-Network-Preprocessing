# Downstream Classifiers

Shared classifier suite for benchmarking Raman preprocessing methods (e.g. Raw, Conventional `pre.py`, `PolyGaussNet V3`, `PolyGaussNet V3.4-8sec`) across multiple model families:

1. **`LinearSpectralClassifier` (`linear.py`)**
   - `StandardScaler` + `LogisticRegression` (`lbfgs`, `C=1.0`, `max_iter=2000`).
   - Minimal-capacity linear baseline; directly tests whether preprocessing improves linear separability.

2. **`XGBSpectralClassifier` (`xgb.py`)**
   - `xgboost.XGBClassifier` (`tree_method="hist"`, `max_depth=6`, `lr=0.05`, early stopping on validation set).
   - Non-linear tabular baseline without 1D spatial inductive bias.

3. **`ResNet1D` / `BacteriaResNetClassifier` (`resnet1d.py`)**
   - Mathematically equivalent modern PyTorch implementation of the 1D ResNet from Ho et al., *Nature Communications* 2019 (`https://github.com/csho33/bacteria-ID/blob/master/resnet.py`).
   - Exact layer/channel/stride layout (`hidden_sizes=[100]*6`, `num_blocks=[2]*6`, `in_channels=64`, `kernel_size=5`, `strides=[1, 2, 2, 2, 2, 2]`, `BatchNorm1d`, Adam `lr=1e-3, betas=(0.5, 0.999)`).
   - Updated only to remove deprecated `torch.autograd.Variable` wrappers and add validation early stopping.

4. **Shared Downstream Runner (`benchmark.py`)**
   - `evaluate_method(...)`: Preprocesses `(X_train, X_val, X_test)` through a single preprocessing method, trains and evaluates all configured classifiers across one or more seeds under fixed hyperparameters/epochs, and cleans up GPU memory between runs.
   - `run_downstream_benchmark(...)`: Evaluates a dictionary of preprocessing methods (`{"Raw": None, "Conventional": ..., "PolyGaussNet V3": ..., "PolyGaussNet V3.4-8sec": ...}`) on identical splits, seeds, and epochs; computes accuracy, balanced accuracy, macro F1, log-loss, OvR ROC-AUC, throughput, paired deltas, and exact two-sided McNemar's test p-values vs baseline; exports `.txt`, `.json`, `.csv`, `.npz`, and `.png` artifacts.
