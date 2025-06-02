# Post-hoc Ensembling Algorithms

All methods expose a unified API (`sample`, `get_weights`, …).

| Category | Method | Brief Description |
|----------|--------|-------------------|
| **Single** | *Single Best* | Chooses the pipeline with lowest validation loss. |
| **Random** | *Random* | Uniformly samples 50 pipelines and averages them. |
| **Top-N** | *Top-5 / Top-50* | Fixed-size ensemble of the best *N* models. |
| **Greedy** | *Caruana ’04* | Iteratively adds the model that most improves the metric. |
| **Quick** | *Strict Greedy* | Adds a model **only** if it strictly improves the metric. |
| **CMA-ES** | Purucker ’23 | Evolutive strategy on the weight simplex. |
| **Model Average** | *BMA-style* | Learns weights via gradient descent. |
| **DivBO** | Shen ’22  | Bayesian Opt. balancing diversity & performance. |
| **Ensemble Opt.** | Levesque ’16 | Iteratively replaces members via BO. |
| **Stackers** | SVM, RF, GBDT, LR, XGB, LGBM | Learns a meta-model on concatenated predictions. |
| **DES** | KNOP, KNORAE, MetaDES | Dynamic ensemble selection (DESlib). |
| **Akaike / PMA** | Wagenmakers ’04  | Analytical weights from relative AIC. |
