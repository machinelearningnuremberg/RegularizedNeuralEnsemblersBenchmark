# Post-hoc Ensembling Algorithms

All methods expose a unified API (`sample`, `get_weights`, …).

| Category     | Method                  | CLI Name      | Brief Description |
|--------------|--------------------------|---------------|-------------------|
| **Single**   | *Single Best*            | `single`      | Chooses the pipeline with lowest validation loss. |
| **Random**   | *Random*                 | `random`      | Uniformly samples 50 pipelines and averages them. |
| **Top-N**    | *Top-5 / Top-50*         | `topm`        | Fixed-size ensemble of the best *N* models. |
| **Greedy**   | *Caruana ’04*            | `greedy`      | Iteratively adds the model that most improves the metric. |
| **Quick**    | *Strict Greedy*          | `quick`       | Adds a model **only** if it strictly improves the metric. |
| **CMA-ES**   | Purucker ’23             | `cmaes`       | Evolutive strategy on the weight simplex. |
| **Model Avg**| *BMA-style*              | `neural`      | Learns weights via gradient descent (our neural ensembler). |
| **DivBO**    | Shen ’22                 |  XXXXXXXXXXXX | Bayesian Opt. balancing diversity & performance. |
| **Ens Opt.** | Levesque ’16             |  XXXXXXXXXXXX | Replaces ensemble members iteratively via BO. |
| **Stackers** | SVM, RF, GBDT, LR, etc.  | `sks`         | Learns a meta-model on concatenated base predictions. |
| **DES**      | KNOP, KNORAE, MetaDES    | `des`         | Dynamic ensemble selection via DESlib. |
| **Akaike**   | Wagenmakers ’04          | `akaike`      | Analytical weights from relative AIC. |
