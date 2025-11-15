| experiment              | metric             |   base_mean |   base_std |   improved_mean |   improved_std |   delta_mean |
|:------------------------|:-------------------|------------:|-----------:|----------------:|---------------:|-------------:|
| ablation_diffusion      | FID                |       65.3  |        nan |           48.7  |            nan |       -25.42 |
| ablation_rag            | BLEU               |        0.42 |        nan |            0.58 |            nan |        38.1  |
| constraint_weight_sweep | Constraint_Factual |        0.72 |        nan |            0.91 |            nan |        26.38 |
| full_model_integration  | CLIPScore          |        0.68 |        nan |            0.79 |            nan |        16.17 |