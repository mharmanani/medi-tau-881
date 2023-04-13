# Prostate Cancer Detection from Ultrasounds using Temperature as Uncertainty for Contrastive Learning

> **N.B.** This repository contains a summary of my conributions to the Med-i TRUS-Net repository, where the data, code, and other experiments from the prostate group is stored. The code in this repo does not work on its own. For full reproducibility, please check: https://github.com/med-i-lab/TRUSnet.

## Abstract
MOTIVATION: Coarse labelling of prostate biopsy cores using histopathology presents a major challenge for supervised deep learning models due to the inherent presence of noise in the labels. This can negatively impact the performance of these algorithms, as they tend to overfit the noise and produce poor generalization. Contrastive learning has emerged as a promising approach for learning robust representations of ultrasound images without the use of labels. Unfortunately, contrastive models cannot estimate the uncertainty of the generated representations at present, which limits their applicability in clinical practice. METHODS: We use a dataset of 1028 prostate biopsy cores from 391 patients obtained in two clinical centers. We apply the Temperature as Uncertainty (TaU) framework for Simple Contrastive Learning (SimCLR) to the problem of prostate cancer (PCa) detection from ultrasounds. We then evaluate this method against other baselines commonly used for uncertainty estimation, comparing both performance and calibration error. RESULTS: TaU-SimCLR outperforms current contrastive learning baselines, with a test AUROC score of 80.5% and a Brier score of 0.21 when evaluated on biopsy core-based cancer detection. These results make it a promising method for reliable detection of prostate cancer from ultrasounds.

## File Structure
### Drivers
The main logic for each model is implemented in what's called a 'driver'. The code is run with the `python3 main.py experiment=[experiment_name]`, where `[experiment_name]` is the name of a `.yaml` file in the `experiment` directory specifying the driver to run. In other words, the following code
```
python3 main.py experiment=04_881_TaU_NoOOD_MHarmanani.yaml
```

will run the `tau_driver.py` file with the hyperparameters specified in the file `04_881_TaU_NoOOD_MHarmanani.yaml`. 

### Experiments
As mentioned, experiment files are `.yaml` files that specify the hyperparameter values for each experiment. For example

```
defaults:
  - /datamodule@driver.datamodule: ssl_tau
  - /datamodule@driver.sl_datamodule: sl_tau

name: tau_test_no_ood

driver: 
  _target_: src.driver.tau_driver.TaUDriver
  t: 0.1
  eps: 0.000001
  train_epochs: 200
  finetune_epochs: 100
  learning_rate: 1e-4
  weight_decay: 1e-6
```

This snippet from `04_881_TaU_NoOOD_MHarmanani.yaml` runs TaU for 200 epochs of contrastive pre-training, and 100 epochs of finetuning, with a learning rate of $10^{-4}$. The `defaults` part at the top specify the data loaders to use for each step. `ssl_tau` uses data from the entire prostate region, and `sl_tau` uses labelled data from the needle region. Both datasets are constructed using cores from centers UVA and CRCEO.

### Figures & Results
The `generate_plots.ipynb` notebook is responsible for reading the evaluation metrics stored in the `<model>_vs_perf_df.csv` files and generating:
- precision vs confidence plots
- BS/ECE point plots
- BS/ECE average values
- remaining cores vs confidence plots

The figures themselves are also saved in the `figs` folder.


### Data modules
Datasets are loaded by a software called a data module. The data modules are not included here, but we include files that control several crucial parameters, including:
- which centers are used, 
- how undersampling is done, 
- which train/val/test split is used,  
- the batch size

The files are located in the `datamodule` directory. In particular, we used `ssl_tau` and `sl_tau`. Both data modules refer to something called a 'split', which refers to instructions on how to split the data. These are located in `datamodule/splits`, and we used the `uva_crceo.yaml` split. 

### WandB Ablation Reports
The results of our ablation experiments are highlighted in the paper in some detail, but their outcome is specified only briefly. We show the training curves and test metrics for each model trained in the `wandb_reports` folder. It will contain 2 reports written in $\LaTeX$, one for the SimCLR baseline, and one for the TaU-SimCLR model.  