# HF-ABC

Research code for **HF-ABC: A constraint-aware multi-fidelity generalized artificial bee colony algorithm for configuring wastewater process forecasting models**.

HF-ABC combines typed candidate generation, feasibility projection, progressive multi-fidelity evaluation and high-fidelity verification. The study evaluates these mechanisms on controlled mixed-variable water-system tasks and applies the configuration workflow to multivariate forecasting at the Avedøre wastewater treatment plant.

## Contents

- `mechanism/`: controlled mixed-variable benchmarks, fixed instance selection and analysis scripts.
- `application/`: forecasting models, HF-ABC configuration search and baseline entry points.
- `data/aved_n2o/manifest.json`: feature order, key channels and local dataset path.

## Data preparation

Obtain the Avedøre source time series from Hansen et al. (2024), Mendeley Data version 4, DOI [10.17632/xmbxhscgpr.4](https://doi.org/10.17632/xmbxhscgpr.4). Place the prepared 24-variable, two-minute training CSV at `data/aved_n2o/water_quality_clean.csv`, matching the included manifest. The study's retained processed input contains 526,320 observations. The generic preprocessing utility is `application/scripts/prepare_water_quality_dataset.py`; run it with `--help` to inspect its options and adjust preprocessing to the intended period and feature set.

Use Python 3.10 for the retained forecasting dependencies. Run forecasting commands from `application/` so relative data paths resolve correctly. New runs write to `reruns/`.

## Section 5 mechanism experiments

Use a dedicated Python environment:

```sh
cd mechanism
python -m pip install -r requirements.txt
python mvs/run_m3.py --execute-smoke --output ../reruns/cpu_smoke
python mvs/run_m3.py --execute-main --seeds 20 --output ../reruns/cpu_main
python mvs/analyze_m3.py --input ../reruns/cpu_main/m3_main_v1_20_seeds
```

The main experiment uses 15 fixed instances, seven methods and search seeds 303600–303619. Nested fidelities are 9/36/72/144 steps; the nominal cost is 3,744 steps per search. Main comparisons aggregate each instance and method by the median across search seeds, then compare paired instances.

The CPU timing supplement uses the same instances and methods:

```sh
python mvs/run_m3_equal_wallclock.py --instances 15 --seeds 5 --seconds 1 --output ../reruns/cpu_timed
python mvs/rebuild_equal_wallclock_checkpoints.py ../reruns/cpu_timed
```

Use `checkpoints_penalized.csv` for this supplement: average seeds within each instance and method, rank methods within each instance, then average ranks over instances. Timing depends on hardware and system load.

Unit checks for the mechanism implementation:

```sh
python -m unittest discover -s mvs -p "test_*.py"
```

## Section 6 forecasting application

Install the forecasting environment separately from the lightweight CPU benchmark. The retained requirements specify PyTorch 2.0.0, NumPy 1.23.5, pandas 1.5.3 and scikit-learn 1.2.2; install a PyTorch build matching your GPU environment.

```sh
cd application
python -m pip install -r requirements.txt
python -m pip install -r requirements_optuna_tpe.txt
python scripts/run_water_quality_suite.py --manifest ../data/aved_n2o/manifest.json --dataset_name AvedoreN2O --pred_len 120 --model_variant iTransformer_CSG --method pf_abc --hf_abc --search_seeds 2026 --final_seeds 2026 2027 2028 --race_fidelities 1 3 6 10 --race_init_size 48 --race_keep 16 8 4 4 --race_challengers 0 8 4 0 --elite_top_m 4 --verify_top_k 4 --verify_epochs 20 --final_train_epochs 20 --num_workers 0 --output_dir ../reruns/avedore_h120
```

`pf_abc` is the retained command-line method name; `--hf_abc` enables the HF-ABC configuration workflow. The H=120 and H=240 schedule uses 232 candidate-epochs for search and 80 for verification. Final retraining is accounted for separately. H=60 uses the retained fast configuration in `application/experiment_configs/`; restore its own budget settings rather than applying the H=120 schedule.

Fixed, Random Search, GA, PSO, ABC, ASHA and Hyperband are supported by `scripts/run_water_quality_suite.py`. Optuna-TPE uses `scripts/run_optuna_tpe_water_quality.py`. Configure each method's candidate budget and training schedule according to the experimental protocol in the paper.

The chronological split is 7:1:2, with normalization fitted on the training period. Key-MSE and Key-MAE aggregate N₂O, NH₄, NO₃ and PO₄; all-channel metrics cover 24 variables. Final retraining seeds are distinct from optimizer search seeds. The controlled CPU benchmarks and the forecasting application answer their respective mechanism and application questions.

## Attribution

The forecasting backbone builds on iTransformer by Liu et al. (ICLR 2024). Its retained MIT license is included as `application/LICENSE`. See `THIRD_PARTY_NOTICES.md` for attribution and dataset licensing. The repository contains research code and associated records; a repository-wide license for original HF-ABC contributions has not been added.
