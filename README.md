Forked https://github.com/ml-jku/mgenerators-failure-modes

#  A Molecular Assays Simulator to Unravel Predictors Hacking in goal-directed molecular generations
Joseph-André Turk <sup>:fire:</sup>,
Philippe Gendreau  <sup>:fire:</sup>,
Nicolas Drizard  <sup>:fire:</sup>,
Yann Gaston-Mathé  <sup>:fire:</sup>,

<sup>:fire:</sup> [Iktos](https://www.iktos.ai)


The paper can be found here:
https://chemrxiv.org/engage/chemrxiv/article-details/62a338aabb75190ef7492fba

Feel free to send questions to philippe.gendreau@iktos.com

### Install dependencies
```
poetry install
```

### Prepare oracle

```
python create_oracle.py irak4_bayer 19 1 15
```
(arguments are: `dataset_name, seed, n_targets, power`)


### Run experiment

After having created the relevant oracle

```
python my_run_goal_directed.py --chid irak4_bayer --results_dir my_res_dir --optimizer graph_ga --model_type lr --use_train_cs 1 --target_names 'target_1targs_power15_seed19_targid0' --seed 0
```


### Special thanks
Special thanks goes out to the authors of Guacamol ([Paper](https://pubs.acs.org/doi/10.1021/acs.jcim.8b00839) / [Github](https://github.com/BenevolentAI/guacamol)), which code was used for the generative modelling/optimization part.

Special thanks goes out to the authors of  *On failure modes in molecule generation and optimization* ([Paper](https://www.sciencedirect.com/science/article/pii/S1740674920300159) / [Github](https://github.com/ml-jku/mgenerators-failure-modes)). Their code was very helpful to setup the experiments, and they were the first authors to report on the preeminent issue of failure modes of molecule generators.

Special thanks goes out to the authors of [Explaining and avoiding failure modes in goal-directed generation of small molecules](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-022-00601-y). They provided further interesting analyses and insights on the failure modes.