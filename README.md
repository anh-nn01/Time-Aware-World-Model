<p align="center">
   <h1 align="center">⏱️ Time-Aware World Model 🌎</h1>
   <h3 align="center"><a href="https://arxiv.org/abs/2506.08441">🎓 Paper</a> | 
   <a href="https://icml.cc/media/PosterPDFs/ICML%202025/44469.png?t=1752596515.1086242">📌 Poster</a> | 
   <a href="https://anh-nn01.github.io/time-aware-world-models/">🌐 Website</a> |
   <a href="https://icml.cc/virtual/2025/poster/44469">🎬 Videos</a></h3>
</p>

> [**Time-Aware World Model for Adaptive Prediction and Control**](https://arxiv.org/abs/2506.08441)  <br>
> [Anh N. Nhu](https://openreview.net/profile?id=%7EAnh_N_Nhu2/)<sup>\*, 1</sup>, [Sanghyun Son](https://sanghyun.phd.sh/)<sup>\*, 1</sup>, [Ming C. Lin](https://www.cs.umd.edu/~lin/)<sup>1</sup>  <br>
> <sup>1</sup> University of Maryland, College Park
> <br>
> International Conference on Machine Learning (ICML) 2025 <br>
>

## 📖 Introduction
This is the official code for Time-Aware World Model (TAWM), a model-agnostic and more efficient training method for world model. <br>

**TL;DR:** Time-Aware World Model (TAWM) is a model-agnostic training method that improves dynamics learning by explicitly incorporating time step size Δt and sampling observations at varying frequencies. This addresses the real-world constraint of varying observation rates, enabling efficient learning across temporal scales and outperforming fixed-Δt baselines under the same training budget.

<br><br>

Since TAWM's core contribution is the **time-aware concepts and training method**, which is ***architecture-agnostic***, it can be seemlessly incorporated into any world model training pipeline, including but not limited to TD-MPC2 and Dreamers. In this work, TAWM is built on top of TD-MPC2 architecture as the basis of the experiments. <br><br>

You can directly incorporate the Time-Aware concept into your world model training pipeline ***even without using our code***. To incorporate **TAWM** into any world model architecture:

1. **Modify the dynamics or temporal state-space model to condition on the time step** $\Delta t$  
   Example (Euler integration model):  
   $$z_{t+\Delta t} = z_t + d_{\theta}(z_t, a_t, \Delta t) \cdot \tau(\Delta t)$$

2. **Train using a mixture of time step sizes:**  
   $$\Delta t \sim \text{Log-Uniform}(\Delta t_{\min}, \Delta t_{\max}) \quad \text{(or Uniform sampling)}$$


## 1. Dependencies Installations

1. Installation:<br>
   **Base Conda env (recommended)**
   install Miniconda3 + dependencies:
   ```sh
   conda env create -f environment.yaml
   pip3 install gym==0.21.0
   pip3 install torch==2.3.1
   pip3 install torchvision==0.18.1
   ```
   


   **Additional dependecies for control tasks**:
   
   ```sh
   # install metaworld envs
   pip3 install git+https://github.com/Farama-Foundation/Metaworld.git@04be337a12305e393c0caf0cbf5ec7755c7c8feb
   pip3 install gymnasium

   # install controlgym envs (PDE)
   cd tdmpc2/envs
   git clone https://github.com/xiangyuan-zhang/controlgym.git
   rm -r controlgym/.git/
   ```

   **Additional dependecies for F16 aircraft control tasks** (work in progress):
   ```sh
   # source: https://github.com/GongXudong/fly-craft/
   pip install flycraft
   cd tawm/tawm/envs
   git clone https://github.com/GongXudong/fly-craft-examples.git
   ```

   ---
   <b> IMPORTANT NOTE </b>: <br>
   If for some reasons, `import controlgym` causes the following warning to TERMINATE the program (which it shouldn't):
   `UserWarning: We've integrated functorch into PyTorch. As the final step of the integration, functorch.combine_state_for_ensemble is deprecated as of PyTorch 2.0 and will be deleted in a future version of PyTorch >= 2.3.` <br>
   <b> Normally, the warning MUST NOT TERMINATE the program (just a warning) </b> <br><br>
   The following fix works for us:
   * see the error log like this: `~/miniconda3/envs/tdmpc2/lib/python3.9/site-packages/torch/_functorch/deprecated.py:38, in warn_deprecated(api, new_api)` on your machine / environment
   * execute `vim <path to _functorch/deprecated.py>` <br>
      in our case, it was `vim ~/miniconda3/envs/tdmpc2/lib/python3.9/site-packages/torch/_functorch/deprecated.py`
   * comment out line 38: `# warnings.warn(warning, stacklevel=2)`
   

## Additional To-Do List

- [ ] Potential additional experiments on F16 aircraft controls: https://github.com/GongXudong/fly-craft
- [ ] Installation guide for F16 control environments
- [ ] Implementation guide for F16 control environments

## 2. TAWM and baseline training

1. Activate conda env
   ```sh
   conda activate tawm
   ```

2. TAWM Training Examples:
   ```sh
   cd tdmpc2
   # TAWM: meta-world environments
   python train.py task=mw-basketball multi_dt=true steps=1500000 seed=3
   # TAWM: pde-control environments: no rendering/video
   python train.py task=pde-burgers save_video=false multi_dt=true steps=1500000 seed=5

   python train.py task=f16 save_video=false multi_dt=true steps=6000000 eval_freq=100000 lr=1e-3 seed=3
   ```

   ---
   Baseline Training Examples:
   ```sh
   cd tdmpc2
   # baseline: meta-world environments
   python train.py task=mw-basketball multi_dt=false steps=2000000 seed=3
   # baseline: pde-control environments: no rendering/video
   python train.py task=pde-burgers save_video=false multi_dt=false steps=2000000 seed=5
   ```

   Every 50,000 steps, the model checkpoints are saved in `<cfg.checkpoint>/<task>/<model_type>-<dt_sampler>-<integrator>/<seed>`, where:
   * `cfg.checkpoint`: saving directory of model checkpoints
   * `task`: control task (e.g. `mw-assembly`, `mw-basketball`, `pde-burgers`)
   * `model_type`: `multidt` (TAWM) or `singledt` (baseline)
   * `dt_sampler`: $\Delta t$ sampling method; `log-uniform` (default) or `uniform`
   * `integrator`: integration method; `euler` (default) or `rk4`

<!-- 5. Evaluation: <br>
   The evaluation code evalutate the performance of the world model on specified task with different simulation time steps. For `meta-world` environment, it also provides success rate for the task (success rate ranges from 0.0 to 1.0).

   Example evaluation on `mw-faucet-open` using MPC planner (better performance for some tasks):
   ```sh
   python eval_model_multidt.py task=mw-faucet-open multitask=false checkpoint=<path to .pt file>
   ```

   Example evaluation on `mw-faucet-open` using model-free planner $\pi$ (significantly faster):
   ```sh
   python eval_model_multidt.py task=mw-faucet-open multitask=false checkpoint=<path to .pt file> mpc=false
   ``` -->

## 3. Model Evaluation

<h3> 3a. Evaluation Scripts </h3>
The evaluation scripts are provided as reference script for TAWM experiment/deployments. <br>

The evaluation code evalutate the performance of the world model on specified task with different simulation time steps. For `meta-world` environment, it also provides success rate for the task (success rate ranges from 0.0 to 1.0). <br>

**NOTE: You need to update your local `model_path`**
1. `eval_model_multidt.py`: test model performance on `task` on various inference-time observation rates <br>
   ```sh
   # test TAWM (Euler + Log-Uniform) on `mw-basketball`
   python eval_model_multidt.py task=mw-basketball checkpoint={model_path} seed={seed} dt_sampler=log-uniform multi_dt=true integrator=euler
   ```

   ```sh
   # test non-time-aware baseline on `mw-basketball`
   python eval_model_multidt.py task=mw-basketball checkpoint={model_path} seed={seed} multi_dt=false eval_steps_adjusted=true
   ```
2. `eval_model_multidt_all`: comprehensively test all model performance across all tasks on various inference-time observation rates <br>
   * Models: <br>
      (1a) Baseline (non-time-aware) trained on $\Delta t_{default}$ <br>
      (1b) Baseline (non-time-aware) trained on $\Delta t \neq \Delta t_{default}$ <br>
      (2a) TAWM (RK4 + Log-Uniform Sampling) <br>
      (2b) TAWM (Euler + Log-Uniform Sampling) <br>
   * Use `eval_model_multidt.py`

3. `eval_model_learning_curve`: Evaluate intermediate models and save learning curves on each task across seeds
   * **NOTE: You need to have model saved at each step for this evaluation.** By default, a model checkpoint is saved every 50,000 steps.
<br>

<h3> 3b. Evaluation Results </h3>

The evaluation results are saved in `tdmpc2/logs/<task>/<eval-type>.csv`.
* `task`: the control task evaluated on
* `eval_type`: evaluated model type (e.g. baseline, TAWM-RK4, TAWM-Euler, etc.)

---
* **NOTE 1**: for the non-time-aware baseline models trained on fixed default $\Delta t$, we used the trained weights of the original TD-MPC2 model for each Meta-World control task.
The trained weights are available here: https://huggingface.co/nicklashansen/tdmpc2/tree/main/metaworld. <br>
* **NOTE 2**: The learning curves of the non-time-aware baseline models evaluated at the default $\Delta t$ are taken from the original TD-MPC2 model, whose learning curves (at default $\Delta t$) are publicly available at: https://github.com/nicklashansen/tdmpc2/tree/main/results/tdmpc2.

## (Optional) Experiments with MTS3
The original MTS3 is prediction-only world model and does not support evaluation on control tasks. If you are interested in experimenting with MTS3 as a comparison to our TAWM, please use our modified MTS3+MPC for comparison on control tasks. 

1. **Offline data collection** <br>
   The MTS3 model is prediction-only world model, so it does not interact with environments like `Meta-World`. Therefore, we need to collect offline dataset for it before training MTS3.

2. **Offline Data collection**: <br>
   **NOTE**: to collect data for an individual task only, specify `specific_task=<task name>`. <br>
   a. Collect offline data for **Time-Aware World Model** <br>
   ```sh
   python collect_offline_dataset.py task_set=mt9 specific_task=mw-basketball num_eps=40000 ep_length=100 multitask=false multi_dt=true data_dir=/fs/nexus-scratch/anhu/mt9_multidt_40k
   ```

   b. Collect data for **baseline world model**:
   ```sh
   cd tdmpc2/tdmpc2
   python collect_offline_dataset.py task_set=mt9 specific_task=mw-basketball num_eps=40000 ep_length=100 multitask=false multi_dt=false data_dir=/fs/nexus-scratch/anhu/mt9_singledt_40k
   ```

3. **MTS3 settings:** `H`: slow time scale factor for MTS3 
   * access config file in `MTS3/experiments/basketball/conf/model/default_mts3.yaml`
   * set `time_scale_multiplier = <H>`

4. **Example Training MTS3** for `mw-basketball` (assuming you have collected offline dataset for the task):
   ```
   cd MTS3
   python MTS3/experiments/basketball/mts3_exp.py
   ```

## 📧 Contact
If you have any question or suggestion about our work, please feel free to open an issue or contact us at anhu@umd.edu.

## 📚 Citation
If you find the insights and findings in our work useful, please consider citing our paper with the following BibTeX entry.

<h3>📄 ArXiv</h3>

```BibTeX
@misc{nhu2025timeawareworldmodeladaptive,
   title={Time-Aware World Model for Adaptive Prediction and Control}, 
   author={Anh N. Nhu and Sanghyun Son and Ming Lin},
   year={2025},
   eprint={2506.08441},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2506.08441}, 
}
```

OR 

<h3>🎓 ICML 2025</h3>

```BibTeX
@InProceedings{pmlr-v267-nhu25a,
  title = 	 {Time-Aware World Model for Adaptive Prediction and Control},
  author =       {Nhu, Anh N and Son, Sanghyun and Lin, Ming},
  booktitle = 	 {Proceedings of the 42nd International Conference on Machine Learning},
  pages = 	 {46265--46287},
  year = 	 {2025},
  editor = 	 {Singh, Aarti and Fazel, Maryam and Hsu, Daniel and Lacoste-Julien, Simon and Berkenkamp, Felix and Maharaj, Tegan and Wagstaff, Kiri and Zhu, Jerry},
  volume = 	 {267},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {13--19 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v267/main/assets/nhu25a/nhu25a.pdf},
  url = 	 {https://proceedings.mlr.press/v267/nhu25a.html},
}

```

We appreciate your interest in our work and hope that it is useful to your projects!
   

   <!-- It is very likely that when running the training process with adaptive timesteps, you will get an error like this:

   ```
   Maximum path length allowed by the benchmark has been exceeded
   ```


   This is a mujoco check that we can disable. To do this, simply go to ~/anaconda3/envs/mtrl/lib/python3.8/site-packages/metaworld/envs/mujoco/mujoco_env.py and comment the lines 107 and 108:

   ```
   if getattr(self, 'curr_path_length', 0) > self.max_path_length:
      raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
   ``` -->