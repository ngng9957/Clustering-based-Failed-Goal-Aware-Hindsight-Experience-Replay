# Clustering-based Failed goald award Hindsight experience replay (FAHER)

## Acknowledgement:
- [Openai Baselines](https://github.com/openai/baselines)
- [Reference Repository](https://github.com/TianhongDai/hindsight-experience-replay/tree/master)

## Environments
- Unbuntu 18.04
- python=3.6
- torch=1.10.1
- gym=0.17.2
- mujoco-py
- mpi4py
- sklearn

## Main Contributions
1. Clustering the achieved goals by using a cluster model which is fit to failed goals. This is implemented in [`/rl_modules/replay_buffer.py`](./rl_modules/replay_buffer.py).
2. Sampling episodes from each clustered buffer. This process is implemented in [`/her_modules/her.py`](/her_modules/her.py).

## Instructions
- run the **FetchPush-v1**:
```bash
python train.py --env-name='FetchPush-v1'

```
- run the **FetchPickAndPlace-v1**:
```bash
python train.py --env-name='FetchPickAndPlace-v1'

```
- run the **FetchSlide-v1**:
```bash
python train.py --env-name='FetchSlide-v1'

```
- run the demo (e.g. **FetchPickAndPlace**)
```bash
python demo.py --env-name='FetchPickAndPlace-v1'

```
## Plot Curves
The script to plot learning curves is provided in [`/figures/graph_hor.py`](/figures/graph_hor.py).
