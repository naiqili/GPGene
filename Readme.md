# Stochastic Deep Gaussian Processes over Graphs
*code and results for a TCBB paper submission*


## Prerequests
our implementation is mainly based on following packages:

```
python 3.7
pip install keras==2.3.1
pip install gpuinfo
pip install tensorflow-gpu==1.15
pip install gpflow==1.5
```

Besides, some basic packages like `numpy` are also needed.

### Files

- `main.py`: Main program for static GRN inference.
- `main_ts.py`: Main program for dynamic GRN inference.
- `main_monocle_ts.py`: Main program for realistic HSMM dataset.
- `main_monocle_ts.py`, `main_monocle_ts.py`, `main_monocle_ts.py`, `main_monocle_ts.py`: Scripts for running experiments. Paths need to be reconfigured before execution.
- `draw_fig.ipynb`: Notebook for visualizing static GRN inference results.
- `demo_toy.ipynb`: Notebook for dynamic GRN inference results on the toy dataset.
- `dynamic_synthetic_analysis.ipynb`: Notebook for visualizing dynamic GRN inference results on the synthetic dataset.
- `dynamic_HSMM_analysis.ipynb`: Notebook for visualizing dynamic GRN inference results on the realistic HSMM dataset.
- `./eval/*`:  Scripts for evaluation static GRN inference results, using [GeneNetWeaver](https://github.com/tschaffter/gnw).  Paths need to be reconfigured before execution.
- `./results/synthetic/*`:  Inferred network for the synthetic dataset in each time step.
- `./results/HSMM/*`:  Inferred network for the realistic HSMM dataset in each time step.

### Parameters

- `infile`: Path of the input file.
- `outfile`: Path of the output file.
- `sizen`: Number of training instances.
- `sizem`: Number of inducing points.
- `gene`: Number of genes.
- `iter`: Steps of iterations.
- `ktype`: Kernel type, Poly1 (linear kernel) or Poly2 (degree 2 polynomial kernel).
- `lr`: Learning rate.

### Datasets

- `./data/static/*`: 5 networks and multifactorial data for static GRN experiments.
- `./data/dynamic/synthetic/*`: Multifactorial and time series data of the synthetic dataset.
- `./data/dynamic/HSMM/*`: Time series data of the HSMM dataset.