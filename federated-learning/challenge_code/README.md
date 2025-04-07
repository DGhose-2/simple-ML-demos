# Coding challenge

A simple implementation of a federated learning training run for differential privacy using the Federated Averaging algorithm (FedAvg), written in PyTorch.

In the script are two classes `Orchestrator` and `Gateway`, which represent the federation server and client. The orchestrator takes a list of `Gateway` objects and runs a federated training job using them.

Each Gateway trains a local model based on a number of samples, for which toy data is generated using the `_generate_training_batch` function. The Orchestrator then gathers all the trained weights from the gateways, aggregates them and redistributes to the gateways, before starting the next federation round.

## Running instructions

### Docker

You can use the provided Dockerfile, e.g.,

```
docker build -t federated-learning-dp -f Dockerfile . && \
    docker run -v <ABSOLUTE_DIR_PATH>:<ABSOLUTE_DIR_PATH> --gpus all \
        -it federated-learning-dp:latest /bin/bash
```

where `<ABSOLUTE_DIR_PATH>` is the path to this directory (containing `Dockerfile` and the Python script), e.g. `/path/to/directory`.

This will install all dependencies (Python 3.10, PyTorch 2.4.1, and `opacus` 0.14). It requires CUDA version >= 12.1.1 be available on your machine.

### Basic requirements

I think the code should be fine in any recent version of PyTorch (>= 2.3.1) and Python (>= 3.9). The one external package that needs installing is `opacus`:

```
pip install opacus==0.14.0
```

The Docker container will handle all of this.

### Running the script for e2e runs

Once the environment is set up you can run the script

```
python fedavg.py
```

with argument `--noise-level batch` or `--noise-level sample` to use per-batch or per-sample differential privacy respectively, as I've explained below.

## Design choice discussion

### Differential privacy on per-batch or per-sample level

My initial implementation clipped the overall gradient norm of the whole batch, which an internet search showed is not how Differential Privacy algorithms implemented in production work -- instead it's necessary to compute per-sample gradients, clip each individually, and only then add noise.

I wasn't sure whether this was absolutely necessary in our case, because I figured per-batch privacy should be enough to ensure the Gateway’s contribution as a whole is protected from being exposed to other Gateways or via the aggregate model. So, I left my per-batch implementation in the final script (with argument `--noise-level batch` when running). I suppose per-sample privacy has added guarantees about minimising possibility of leaking information about specific individual samples.

With a view to including an implementation of per-sample differential privacy, I first tried to implement DP-SGD as it is described online, based on algorithms in the original paper and code examples I found, but this appeared not to reach the 80-90% accuracy quoted within the number of epochs in the boilerplate, which I didn't want to change (it did reach that accuracy after a bit longer like 2x the number of epochs). Maybe SGD was just not fast enough at converging: the same finding earlier had led me to use Adam instead. So looked into DP-Adam, which ultimately seemed a bit cumbersome to implement myself, so I instead used a third-party library [Opacus](https://github.com/pytorch/opacus) to provide the functionality. In order to fit with the boilerplate it was easiest to use an older version than the latest (0.14), but I thought for a straightforward application like implementing DP-Adam it shouldn't make much difference. This is also available in the final script (run with arg `--noise-level sample`).

### Testing for privacy

The unit-test for the differential privacy algorithm attempts to check that every gradient is changed by more than 2% of its initial value. This test sometimes fails, as I haven't tuned the privacy parameters yet (it makes sense to have a robust way to measure the privacy first). So, this test structure should be sufficient to meet the purpose of checking that noise-addition is happening and fulfilling its basic job (perturbing the gradient values and introducing "significant" changes for some definition of significance), but it doesn't offer any statistical evidence that each individual gradient has been stripped of information that could make it an outlier, which could be key for privacy. To start to quantify to what extent privacy is assured, there could be better approaches, such as potentially clustering-based outlier detection (still comparing before vs after the addition of noise).
