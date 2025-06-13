# BEAM: Blockwise Error Minimization for One-shot Compression of LLMs

Large language models (LLMs) achieve strong performance but are notoriously expensive to deploy due to their scale. While compression techniques like pruning and quantization reduce memory and compute costs, they typically require costly retraining to recover lost accuracy. BEAM (Block-wise Error Minimization) addresses this bottleneck by splitting the model into transformer blocks and optimizing each block independently to minimize compression-induced errors. This design enables efficient fine-tuning using only a single GPU and a small amount of data, without sacrificing performance. BEAM is fully compatible with existing compression methods like Wanda, SparseGPT, GPTQ, and SLiM, and improves accuracy by up to 4.3% over compressed baselines. Despite its benefits, BEAM's runtime remains comparable to standard compression pipelines—taking under 4 hours to tune a sparse and quantized 12B model. Check out our [blog post](https://www.cs.toronto.edu/~mmozaffari/compression-trinity/beam/index.html) for more details.

<img src="./assets/beam.png" alt="BEAM" width="500">

## Setup

To clone the repository, run the following command:

```bash
git clone --recurse-submodules https://github.com/Mohammad-Mozaffari/beam.git
```

The `--recurse-submodules` flag is used to clone the [SLiM repository](https://github.com/Mohammad-Mozaffari/slim/tree/main) as a submodule. The SLiM repository is located in the `slim_local` directory.


The list of requirements can be found in the `requirements.txt` file. To install the requirements, run the following command:

```bash 
pip install -r requirements.txt
```


## Quick Start
By using SLiM's code base, we supports multiple pruning, quantization, and low-rank approximation techniques. For more details about compression the models, please refer to the [SLiM repository](https://github.com/Mohammad-Mozaffari/slim/tree/main). Below is an examples of how to use BEAM to tune a compressed LLaMA-3.2 1B model. 

**Compress Model:** We assume a compressed model is already available as the variable `compressed_model` and the tokenizer is available as the variable `tokenizer`.

**Loading Original Model:** BEAM uses the original model to recover the accuracy of the compressed model. The original model is loaded using the `get_llm` function from the SLiM repository.

```python
from slim.utils.model import get_llm

original_model, _ = get_llm(
    model_name="meta-llama/Llama-3.2-1B",
    local_files_only=True,
)
original_model = original_model.to(torch.bfloat16)
```

**BEAM Tuning:** BEAM is implemented in the `beam_recovery` function. The function takes in the original model, the compressed model, the tokenizer, and the model name. The function will tune the compressed model to recover the accuracy of the original model. More details about the `beam_recovery` function are provided in the **Function Documentation** section.

```python
from beam.beam import beam_recovery

beam_recovery(
    original_model=original_model,
    compressed_model=compressed_model,
    tokenizer=tokenizer,
    model_name="meta-llama/Llama-3.2-1B",
    nsamples=128,
)
```

**Optional Post Processing:** The capabilities used in SLiM, such as optional parameter-efficient fine-tuning and low-rank adapter quantization, are also available in BEAM. For more details about the post-processing capabilities, please refer to the [SLiM repository](https://github.com/Mohammad-Mozaffari/slim/tree/main).


**Additional Scripts:** For a more thorough example, please refer to the [scripts/run.sh](scripts/run.sh) script. Additionally, a script for SLURM-based job submission is provided in the [scripts/submit_jobs.sh](scripts/submit_jobs.sh) script.



## Experimental Results


## Function Documentation


**beam.beam.beam_recovery:** 
* `original_model`: The original model to use for tuning.
* `compressed_model`: The compressed model to tune.
* `tokenizer`: The tokenizer to use for the model.
* `model_name`: The name of the model.
* `nsamples`: The number of samples to use for tuning.
* `num_epochs`: The number of epochs to use for tuning.
* `optimizer`: The optimizer to use for tuning. Currently, the following optimizers are supported: `adam`, `sgd`, `adamw`, `adafactor`.
* `seed`: The seed to use for tuning.
* `block_granularity`: The number of layers to optimize at a time.
* `wandb_log`: Whether to log to W&B.
* `beam_online_tune`: Whether to use the output of the compressed model as the input of the next layer [`True`] or use generate all the inputs using the original model [`False`].

**beam.param_optim.block_wise_optimize_parameters:** 
* `block`: The block of the model to optimize.
* `model_kwargs`: The kwargs to pass to the model.
* `input_list`: The list of inputs to the model.
* `output_list`: The list of outputs from the model.
* `num_epochs`: The number of epochs to use for tuning.
* `compute_dtype`: The dtype to use for the model.
* `optimizer`: The optimizer to use for tuning. Currently, the following optimizers are supported: `adam`, `sgd`, `adamw`, `adafactor`.
* `verbose`: Whether to print verbose output.
* `val_set_size`: The size of the validation set.
* `checkpoint_name`: The name of the checkpoint to save the model to.



## Acknowledgement
This repository is build upon the [SLiM](https://github.com/Mohammad-Mozaffari/slim) repository.

## Citation
If you use BEAM in your research, please cite our paper:

```angular2html
@misc{mozaffari2025beam,
  author = {Mozaffari, Mohammad},
  title = {BEAM: Blockwise Error Minimization for One-shot Compression of LLMs},
  year = {2025},
  month = {June},
  day = {12},
  howpublished = {\url{https://www.cs.toronto.edu/~mmozaffari/compression-trinity/beam/index.html}},
  note = {Blog post}
}
```
