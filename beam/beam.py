import torch
from slim.data import get_loaders
from slim.utils import get_layers_list
import tqdm.auto as tqdm
from .param_optim import block_wise_optimize_parameters
from slim.prune import prepare_calibration_input
from joblib import Parallel, delayed
import os
import numpy as np
import time
import wandb
import gc


def beam_recovery(
        original_model,
        compressed_model,
        tokenizer,
        model_name,
        nsamples=128,
        num_epochs=3,
        optimizer="adam",
        seed=0,
        block_granularity=1,
        wandb_log=False,
        beam_online_tune=False,
):
    """
    Prune a model using WANDA and quantize weights using SLiM-Quant or AbsMax and add low-rank adapter using SLiM or SVD.

    Args:
        original_model (torch.nn.Module): The original model to be pruned and quantized.
        compressed_model (torch.nn.Module): The compressed model to be pruned and quantized.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer used to prepare the calibration data.
        nsamples (int): The number of samples used in the calibration.
        num_epochs (int): The number of epochs to optimize the parameters.
        optimizer (str): The optimizer used in the optimization.
        seed (int): The seed used to generate the calibration data.
        block_granularity (int): The number of layers to optimize at a time.
        wandb_log (bool): Whether to log to W&B.
        beam_online_tune (bool): Whether to use online data generation for BEaM.

    Returns:
        None
    """
    global original_layer
    use_cache = compressed_model.config.use_cache
    original_model.config.use_cache = False
    compressed_model.config.use_cache = False

    compressed_model = compressed_model.cpu()

    compressed_layers = get_layers_list(compressed_model)
    original_layers = get_layers_list(original_model)

    class MaskedSequential(torch.nn.Sequential):
        def forward(self, x, **kwargs):
            for module in self._modules.values():
                y = module(x, **kwargs)[0]
                x = y
            return y

    if not beam_online_tune:
        data_dir = model_name.split("/")[-1]
        data_dir += f"-{nsamples}samples"
        np.random.seed(np.int64(time.time()))
        randint = np.random.randint(0, 1000)

        if not os.path.exists(f"tmp/{data_dir}"):
            os.makedirs(f"tmp/{data_dir}")
            dataloader, _ = get_loaders(
                "c4",
                nsamples=nsamples,
                seed=seed,
                seqlen=compressed_model.config.max_position_embeddings,
                tokenizer=tokenizer
            )
            with torch.no_grad():
                inps, outs, kwargs = prepare_calibration_input(original_model, dataloader, nsamples)

            progress_bar = tqdm.tqdm(range(len(original_layers)))
            for i in progress_bar:
                progress_bar.set_description(f"Layer {i} - Gathering data")
                original_layer = original_layers[i].cuda(0)


                inputs, outputs = [], []


                for j in range(nsamples):
                    with torch.no_grad():
                        for key in kwargs:
                            if isinstance(kwargs[key], torch.Tensor):
                                kwargs[key] = kwargs[key].cuda()
                            if isinstance(kwargs[key], tuple):
                                kwargs[key] = tuple([k.cuda() for k in kwargs[key]])
                        outs[j] = original_layer(inps[j].unsqueeze(0).cuda(), **kwargs)[0].to(outs.device)

                        inputs.append(inps[j].cpu().clone().detach())
                        outputs.append(outs[j].cpu().clone().detach())

                torch.save(inputs, f"tmp/{data_dir}/inputs{i}.pt")
                torch.save(outputs, f"tmp/{data_dir}/outputs{i}.pt")
                torch.save(kwargs, f"tmp/{data_dir}/kwargs.pt")

                inps, outs = outs, inps

                original_layers[i] = original_layer.cpu()
                del original_layer, inputs, outputs
                torch.cuda.empty_cache()
                gc.collect()
        else:
            kwargs = torch.load(f"tmp/{data_dir}/kwargs.pt")

        original_model = original_model.cpu()

        def fine_tune_layers(compressed_layers, layer_ids, device_id, model_kwargs, block_granularity=1, file_name_prepend="", data_dir=""):
            # Use progress bar with verbose only on device 0
            progress_bar = tqdm.tqdm(range(0, len(layer_ids), block_granularity), disable=device_id != 0)
            result = ""
            for list_idx in progress_bar:
                block_layer_ids = layer_ids[list_idx:list_idx + block_granularity]

                layers = MaskedSequential(*[compressed_layers[layer_id] for layer_id in block_layer_ids])
                layers = layers.cuda(device_id)
                start_layer_id = list_idx
                end_layer_id = block_layer_ids[-1]
                inputs = torch.load(f"tmp/{data_dir}/inputs{start_layer_id}.pt")
                outputs = torch.load(f"tmp/{data_dir}/outputs{end_layer_id}.pt")
                checkpoint_name = f"/tmp/{file_name_prepend}-layer{start_layer_id}-{end_layer_id}.pt"
                init_loss, final_loss = block_wise_optimize_parameters(
                    layers,
                    model_kwargs,
                    inputs,
                    outputs,
                    num_epochs=num_epochs,
                    optimizer=optimizer,
                    verbose=device_id == 0,
                    checkpoint_name=checkpoint_name
                )
                if wandb_log and device_id == 0:
                    wandb.log({
                        "layer_initial_loss": init_loss,
                        "layer_final_loss": final_loss,
                    }, step=list_idx)
                for layer_id in block_layer_ids:
                    torch.save(compressed_layers[layer_id].state_dict(), f"tmp/{file_name_prepend}-layer{layer_id}.pt")
                layers = layers.cpu()
                torch.cuda.empty_cache()
                result += (f"Layers {start_layer_id}:{end_layer_id + 1} - Initial Loss: {init_loss:.2e}, "
                           f"Final Loss: {final_loss:.2e}\n")
                del layers, inputs, outputs
                gc.collect()
            return f"Device {device_id} completed:\n{result}"

        num_gpus = torch.cuda.device_count()

        layers_per_gpu = len(compressed_layers) // num_gpus
        remainder_layers = len(compressed_layers) % num_gpus

        assignments = []
        start = 0
        for device_id in range(num_gpus):
            # Assign layers to each device
            end = start + layers_per_gpu + (1 if device_id < remainder_layers else 0)
            layer_ids = list(range(start, end))
            assignments.append((device_id, layer_ids))
            start = end

        fine_tune_layers(
            compressed_layers,
            layer_ids,
            device_id,
            kwargs,
            block_granularity,
            randint,
            data_dir,
            )

        for layer_index, layer in enumerate(compressed_layers):
            layer.load_state_dict(torch.load(f"tmp/{randint}-layer{layer_index}.pt"))
            layer = layer.to(torch.bfloat16)
            os.remove(f"tmp/{randint}-layer{layer_index}.pt")

    else:
        original_model = original_model.cpu()

        # 1. Get calibration data
        dataloader, _ = get_loaders(
            "c4",
            nsamples=nsamples,
            seed=seed,
            seqlen=compressed_model.config.max_position_embeddings,
            tokenizer=tokenizer
        )
        with torch.no_grad():
            inps, _, kwargs = prepare_calibration_input(original_model, dataloader, nsamples)

        progress_bar = tqdm.tqdm(range(0, len(compressed_layers), block_granularity))
        current_inps = inps  # This is a tensor on CPU

        for list_idx in progress_bar:
            device = 'cuda:0'  # Hardcoding for now, as it's sequential.
            end_idx = min(list_idx + block_granularity, len(compressed_layers))
            progress_bar.set_description(f"Optimizing layers {list_idx} to {end_idx - 1}")

            original_block = MaskedSequential(
                *[original_layers[i] for i in range(list_idx, end_idx)])
            compressed_block = MaskedSequential(
                *[compressed_layers[i] for i in range(list_idx, end_idx)])

            original_block.to(device)
            compressed_block.to(device)

            gpu_kwargs = {}
            for key in kwargs:
                if isinstance(kwargs[key], torch.Tensor):
                    gpu_kwargs[key] = kwargs[key].to(device)
                elif isinstance(kwargs[key], tuple):
                    gpu_kwargs[key] = tuple([k.to(device) for k in kwargs[key]])
                else:
                    gpu_kwargs[key] = kwargs[key]

            # Generate target outputs from the original model block
            target_outs = []
            with torch.no_grad():
                original_block.eval()
                for j in range(nsamples):
                    sample_inp = current_inps[j].unsqueeze(0).to(device)
                    output = original_block(sample_inp, **gpu_kwargs)
                    target_outs.append(output.cpu())
            target_outs = torch.cat(target_outs, dim=0)

            # Fine-tune the compressed block
            cpu_inps_list = [d.cpu() for d in current_inps]
            init_loss, final_loss = block_wise_optimize_parameters(
                compressed_block,
                kwargs,  # Pass original CPU kwargs
                cpu_inps_list,
                target_outs,
                num_epochs=num_epochs,
                optimizer=optimizer,
                verbose=True,
            )

            if wandb_log:
                wandb.log({
                    "layer_initial_loss": init_loss,
                    "layer_final_loss": final_loss,
                }, step=list_idx)

            # Generate input for the next block
            next_inps = []
            with torch.no_grad():
                compressed_block.eval()
                for j in range(nsamples):
                    sample_inp = current_inps[j].unsqueeze(0).to(device)
                    output = compressed_block(sample_inp, **gpu_kwargs)
                    next_inps.append(output.cpu())
            current_inps = torch.cat(next_inps, dim=0)

            # Cleanup
            original_block.cpu()
            compressed_block.cpu()
            del original_block, compressed_block, target_outs, next_inps, gpu_kwargs, cpu_inps_list
            torch.cuda.empty_cache()
            gc.collect()


    compressed_model.config.use_cache = use_cache
    compressed_model = compressed_model.cuda()
    torch.cuda.empty_cache()