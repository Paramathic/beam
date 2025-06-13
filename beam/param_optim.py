import torch
import numpy as np
from tqdm import tqdm
import transformers
from types import MethodType
from slim.quantization.quantization import QuantizedMatmul
import os



def quantized_linear(module, input):
    """
    This function is used to perform the quantized matmul operation.
    Args:
        module: The module to perform the operation on.
        input: The input to the module.
    """
    output = QuantizedMatmul.apply(input, module.weight.t(), module.quantizer)
    if module.bias is not None:
        output += module.bias
    return output


def get_optimizer(optimizer, params, lr):
    """
    This function is used to get the optimizer for the model.
    Args:
        optimizer: The optimizer to use for the model.
        params: The parameters to optimize.
        lr: The learning rate to use for the optimizer.
    """
    if optimizer == "adam":
        optimizer = torch.optim.Adam(params, lr=lr)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(params, lr=lr)
    elif optimizer == "adamw":
        optimizer = torch.optim.AdamW(params, lr=lr)
    elif optimizer == "adafactor":
        optimizer = transformers.Adafactor(params, lr=lr, relative_step=False)
    else:
        raise NotImplementedError(f"Optimizer {optimizer} not implemented")
    return optimizer


def block_wise_optimize_parameters(
        block,
        model_kwargs,
        input_list,
        output_list,
        num_epochs=3,
        compute_dtype=torch.bfloat16,
        optimizer="adam",
        verbose=True,
        val_set_size=32,
        checkpoint_name="/tmp/checkpoint.pt"
):
    """
    This function is used to optimize the parameters of a block of the model.
    Args:
        block: The block of the model to optimize.
        model_kwargs: The kwargs to pass to the model.
        input_list: The list of inputs to the model.
        output_list: The list of outputs from the model.
        num_epochs: The number of epochs to train for.
        compute_dtype: The dtype to use for the model.
        optimizer: The optimizer to use for the model.
        verbose: Whether to print verbose output.
        val_set_size: The size of the validation set.
        checkpoint_name: The name of the checkpoint to save the model to.
    """
    masks = {}
    for param in block.parameters():
        param.requires_grad = True
        masks[param] = param.data == 0

    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(quantized_linear, module)
            if not hasattr(module, "quantizer"):
                module.quantizer = None

    dtype = param.dtype
    device = param.device
    block = block.to(compute_dtype)

    metric = torch.nn.MSELoss()

    torch.save(block.state_dict(), checkpoint_name)

    for key in model_kwargs:
        if isinstance(model_kwargs[key], torch.Tensor):
            model_kwargs[key] = model_kwargs[key].to(device)
        elif isinstance(model_kwargs[key], tuple):
            model_kwargs[key] = tuple([k.to(device) for k in model_kwargs[key]])

    with torch.set_grad_enabled(True):
        print("Searching for the best learning rate.")
        average_losses = []
        lr_list = [1e-3, 5e-4, 1e-4, 5e-5, 1e-6]
        for lr in lr_list:
            # block_copy = copy.deepcopy(block)
            block_copy = block
            block_copy.load_state_dict(torch.load(checkpoint_name))
            mask_search_optimizer = get_optimizer(optimizer, block_copy.parameters(), lr)
            mask_search_scheduler = torch.optim.lr_scheduler.LinearLR(mask_search_optimizer,
                                                     start_factor=1.0,
                                                     end_factor=1e-2,
                                                     total_iters=num_epochs * len(input_list)) # We use the exact same scheduler as in the actual training
            lr_search_masks = {}
            for param in block_copy.parameters():
                param.requires_grad = True
                lr_search_masks[param] = param.data == 0
            losses = []
            for input, output in zip(input_list[:val_set_size], output_list[:val_set_size]):
                input = input.to(device).to(compute_dtype)
                output = output.to(device).to(compute_dtype)
                y = block_copy(input.unsqueeze(0), **model_kwargs)[0].squeeze(0)
                loss = metric(y, output)
                loss.backward()
                mask_search_optimizer.step()
                mask_search_scheduler.step()
                mask_search_optimizer.zero_grad()
                for param in block_copy.parameters():
                    param.data[lr_search_masks[param]] = 0
                losses.append(loss.item())
            average_loss = np.mean(losses[-val_set_size//2:])
            average_losses.append(average_loss)
        lr = lr_list[np.argmin(average_losses)]
        del block_copy, losses, average_losses, lr_search_masks, mask_search_optimizer

    block.load_state_dict(torch.load(checkpoint_name))
    os.remove(checkpoint_name)

    params = block.parameters()
    optimizer = get_optimizer(optimizer, params, lr)

    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,
                                                     start_factor=1.0,
                                                     end_factor=1e-2,
                                                     total_iters=num_epochs * len(input_list))

    progress_bar = tqdm(range(num_epochs * len(input_list)), disable=not verbose)
    losses = []
    init = True
    with torch.set_grad_enabled(True):
        for epoch in range(num_epochs):
            for input, output in zip(input_list, output_list):
                input = input.to(device).to(compute_dtype)
                output = output.to(device).to(compute_dtype)
                y = block(input.unsqueeze(0), **model_kwargs)[0].squeeze(0)
                loss = metric(y, output)
                norm = metric(y, torch.zeros_like(y))
                if init:
                    init_loss = loss.item() / norm.item()
                    init = False
                loss.backward()
                losses.append(loss.item()  / norm.item())
                average_loss = np.mean(losses[-100:])
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                for param in block.parameters():
                    param.data[masks[param]] = 0
                progress_bar.update(1)
                progress_bar.set_postfix({'loss': average_loss, "lr": lr})
    if verbose:
        print(f"Initial Loss: {init_loss:.2e} - Final Loss: {average_loss:.2e}")

    block = block.to(dtype)
    block = block.eval()

    for name, module in block.named_modules():
        if isinstance(module, torch.nn.Linear):
            if module.quantizer is not None:
                module.weight.data = module.quantizer.dequantize_absmax(
                    module.quantizer.quantize_weight(module.weight.data)
                )
            module.quantizer = None

    return init_loss, average_loss