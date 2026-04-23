import torch
import random
import copy
import logging


def stochastic_layer_average(old_model, curr_model, curr_alpha=0.5, layer_percentage=0.1):
    with torch.no_grad():
        # Get all layer parameters as a list for easy indexing
        layers = list(zip(old_model.parameters(), curr_model.parameters()))
        
        # Determine the number of layers to update based on layer_percentage
        num_layers = len(layers)
        num_layers_to_merge = int(num_layers * layer_percentage)
        
        # Select random layers for merging
        layer_indices = random.sample(range(num_layers), num_layers_to_merge)
        
        # Perform weighted averaging only on selected layers
        for index in layer_indices:
            old_param, curr_param = layers[index]
            curr_param.data = (1 - curr_alpha) * old_param.data + curr_alpha * curr_param.data


def stochastic_weight_average(old_model, curr_model, curr_alpha=0.5, percentage=0.1):
    with torch.no_grad():
        for old_param, curr_param in zip(old_model.parameters(), curr_model.parameters()):
            # Flatten the parameters for easier indexing
            old_data = old_param.data.view(-1)
            curr_data = curr_param.data.view(-1)
            
            # Determine the number of elements to update based on percentage
            num_elements = old_data.numel()
            num_to_merge = int(num_elements * percentage)
            # Select random indices for merging
            indices = random.sample(range(num_elements), num_to_merge)
            
            # Perform weighted averaging only on selected indices
            curr_data[indices] = (1 - curr_alpha) * old_data[indices] + curr_alpha * curr_data[indices]
            
            # Reshape back to the original parameter shape
            curr_param.data = curr_data.view_as(curr_param.data)


def weight_average(old_model, curr_model, curr_alpha=0.5):
    # In-place weighted averaging of curr_model parameters with old_model
    with torch.no_grad():
        for old_param, curr_param in zip(old_model.parameters(), curr_model.parameters()):
            curr_param.data = (1 - curr_alpha) * old_param.data + curr_alpha * curr_param.data

def ema_update(ema_model, current_model, momentum):
    with torch.no_grad():
        for ema_param, current_param in zip(ema_model.parameters(), current_model.parameters()):
            ema_param.data.mul_(momentum).add_(current_param.data, alpha=1 - momentum)


def geometric_sphere_center(w0_state, w1_state, w2_state):
    """
    Approximate the sphere center using geometric interpolation given
    a pre-trained model (w0_state) and two fine-tuned models (w1_state, w2_state).
    """
    dot_product = 0.0
    norm_w1 = 0.0
    norm_w2 = 0.0

    for key in w0_state.keys():
        # Ensure shapes match and parameters are convertible to float
        if w0_state[key].shape == w1_state[key].shape == w2_state[key].shape:
            # Compute differences and cast to float
            diff1 = (w1_state[key] - w0_state[key]).flatten().float()
            diff2 = (w2_state[key] - w0_state[key]).flatten().float()
            dot_product += torch.dot(diff1, diff2).item()
            norm_w1 += torch.sum(diff1**2).item()
            norm_w2 += torch.sum(diff2**2).item()

    cos_theta = dot_product / (math.sqrt(norm_w1 * norm_w2) + 1e-10)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    t = (2 * cos_theta) / (1 + cos_theta + 1e-10)

    center_state = {}
    for key in w0_state.keys():
        if w0_state[key].shape == w1_state[key].shape == w2_state[key].shape:
            avg_w1_w2 = (w1_state[key].float() + w2_state[key].float()) / 2.0
            center_state[key] = t * w0_state[key].float() + (1 - t) * avg_w1_w2
        else:
            center_state[key] = w0_state[key].clone()

    return center_state

import math
import torch

def geometric_sphere_center_pair(w0_state, w1_state, w2_state):
    """
    Approximate the sphere center using geometric interpolation for two fine-tuned models.
    """
    dot_product = 0.0
    norm_w1 = 0.0
    norm_w2 = 0.0

    for key in w0_state.keys():
        if w0_state[key].shape == w1_state[key].shape == w2_state[key].shape:
            diff1 = (w1_state[key] - w0_state[key]).flatten().float()
            diff2 = (w2_state[key] - w0_state[key]).flatten().float()
            dot_product += torch.dot(diff1, diff2).item()
            norm_w1 += torch.sum(diff1**2).item()
            norm_w2 += torch.sum(diff2**2).item()

    cos_theta = dot_product / (math.sqrt(norm_w1 * norm_w2) + 1e-10)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    t = (2 * cos_theta) / (1 + cos_theta + 1e-10)

    center_state = {}
    for key in w0_state.keys():
        if w0_state[key].shape == w1_state[key].shape == w2_state[key].shape:
            avg_w1_w2 = (w1_state[key].float() + w2_state[key].float()) / 2.0
            center_state[key] = t * w0_state[key].float() + (1 - t) * avg_w1_w2
        else:
            center_state[key] = w0_state[key].clone()

    return center_state

def geometric_sphere_center_multi(w0_state, *fine_tuned_states):
    """
    Iteratively approximate the sphere center using geometric interpolation 
    for more than two fine-tuned models.
    
    Args:
        w0_state: Pre-trained model state_dict.
        *fine_tuned_states: Variable number of fine-tuned model state_dicts.
        
    Returns:
        Approximate center state_dict using iterative geometric interpolation.
    """
    if len(fine_tuned_states) < 2:
        raise ValueError("At least two fine-tuned models are required.")

    # Initialize by combining the first two models
    current_center = geometric_sphere_center_pair(w0_state, fine_tuned_states[0], fine_tuned_states[1])
    
    # Iteratively merge the rest of the models with the current center
    for model_state in fine_tuned_states[2:]:
        current_center = geometric_sphere_center_pair(w0_state, current_center, model_state)
    
    return current_center


def compute_sphere_center(*models):
    """
    Compute the center of the sphere using the provided models' state_dicts.
    Align parameters based on keys and handle size mismatches gracefully.
    """
    # Extract state_dicts from models or use them directly if dicts
    state_dicts = [model if isinstance(model, dict) else model.state_dict() for model in models]

    # Initialize dictionaries to accumulate parameter values and counts
    param_accumulator = {}
    count_dict = {}

    for state_dict in state_dicts:
        for key, param in state_dict.items():
            # Initialize accumulator for new keys
            if key not in param_accumulator:
                param_accumulator[key] = torch.zeros_like(param, device=param.device)
                count_dict[key] = 0
            # Accumulate only if shapes match
            if param.shape == param_accumulator[key].shape:
                param_accumulator[key] += param
                count_dict[key] += 1

    # Average the accumulated parameters where count > 0
    averaged_params = {key: param_accumulator[key] / count_dict[key]
                       for key in param_accumulator if count_dict[key] > 0}

    return averaged_params


def compute_radius(center, model):
    """
    Compute the radius of the sphere using the center and model weights.
    Align parameters based on keys to handle size mismatches.
    """
    # Get model parameters from a state_dict if necessary
    model_params = model if isinstance(model, dict) else model.state_dict()

    flat_center = []
    flat_model = []

    for key, param in model_params.items():
        # Only consider parameters present in center with matching shape
        if key in center and param.shape == center[key].shape:
            flat_center.append(center[key].view(-1))
            flat_model.append(param.view(-1))
        elif key in center:
            logging.warning(f"Shape mismatch for key {key}: {param.shape} vs {center[key].shape}")

    if not flat_model or not flat_center:
        raise ValueError("No matching parameters found between model and center.")

    # Concatenate aligned parameters into single vectors
    flat_center = torch.cat(flat_center)
    flat_model = torch.cat(flat_model)

    # Compute radius as the Euclidean norm of the difference
    radius = torch.norm(flat_model - flat_center)
    return radius


def compute_distance(center, model):
    """
    Compute the distance of model weights from the sphere center.
    Align parameters based on keys to handle size mismatches, similarly to compute_radius.
    """
    model_params = model if isinstance(model, dict) else model.state_dict()

    flat_center = []
    flat_model = []

    for key, param in model_params.items():
        if key in center and param.shape == center[key].shape:
            flat_center.append(center[key].view(-1))
            flat_model.append(param.view(-1))
        elif key in center:
            logging.warning(f"Shape mismatch for key {key}: {param.shape} vs {center[key].shape}")

    if not flat_model or not flat_center:
        raise ValueError("No matching parameters found between model and center.")

    # Concatenate into single vectors
    flat_center = torch.cat(flat_center)
    flat_model = torch.cat(flat_model)

    # Compute distance as the Euclidean norm of the difference
    distance = torch.norm(flat_model - flat_center)
    return distance
