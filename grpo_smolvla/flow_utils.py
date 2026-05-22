"""Custom denoising loop, prefix KV-cache computation, and group trajectory sampling."""

import torch
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks


def compute_prefix_cache(policy, obs_batch):
    """
    Run the expensive VLM prefix forward pass once and cache the KV tensors.

    policy: SmolVLAPolicy instance
    obs_batch: preprocessed observation dict (from lerobot pipeline)
    Returns dict with past_key_values and prefix_pad_masks.
    """
    model = policy.model  # VLAFlowMatching

    # Extract tensors using policy's own prepare methods (handles padding, batch dim, etc.)
    images, img_masks = policy.prepare_images(obs_batch)
    state = policy.prepare_state(obs_batch)
    lang_tokens = obs_batch["observation.language.tokens"]
    lang_masks = obs_batch["observation.language.attention_mask"]

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )

    return {
        "past_key_values": past_key_values,
        "prefix_pad_masks": prefix_pad_masks,
    }


def expand_kv(pkv, bsize):
    """
    Safely expand past_key_values to a new batch size.
    Handles nested tuples (standard Transformers format).
    """
    if pkv is None:
        return None
    
    # DEBUG: Inspect the first layer's structure
    print(f"DEBUG: pkv type: {type(pkv)}")
    if isinstance(pkv, (list, tuple)) and len(pkv) > 0:
        first_layer = pkv[0]
        print(f"DEBUG: first layer type: {type(first_layer)}")
        if isinstance(first_layer, dict):
            print(f"DEBUG: first layer keys: {first_layer.keys()}")
            for k, v in first_layer.items():
                if isinstance(v, torch.Tensor):
                    print(f"DEBUG: key '{k}' shape: {v.shape}")
        elif isinstance(first_layer, (tuple, list)):
            for i, v in enumerate(first_layer):
                if isinstance(v, torch.Tensor):
                    print(f"DEBUG: index {i} shape: {v.shape}")

    # Handle newer Transformers Cache objects (DynamicCache, etc.)
    if hasattr(pkv, "batch_repeat"):
        return pkv.batch_repeat(bsize)
    
    if isinstance(pkv, (tuple, list)):
        res = []
        for item in pkv:
            if isinstance(item, dict):
                # Handle list of dicts: [ {"key_states": t1, "value_states": t2}, ... ]
                new_dict = {}
                for k, v in item.items():
                    new_dict[k] = v.expand(bsize, *v.shape[1:]) if isinstance(v, torch.Tensor) else v
                res.append(new_dict)
            elif isinstance(item, (tuple, list)):
                # Standard format: tuple of (key_states, value_states) per layer
                res.append(tuple(
                    t.expand(bsize, *t.shape[1:]) if isinstance(t, torch.Tensor) else t
                    for t in item
                ))
            elif isinstance(item, torch.Tensor):
                res.append(item.expand(bsize, *item.shape[1:]))
            else:
                res.append(item)
        return tuple(res) if isinstance(pkv, tuple) else res
    
    return pkv


def rollout_with_n_steps(flow_model, prefix_cache, noise, num_steps):
    """
    Run Euler integration for num_steps steps from a fixed noise tensor.
    Returns action chunk of shape (B, chunk_size, action_dim).

    Bypasses policy.config.num_steps so we can request 8, 9, or 10 steps
    without mutating config state (thread-safe).
    """
    dt = -1.0 / num_steps
    x_t = noise.clone()
    bsize = noise.shape[0]
    device = noise.device

    # Expand prefix cache to match the batch size of the noise tensor
    # expanded_pad_masks: (B, seq_len)
    expanded_pad_masks = prefix_cache["prefix_pad_masks"].expand(bsize, -1)
    # expanded_past_key_values: nested tuples with batch dimension bsize
    expanded_past_key_values = expand_kv(prefix_cache["past_key_values"], bsize)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=noise.dtype, device=device).expand(bsize)
            
            # denoise_step expects prefix_pad_masks and past_key_values to match x_t batch size
            v_t = flow_model.denoise_step(
                x_t=x_t,
                prefix_pad_masks=expanded_pad_masks,
                past_key_values=expanded_past_key_values,
                timestep=time_tensor,
            )
            x_t = x_t + dt * v_t
    return x_t


def sample_group_trajectories(policy, obs_batch, n_group=8):
    """
    Sample n_group independent trajectories from the policy for a single observation
    in PARALLEL.

    Calls embed_prefix once (expensive VLM + KV-cache forward). Generates all n_group
    noise vectors as a single batch, expanding the KV cache. Runs denoising for 8, 9,
    and 10 steps on the full batch.

    Returns list of dicts, each with keys: noise, actions_8, actions_9, actions_10.
    """
    model = policy.model  # VLAFlowMatching
    action_shape = (n_group, model.config.chunk_size, model.config.max_action_dim)
    device = next(policy.parameters()).device
    model_dtype = next(policy.parameters()).dtype
    original_action_dim = policy.config.action_feature.shape[0]

    prefix_cache = compute_prefix_cache(policy, obs_batch)

    # Generate all noise vectors at once
    z_batch = torch.randn(action_shape, device=device, dtype=model_dtype)

    # Denoise all trajectories in parallel
    a8_batch  = rollout_with_n_steps(model, prefix_cache, z_batch, num_steps=8)[:, :, :original_action_dim]
    a9_batch  = rollout_with_n_steps(model, prefix_cache, z_batch, num_steps=9)[:, :, :original_action_dim]
    a10_batch = rollout_with_n_steps(model, prefix_cache, z_batch, num_steps=10)[:, :, :original_action_dim]

    # Unpack batch into individual trajectory dicts
    group = []
    for i in range(n_group):
        group.append({
            "noise": z_batch[i].unsqueeze(0),
            "actions_8": a8_batch[i].unsqueeze(0),
            "actions_9": a9_batch[i].unsqueeze(0),
            "actions_10": a10_batch[i].unsqueeze(0)
        })
    return group
