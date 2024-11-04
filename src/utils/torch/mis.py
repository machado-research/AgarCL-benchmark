def preprocess_image_observation(x):
    # Remove any extra dimensions of size 1
    x = x.squeeze()
    
    # Ensure we have 4 dimensions
    if x.dim() == 3:
        x = x.unsqueeze(0)  # Add batch dimension if it's missing
        
    elif x.dim() != 4:
        raise ValueError(f"Expected 3 or 4 dimensions after squeezing, but got shape {x.shape}")
    
    # Rearrange dimensions to [batch, channels, height, width]
    if x.shape[-1] == 3:  # If the last dimension is 3, assume it's RGB channels
        x = x.permute(0, 3, 1, 2)
    
    return x