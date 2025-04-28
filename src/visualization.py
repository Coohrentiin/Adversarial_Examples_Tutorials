# src/visualization.py
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image

def visualize_perturbation(original_tensor, adversarial_tensor, scale=10):
    """
    Visualize the perturbation between original and adversarial images
    
    Args:
        original_tensor (torch.Tensor): Original image tensor
        adversarial_tensor (torch.Tensor): Adversarial image tensor
        scale (int): Scale factor to enhance the perturbation visibility
        
    Returns:
        tuple: (perturbation_image, enhanced_perturbation_image)
    """
    # Convert tensors to numpy arrays
    if original_tensor.dim() == 4:
        original_tensor = original_tensor.squeeze(0)
    if adversarial_tensor.dim() == 4:
        adversarial_tensor = adversarial_tensor.squeeze(0)
    
    original_np = original_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    adversarial_np = adversarial_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Calculate perturbation
    perturbation = adversarial_np - original_np
    
    # Create enhanced perturbation for better visibility
    perturbation_enhanced = perturbation * scale + 0.5
    perturbation_enhanced = np.clip(perturbation_enhanced, 0, 1)
    
    return perturbation, perturbation_enhanced

def plot_attack_results(original_image, adversarial_image, perturbation_enhanced, 
                      original_pred, adversarial_pred, title=None):
    """
    Plot the original image, adversarial image, and perturbation
    
    Args:
        original_image (PIL.Image): Original image
        adversarial_image (PIL.Image): Adversarial image
        perturbation_enhanced (numpy.ndarray): Enhanced perturbation for visualization
        original_pred (tuple): Original prediction (class_idx, class_name, confidence)
        adversarial_pred (tuple): Adversarial prediction (class_idx, class_name, confidence)
        title (str, optional): Super title for the plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original: {original_pred[1]}\nConfidence: {original_pred[2]:.2%}")
    axes[0].axis('off')
    
    # Adversarial image
    axes[1].imshow(adversarial_image)
    axes[1].set_title(f"Adversarial: {adversarial_pred[1]}\nConfidence: {adversarial_pred[2]:.2%}")
    axes[1].axis('off')
    
    # Perturbation
    axes[2].imshow(perturbation_enhanced)
    axes[2].set_title("Perturbation (Enhanced)")
    axes[2].axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

