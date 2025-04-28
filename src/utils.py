# src/utils.py
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

def load_image(image_path):
    """
    Load an image from path
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        PIL.Image: Loaded image
    """
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image

def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image): Input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Add batch dimension
    input_tensor = preprocess(image).unsqueeze(0)
    return input_tensor

def deprocess_tensor(tensor):
    """
    Convert tensor back to PIL Image
    
    Args:
        tensor (torch.Tensor): Input tensor
        
    Returns:
        PIL.Image: Converted image
    """
    # Remove batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Undo normalization
    tensor = tensor.clone().detach()
    tensor = tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    
    # Clip to [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor)
    
    return image

def show_image(image, title=None):
    """
    Display an image
    
    Args:
        image (PIL.Image): Image to display
        title (str, optional): Title for the image
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def show_comparison(original_image, adversarial_image, perturbation=None, 
                    original_pred=None, adversarial_pred=None):
    """
    Display original and adversarial images side by side
    
    Args:
        original_image (PIL.Image): Original image
        adversarial_image (PIL.Image): Adversarial image
        perturbation (numpy.ndarray, optional): Perturbation to display
        original_pred (tuple, optional): Original prediction (class_idx, class_name, confidence)
        adversarial_pred (tuple, optional): Adversarial prediction (class_idx, class_name, confidence)
    """
    fig, axes = plt.subplots(1, 3 if perturbation is not None else 2, figsize=(18, 6))
    
    # Display original image
    axes[0].imshow(original_image)
    title = "Original Image"
    if original_pred:
        title += f"\nClass: {original_pred[1]} ({original_pred[2]:.2%})"
    axes[0].set_title(title)
    axes[0].axis('off')
    
    # Display adversarial image
    axes[1].imshow(adversarial_image)
    title = "Adversarial Image"
    if adversarial_pred:
        title += f"\nClass: {adversarial_pred[1]} ({adversarial_pred[2]:.2%})"
    axes[1].set_title(title)
    axes[1].axis('off')
    
    # Display perturbation if provided
    if perturbation is not None:
        # Enhance perturbation for visibility
        pert_display = perturbation * 10 + 0.5
        pert_display = np.clip(pert_display, 0, 1)
        
        axes[2].imshow(pert_display)
        axes[2].set_title("Perturbation (Enhanced)")
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def calculate_perturbation(original_tensor, adversarial_tensor):
    """
    Calculate the perturbation between original and adversarial images
    
    Args:
        original_tensor (torch.Tensor): Original image tensor
        adversarial_tensor (torch.Tensor): Adversarial image tensor
        
    Returns:
        numpy.ndarray: Perturbation as numpy array
    """
    # Convert tensors to numpy
    if original_tensor.dim() == 4:
        original_tensor = original_tensor.squeeze(0)
    if adversarial_tensor.dim() == 4:
        adversarial_tensor = adversarial_tensor.squeeze(0)
    
    original_np = original_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    adversarial_np = adversarial_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    
    # Calculate perturbation
    perturbation = adversarial_np - original_np
    
    return perturbation

def save_adversarial_example(original_image, adversarial_image, perturbation, 
                            original_pred, adversarial_pred, output_path):
    """
    Save adversarial example visualization to file
    
    Args:
        original_image (PIL.Image): Original image
        adversarial_image (PIL.Image): Adversarial image
        perturbation (numpy.ndarray): Perturbation
        original_pred (tuple): Original prediction (class_idx, class_name, confidence)
        adversarial_pred (tuple): Adversarial prediction (class_idx, class_name, confidence)
        output_path (str): Path to save the visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Display original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original: {original_pred[1]} ({original_pred[2]:.2%})")
    axes[0].axis('off')
    
    # Display adversarial image
    axes[1].imshow(adversarial_image)
    axes[1].set_title(f"Adversarial: {adversarial_pred[1]} ({adversarial_pred[2]:.2%})")
    axes[1].axis('off')
    
    # Display perturbation
    pert_display = perturbation * 10 + 0.5
    pert_display = np.clip(pert_display, 0, 1)
    axes[2].imshow(pert_display)
    axes[2].set_title("Perturbation (Enhanced 10x)")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def compute_metrics(model, original_images, adversarial_images):
    """
    Compute metrics for evaluating adversarial attacks
    
    Args:
        model: PyTorch model
        original_images (list): List of original image tensors
        adversarial_images (list): List of adversarial image tensors
        
    Returns:
        dict: Dictionary containing metrics
    """
    metrics = {
        'success_rate': 0,
        'avg_confidence_drop': 0,
        'avg_perturbation_l2': 0
    }
    
    success_count = 0
    confidence_drops = []
    perturbation_l2_norms = []
    
    for original, adversarial in zip(original_images, adversarial_images):
        # Get predictions
        with torch.no_grad():
            original_output = model(original)
            adversarial_output = model(adversarial)
        
        # Apply softmax to get probabilities
        softmax = torch.nn.Softmax(dim=1)
        original_probs = softmax(original_output)
        adversarial_probs = softmax(adversarial_output)
        
        # Get predicted classes
        _, original_class = torch.max(original_output, 1)
        _, adversarial_class = torch.max(adversarial_output, 1)
        
        # Check if attack was successful
        if original_class != adversarial_class:
            success_count += 1
        
        # Calculate confidence drop
        original_confidence = original_probs[0, original_class].item()
        adversarial_confidence = adversarial_probs[0, original_class].item()
        confidence_drop = original_confidence - adversarial_confidence
        confidence_drops.append(confidence_drop)
        
        # Calculate L2 norm of perturbation
        perturbation = adversarial - original
        l2_norm = torch.norm(perturbation, p=2).item()
        perturbation_l2_norms.append(l2_norm)
    
    # Calculate metrics
    metrics['success_rate'] = success_count / len(original_images)
    metrics['avg_confidence_drop'] = sum(confidence_drops) / len(confidence_drops)
    metrics['avg_perturbation_l2'] = sum(perturbation_l2_norms) / len(perturbation_l2_norms)
    
    return metrics
