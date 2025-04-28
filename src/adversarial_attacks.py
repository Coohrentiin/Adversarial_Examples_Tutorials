# src/attacks.py
import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

def fgsm_attack(model, image, target=None, epsilon=0.01, loss_fn=None):
    """
    Implements the Fast Gradient Sign Method (FGSM) for generating adversarial examples
    
    Args:
        model: PyTorch model
        image: Input tensor (should have requires_grad=True)
        target: Target class for targeted attack (None for untargeted)
        epsilon: Perturbation magnitude
        loss_fn: Loss function to use (default: CrossEntropyLoss)
        
    Returns:
        Adversarial example tensor
    """
    # Make a copy and ensure requires_grad is set
    image_copy = image.clone().detach().requires_grad_(True)
    
    # Default loss function
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    # Forward pass
    output = model(image_copy)
    
    # Calculate loss
    if target is None:
        # Untargeted attack: maximize loss for correct prediction
        # First get the correct prediction
        _, init_pred = torch.max(output, 1)
        loss = loss_fn(output, init_pred)
    else:
        # Targeted attack: minimize loss for target class
        # Convert target to torch tensor if it's not already
        if not isinstance(target, torch.Tensor):
            target = torch.tensor([target], device=image_copy.device)
        loss = -loss_fn(output, target)  # Negative sign for minimization
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Create perturbation
    perturbation = epsilon * torch.sign(image_copy.grad.data)
    
    # Create adversarial example
    adversarial_image = image_copy.detach() + perturbation
    
    # Ensure pixel values are still in valid range [0, 1]
    # adversarial_image = torch.clamp(adversarial_image, 0, 1)
    
    return adversarial_image

def pgd_attack(model, image, target=None, alpha=0.001, iterations=40, loss_fn=None):
    """
    Implements Projected Gradient Descent (PGD) attack
    
    Args:
        model: PyTorch model
        image: Input tensor
        target: Target class for targeted attack (None for untargeted)
        epsilon: Maximum perturbation
        alpha: Step size
        iterations: Number of iterations
        loss_fn: Loss function to use
        
    Returns:
        Adversarial example tensor
    """
    if loss_fn is None:
        loss_fn = torch.nn.CrossEntropyLoss()
    
    original_image = image.clone().detach()
    adversarial_image = original_image.clone().detach()
    
    for i in range(iterations):
        adversarial_image.requires_grad = True
        
        # Forward pass
        output = model(adversarial_image)
        
        # Calculate loss
        if target is None:
            # Untargeted attack
            _, current_pred = torch.max(output, 1)
            loss = loss_fn(output, current_pred)
        else:
            # Targeted attack
            if not isinstance(target, torch.Tensor):
                target = torch.tensor([target], device=adversarial_image.device)
            loss = -loss_fn(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update adversarial image
        with torch.no_grad():
            adversarial_image = adversarial_image.detach() + alpha * torch.sign(adversarial_image.grad)
            
            # # Project back to epsilon ball around original image
            # delta = torch.clamp(adversarial_image - original_image, -epsilon, epsilon)
            # adversarial_image = original_image + delta
            # adversarial_image = torch.clamp(original_image + delta, 0, 1)
    
    return adversarial_image

def deepfool_attack(model, image, num_classes=10, max_iterations=50, overshoot=0.02):
    """
    Implements the DeepFool attack
    
    Args:
        model: PyTorch model
        image: Input tensor
        num_classes: Number of classes to consider
        max_iterations: Maximum number of iterations
        overshoot: Overshoot parameter
        
    Returns:
        Adversarial example tensor
    """
    image = image.clone().detach().requires_grad_(True)
    output = model(image)
    
    # Get original prediction
    _, original_class = torch.max(output, 1)
    original_class = original_class.item()
    
    # Initialize variables
    current_class = original_class
    iteration = 0
    
    # Get number of classes from model's output if possible
    if hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
        actual_num_classes = model.fc.out_features
        num_classes = min(num_classes, actual_num_classes)
    
    while current_class == original_class and iteration < max_iterations:
        # Zero gradients
        if image.grad is not None:
            image.grad.zero_()
        
        # Get current output
        output = model(image)
        
        # Get top k classes
        sorted_outputs, indices = torch.sort(output[0])
        # Consider top 'num_classes' predictions
        indices = indices[-num_classes:]
        
        # Initialize minimal perturbation
        min_perturbation = None
        min_distance = float('inf')
        
        # Find closest hyperplane
        for k in indices:
            # Skip the original class
            if k == original_class:
                continue
            
            # Zero gradients
            model.zero_grad()
            
            # Get gradients for the current class and original class
            loss = output[0, k] - output[0, original_class]
            loss.backward(retain_graph=True)
            
            # Get gradient
            grad = image.grad.data
            
            # Calculate perturbation
            w_norm = torch.norm(grad)
            if w_norm.item() == 0:
                continue
                
            perturbation = abs(loss.item()) / w_norm.item() * grad
            
            # Check if this perturbation is smaller
            distance = torch.norm(perturbation).item()
            if distance < min_distance:
                min_distance = distance
                min_perturbation = perturbation
        
        # If no viable perturbation found, break
        if min_perturbation is None:
            break
        
        # Apply the perturbation with overshoot
        with torch.no_grad():
            image = image + (1 + overshoot) * min_perturbation
            # Ensure the image is still in valid range
            # image = torch.clamp(image, 0, 1)
        
        # Check if class has changed
        output = model(image)
        _, current_class = torch.max(output, 1)
        current_class = current_class.item()
        
        iteration += 1
    
    return image.detach()


def cw_attack(model, image, target=None, kappa=0, max_iterations=20, 
              learning_rate=0.01, binary_search_steps=9, c=1e-3,
              abort_early=True):
    """
    Implements the Carlini & Wagner (C&W) L2 attack
    
    Args:
        model: PyTorch model
        image: Input tensor (B, C, H, W)
        target: Target class for targeted attack (None for untargeted)
        kappa: Confidence margin
        max_iterations: Maximum optimization iterations
        learning_rate: Learning rate for optimizer
        binary_search_steps: Number of steps for binary search on c
        initial_const: Initial value of Confidence coefficient (trade-off constant)
        abort_early: Whether to abort early if loss hasn't decreased
        
    Returns:
        Adversarial example tensor
    """    
    # Make sure image is detached and requires_grad = False
    image = image.clone().detach()
    batch_size = image.shape[0]
    
    # Initialize binary search
    lower_bound = torch.zeros(batch_size, device=image.device)
    upper_bound = torch.ones(batch_size, device=image.device) * 1e10
    const = torch.ones(batch_size, device=image.device) * c
    
    # Get original prediction if doing untargeted attack
    if target is None:
        output = model(image)
        _, original_class = torch.max(output, 1)
    else:
        if not isinstance(target, torch.Tensor):
            target = torch.tensor([target], device=image.device)
    
    # Best attack found so far
    best_adv_images = image.clone()
    best_l2 = torch.ones(batch_size, device=image.device) * 1e10
    best_score = torch.ones(batch_size, device=image.device) * -1
    
    # Transform input to tanh-space to ensure output image is in [0, 1]
    boxmin = -2
    boxmax = 2
    boxmul = (boxmax - boxmin) / 2.0
    boxplus = (boxmin + boxmax) / 2.0
    
    # Map image from [-2, 2] to [-1, 1]
    image_tanh = (image - boxplus) / boxmul
    # image_tanh = image
    
    # Binary search for the optimal c value
    for search_step in tqdm(range(binary_search_steps)):
        # Initialize perturbation in tanh space
        modifier = torch.zeros_like(image, requires_grad=True)
        optimizer = optim.Adam([modifier], lr=learning_rate)
        
        # Track best loss for early stopping
        prev_loss = torch.ones(batch_size, device=image.device) * 1e10
        
        # Optimization loop
        for iteration in range(max_iterations):
            # Calculate adversarial image in real space [0, 1]
            adv_images_tanh = torch.tanh(modifier + image_tanh)
            adv_images = adv_images_tanh * boxmul + boxplus
            
            # Calculate L2 distance
            l2_dist = torch.sum((adv_images - image) ** 2, dim=[1, 2, 3])
            
            # Get model output
            outputs = model(adv_images)
            
            # Calculate confidence
            if target is not None:
                # Targeted attack - we want to increase confidence in target class
                # and decrease confidence in all other classes
                target_outputs = outputs.gather(1, target.unsqueeze(1)).squeeze(1)
                max_other = torch.max(outputs * (1 - F.one_hot(target, num_classes=outputs.shape[1])), dim=1)[0]
                confidence = max_other - target_outputs + kappa
            else:
                # Untargeted attack - we want to decrease confidence in original class
                original_outputs = outputs.gather(1, original_class.unsqueeze(1)).squeeze(1)
                max_other = torch.max(outputs * (1 - F.one_hot(original_class, num_classes=outputs.shape[1])), dim=1)[0]
                confidence = original_outputs - max_other + kappa
            
            # Only consider adversarial examples that fool the model
            confidence = torch.clamp(confidence, min=0)
            
            # Calculate total loss
            loss = l2_dist + const * confidence
            
            # Update with gradient descent
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            
            # Update best results
            # For targeted attack, check if target class is top prediction
            # For untargeted attack, check if original class is no longer top prediction
            if target is not None:
                pred_classes = torch.argmax(outputs, dim=1)
                succeeded = pred_classes == target
            else:
                pred_classes = torch.argmax(outputs, dim=1)
                succeeded = pred_classes != original_class
            
            for i in range(batch_size):
                if succeeded[i] and l2_dist[i] < best_l2[i]:
                    best_l2[i] = l2_dist[i]
                    best_score[i] = outputs[i, pred_classes[i]]
                    best_adv_images[i] = adv_images[i]
            
            # Early stopping
            if abort_early and iteration % (max_iterations // 10) == 0:
                if torch.all(loss >= prev_loss * 0.9999):
                    break
                prev_loss = loss
        
        # Binary search: adjust const
        for i in range(batch_size):
            if succeeded[i]:
                # Attack succeeded, try to lower c
                upper_bound[i] = const[i]
                if upper_bound[i] < 1e9:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
            else:
                # Attack failed, increase c
                lower_bound[i] = const[i]
                if upper_bound[i] < 1e9:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    const[i] *= 10
    
    return best_adv_images