# src/models.py
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# Download the labels file if not already present
try:
    import requests
    response = requests.get(imagenet_labels_url)
    with open("imagenet_labels.json", "wb") as f:
        f.write(response.content)
    imagenet_labels = response.json()
except Exception as e:
    print(f"Error downloading labels: {e}")


def load_pretrained_model(model_name):
    """
    Load a pretrained model from torchvision
    
    Args:
        model_name (str): Name of the model to load ('resnet50', 'vgg16', 'mobilenet_v2')
        
    Returns:
        model: Pretrained PyTorch model in evaluation mode
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    model.eval()  # Set to evaluation mode
    return model

# ImageNet class labels
def get_imagenet_label(idx):
    """Map ImageNet class index to class name"""
    # This is a simplified version. In practice, you would load the full mapping
    labels_map = imagenet_labels[idx]
    return labels_map #labels_map.get(idx, f"Class {idx}")

def get_imagenet_index(class_name):
    # find the index of the class name in the labels
    # find a class name similar to the one provided
    for i, label in enumerate(imagenet_labels):
        if class_name.lower() in label.lower():
            return i
    print(f"Class name '{class_name}' not found in ImageNet labels.")
    return None

def predict(model, input_tensor):
    """
    Make a prediction using the model
    
    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor
        
    Returns:
        tuple: (predicted class index, class name, confidence)
    """
    with torch.no_grad():
        output = model(input_tensor)
        
    # Get the predicted class
    _, predicted_idx = torch.max(output, 1)
    
    # Calculate confidence (softmax)
    softmax = torch.nn.Softmax(dim=1)
    probabilities = softmax(output)
    confidence = probabilities[0][predicted_idx].item()
    
    # Get class name
    idx = predicted_idx.item()
    class_name = get_imagenet_label(idx)
    
    return (idx, class_name, confidence)
