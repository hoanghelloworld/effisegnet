import os
import torch
import cv2
import numpy as np
from pathlib import Path
import argparse
from models.effisegnet import EffiSegNetBN
from datamodule import CustomSegDataset
from network_module import Net
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_model(checkpoint_path, model_config):
    """Load trained model from checkpoint"""
    model = EffiSegNetBN(**model_config)
    net = Net.load_from_checkpoint(checkpoint_path, model=model)
    net.eval()
    return net


def predict_single_image(model, image_path, img_size=(384, 384)):
    """Predict mask for a single image"""
    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(*img_size, interpolation=cv2.INTER_LANCZOS4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed["image"].unsqueeze(0)
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        model = model.cuda()
    
    with torch.no_grad():
        if hasattr(model.model, 'deep_supervision') and model.model.deep_supervision:
            logits, _ = model(image_tensor)
        else:
            logits = model(image_tensor)
        
        prob = torch.sigmoid(logits)
        mask = (prob > 0.5).float()
        
    return mask[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict masks for test images")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--test_dir", required=True, help="Directory containing test images")
    parser.add_argument("--output_dir", default="./predictions", help="Output directory for predictions")
    parser.add_argument("--img_size", nargs=2, type=int, default=[384, 384], help="Image size for model input")
    
    args = parser.parse_args()
    
    # Model configuration
    model_config = {
        'ch': 64,
        'pretrained': False,  # Not needed for inference
        'freeze_encoder': False,
        'deep_supervision': False,
        'model_name': 'efficientnet_v2_s'
    }
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, model_config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all test images
    test_images = list(Path(args.test_dir).glob("*.jpg"))
    
    print(f"Found {len(test_images)} test images")
    
    # Process each image
    for img_path in test_images:
        print(f"Processing: {img_path.name}")
        
        # Predict mask
        mask = predict_single_image(model, str(img_path), tuple(args.img_size))
        
        # Convert to 0-255 range
        mask = (mask * 255).astype(np.uint8)
        
        # Save prediction
        output_path = Path(args.output_dir) / f"{img_path.stem}_pred.png"
        cv2.imwrite(str(output_path), mask)
    
    print(f"Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
