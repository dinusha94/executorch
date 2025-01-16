from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Load and preprocess the image
image_path = '/home/dinusha/Downloads/dog2.jpeg'  # Update with your image path
# image = Image.open(image_path).convert('L')

# Define the preprocessing transform
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((28, 28)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


image = Image.open(image_path).convert("RGB")


# image = image.resize((224, 224), Image.Resampling.LANCZOS)

# Convert the image to a numpy array and normalize pixel values to [-1, 1]
# image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
# normalized_array = (image_array - 0.5) / 0.5  # Normalize to [-1, 1]

# Flatten the numpy array
# flattened_image = normalized_array.flatten()

# Apply the transform
processed_image = transform(image).unsqueeze(0)  # Add batch dimension

# Convert the processed image to a flattened numpy array
flattened_image = processed_image.squeeze().numpy().flatten()

# Generate a C++ header file
header_content = f"""#ifndef IMAGE_DATA_H
#define IMAGE_DATA_H

const float image_data[] = {{
{', '.join([f"{value:.6f}" for value in flattened_image])}
}};

#endif // IMAGE_DATA_H
"""

# Save the header file
header_file_path = "image_data.h"  # Output header file name
with open(header_file_path, "w") as header_file:
    header_file.write(header_content)

print(f"C++ header file '{header_file_path}' generated successfully!")
