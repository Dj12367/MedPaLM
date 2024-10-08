import torch
from medpalm.model import MedPalm
from transformers import AutoTokenizer, CLIPProcessor
from model import MedPalmTokenizer
from PIL import Image
import torch.nn.functional as F

# Assuming MedPalmTokenizer class is already defined

# Initialize the tokenizer and model
tokenizer = MedPalmTokenizer()
model = MedPalm()
img = Image.open('download.jpeg')

# Example text and image (assuming you have a loaded image as `img`)
sample = {
    "target_text": "This is a medical description involving an image.",
    "image": img,  # Your PIL image or numpy array image
}

# Tokenize the sample
tokenized_data = tokenizer.tokenize(sample)

# Extract the tokenized inputs
text_tokens = tokenized_data["text_tokens"]  # Tokenized text
image_tokens = tokenized_data["images"]  # Preprocessed image

# Define the desired shapes
desired_text_shape = (1, 4096)
desired_image_shape = (1, 3, 256, 256)

# Pad the text tokens
text_pad_amount = desired_text_shape[1] - text_tokens.shape[1]
text_tokens_padded = F.pad(text_tokens, (0, text_pad_amount), "constant", 0)

min_val = text_tokens_padded.min()
max_val = text_tokens_padded.max()
text_tokens_scaled = ((text_tokens_padded - min_val) / (max_val - min_val)) * 1999

text_tokens_scaled = text_tokens_scaled.long()

# Print the minimum and maximum of the scaled tokens to verify
print("Scaled Text Tokens Min:", text_tokens_scaled.min().item())
print("Scaled Text Tokens Max:", text_tokens_scaled.max().item())
print("Scaled Text Tokens Shape:", text_tokens_scaled.shape)

# Pad the image tokens
image_pad_height = desired_image_shape[2] - image_tokens.shape[2]
image_pad_width = desired_image_shape[3] - image_tokens.shape[3]
image_tokens_padded = F.pad(image_tokens, (0, image_pad_width, 0, image_pad_height), "constant", 0)

print("Image Tokens Shape:", image_tokens_padded.shape)

# Perform forward pass with the model
with torch.no_grad():  # If you're just inference, no need to compute gradients
    output_logits = model(image_tokens_padded, text_tokens_scaled)

print(output_logits)
print(output_logits[0].shape)
print(output_logits[1].shape)


# Get the predicted token ids from the output logits (argmax for highest probability tokens)
predicted_token_ids = torch.argmax(output_logits[0], dim=-1)

# Convert the token ids back into text using the tokenizer
generated_text = tokenizer.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

# Print the generated text
print("Generated text:", generated_text)
