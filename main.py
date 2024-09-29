import torch
from medpalm.model import MedPalm
from transformers import AutoTokenizer, CLIPProcessor
from model import MedPalmTokenizer
from PIL import Image

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

# Print shapes
print("Text Tokens Shape:", text_tokens.shape)
print("Image Tokens Shape:", image_tokens.shape)

# Perform forward pass with the model
with torch.no_grad():  # If you're just inference, no need to compute gradients
    output_logits = model(image_tokens, text_tokens)

# Get the predicted token ids from the output logits (argmax for highest probability tokens)
predicted_token_ids = torch.argmax(output_logits, dim=-1)

# Convert the token ids back into text using the tokenizer
generated_text = tokenizer.tokenizer.batch_decode(predicted_token_ids, skip_special_tokens=True)

# Print the generated text
print("Generated text:", generated_text)
