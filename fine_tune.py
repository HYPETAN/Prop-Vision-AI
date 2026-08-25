import torch
import torch.nn as nn
import open_clip
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# 1. Check Device (Use GPU if available for faster training)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load Pre-trained CLIP Model and Preprocessor
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
model.to(device)

# Put model in training mode
model.train()

# 3. Create a Custom PyTorch Dataset for Real Estate Data
class RealEstateDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, preprocess):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = self.preprocess(item['Image'])
        caption = item['Caption']
        
        # Tokenize text
        text_tokens = self.tokenizer(caption).squeeze(0)
        return image, text_tokens

# Load a small slice of data for fine-tuning demonstration
hf_data = load_dataset("jashu66/realestate-image-dataset", split="train[:32]")
train_dataset = RealEstateDataset(hf_data, tokenizer, preprocess)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# 4. Define Optimizer (We only fine-tune the projection/vision layers lightly)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)

# 5. Training Loop (Contrastive Loss via Cosine Similarity)
print("\n--- Starting CLIP Fine-Tuning Epochs ---")
epochs = 3

for epoch in range(epochs):
    total_loss = 0.0
    for images, texts in train_loader:
        images = images.to(device)
        texts = texts.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass: Get image and text features from CLIP
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Cosine similarity matrix
        logit_scale = model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        
        # Symmetric Cross Entropy Loss (Standard CLIP Contrastive Loss)
        labels = torch.arange(len(images), device=device)
        loss_img = nn.functional.cross_entropy(logits, labels)
        loss_txt = nn.functional.cross_entropy(logits.t(), labels)
        loss = (loss_img + loss_txt) / 2.0
        
        # Backward pass & optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")

print("\nFine-tuning complete! Model weights successfully updated for domain-specific context.")

# Save the fine-tuned model state
torch.save(model.state_dict(), "clip_realestate_finetuned.pt")
print("Saved fine-tuned weights to 'clip_realestate_finetuned.pt'")