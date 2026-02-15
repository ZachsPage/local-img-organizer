"""Image classification"""

from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_model(device: str = "cuda") -> tuple[CLIPModel, CLIPProcessor]:
    """Return a tuple of (model, processor)

    :param device: "cuda" for NVIDIA GPU, "cpu" for processor (much slower)

    Notes:
    - Configures CLIP (Contrastive Language-Image Pre-training) model and processor
    - model
        The neural network itself - millions of numbers (called "weights" or "parameters")
        that were learned during training. The model contains two sub-networks:
        - A vision encoder: converts images into vectors (lists of ~768 numbers)
        - A text encoder: converts text into vectors of the same size
        When an image and text are related, their vectors point in similar directions.
    - processor
        A preprocessing pipeline that prepares raw inputs for the model. Models can't understand
        JPEGs or strings directly - they need numerical arrays in very specific formats.
        The processor handles:

        For images:
        - Resize to exactly 224x224 pixels (what this model expects)
        - Convert pixel values from 0-255 integers to 0-1 floats
        - Normalize using specific mean/std values (so inputs match training data)
        - Arrange into tensor shape [batch, channels, height, width]

        For text:
        - Tokenization: split text into subwords the model knows
          e.g., "outdoor" might become ["out", "door"]
        - Convert tokens to integer IDs from the model's vocabulary
        - Add special tokens (start/end markers the model expects)
        - Pad sequences so they're all the same length

    """
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    model.to(device)
    # Set model to evaluation mode (as opposed to training mode)
    # This disables features only needed during training like dropout
    # (dropout randomly zeros some values to prevent overfitting during training)
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor


def classify_folder(
    folder_path: str,
    labels: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    threshold: float = 0.25,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict[str, str]:
    """Return a dict mapping image paths (as strings) to their category (or None)

    Args:
        folder_path: Path to folder containing images
        labels: List of text descriptions for categories
            Tip: phrases like "a photo of a receipt" often work better than just "receipt"
        model: The loaded CLIP model
        processor: The loaded CLIP processor
        threshold: Minimum confidence to assign a label (0-1)
            - Too low: images get incorrectly categorized
            - Too high: too many images marked as None
            - Start with 0.25 and adjust based on results
        batch_size: How many images to process at once (explained below)
        device: "cuda" or "cpu"

    Notes:
    - Batching:
        Instead of processing one image at a time, we group multiple images together.
        GPUs are massively parallel - ex. multiply thousands of numbers simultaneously. Therefore,
        processing 1 image vs 16 images takes almost the same time on a GPU.
        The overhead of sending data to GPU, launching computations, etc. happens once per batch.
        Ex. No batching: 100 images = 100 round trips to GPU vs. batch of 16: 100 images = 7
        Can play with the batch size - too large & the GPU memory will run out

    """
    folder = Path(folder_path)

    # Find all common image files in the folder
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.gif", "*.bmp"]
    image_paths: list[Path] = []
    for ext in image_extensions:
        image_paths.extend(folder.glob(ext))
        image_paths.extend(folder.glob(ext.upper()))  # ex. .JPG

    if not image_paths:
        return {}

    # Pre-compute text embeddings for our provided labels
    # An embedding is a learned vector representation of something (text, image, word, etc.)
    # CLIP's text encoder converts each label into a vector of ~768 numbers
    # These numbers encode semantic meaning - similar concepts have similar vectors
    text_inputs = processor(
        text=labels,
        # "pt" = PyTorch - tells processor to return PyTorch tensors instead of plain Python lists
        return_tensors="pt",
        # User labels have different lengths after tokenization - since neural nets need fixed sized
        # inputs, add special tokens to make all sequences the same length
        padding=True,
    )

    # Move text tensors to GPU, then retrieve the embeddings to use for every image
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    # no_grad = no 'gradients' - mathematical foundation of neural network training
    # During training, we:
    # 1. Make a prediction
    # 2. Calculate how wrong it was (the "loss")
    # 3. Compute gradients: how much each weight contributed to the error
    # 4. Adjust weights in the opposite direction of their gradient
    #
    # Computing gradients requires storing intermediate values from every operation
    # (this is called the "computational graph"). This uses significant memory.
    #
    # During inference (just making predictions, not training), we don't need gradients.
    # torch.no_grad() tells PyTorch to skip building the computational graph.
    # Benefits:
    # - Uses less GPU memory (no stored intermediates)
    # - Runs faster (no gradient bookkeeping)
    # - Our model weights stay frozen (we're just using them, not updating them)
    with torch.no_grad():
        text_embeddings = model.get_text_features(**text_inputs)
        # Normalize to unit length (makes cosine similarity easier to compute later)
        # After normalization, dot product equals cosine similarity
        text_embeddings = text_embeddings / text_embeddings.norm(dim=1, keepdim=True)

    # Process images in batches
    results = {}
    for i in range(0, len(image_paths), batch_size):
        # Python slices for batching: ex. paths[0:16], paths[16:32], etc.
        batch_paths = image_paths[i : i + batch_size]

        # CLIP expects RGB (3 channels), so we convert to cover if any images are
        # grayscale (1 channel) or RGBA (4 channels)
        images = []
        img_paths = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            images.append(img)
            img_paths.append(path)

        # Preprocess the batch of images
        # No padding needed for images - they all resize to 224x224
        image_inputs = processor(images=images, return_tensors="pt")
        image_inputs = {k: v.to(device) for k, v in image_inputs.items()}

        # Skip gradients here as well - but for example, when would they be needed?
        # - Fine-tuning: adapting a pre-trained model to your specific task
        # - Training from scratch: teaching a model from random initialization
        # - Research: analyzing what the model learned
        # For our classification task, we're just using the pre-trained model as-is
        with torch.no_grad():
            # Get image embeddings for this batch
            # Shape: [batch_size, 768] - one 768-dimensional vector per image
            image_embeddings = model.get_image_features(**image_inputs)
            # Normalize to unit length
            image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

            # Compute similarity between each image and each label
            # Matrix multiply: [batch_size, 768] @ [768, num_labels] = [batch_size, num_labels]
            # Each cell [i,j] is the cosine similarity between image i and label j
            # Higher values = image and label are more related
            #
            # .T means transpose - swaps rows and columns
            # We need text_embeddings transposed for the matrix multiplication to work
            similarity = image_embeddings @ text_embeddings.T

            # Softmax converts a list of arbitrary numbers into probabilities that sum to 1.
            #
            # Formula for each element: softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
            #
            # Example:
            #   Raw scores (logits): [2.0, 1.0, 0.5]
            #   After softmax: [0.59, 0.24, 0.17] (sums to 1.0)
            #
            # "logits":
            # - "Logits" are raw, unnormalized scores from a model before applying
            #   any probability transformation. The term comes from "log-odds" in statistics.
            #   In this ex, they'd be real numbers (though cosine similarity is bounded -1 to 1).
            # - To convert logits to probabilities, we apply softmax. "logits" could loosely mean
            #   "the scores before the final activation"
            #
            # Properties:
            # - All outputs are between 0 and 1
            # - All outputs sum to 1 (valid probability distribution)
            # - Preserves ranking (highest input = highest output)
            # - Amplifies differences (the highest score gets even more probability mass)
            #
            # The "100.0 *" scaling amplifies the differences before softmax
            # Without scaling, cosine similarities (roughly -1 to 1) would give mushy probabilities
            # With scaling, we get more decisive probabilities (one label clearly wins)
            # This is sometimes called a "temperature" parameter in ML
            probs = (100.0 * similarity).softmax(dim=1)

        # Extract predictions for each image in the batch
        for j, path in enumerate(img_paths):
            # Get probabilities for this specific image
            # probs[j] is a 1D tensor with one probability per label
            image_probs = probs[j]

            # "argmax" returns the index of the maximum value, not the value itself. Ex:
            #   values = [0.1, 0.6, 0.3]
            #   max(values) = 0.6  (the maximum value)
            #   argmax(values) = 1  (the index where 0.6 lives)
            # - Use it here because we want to know which label won, not what its score was
            # - .item() converts a single-element tensor to a plain Python number
            best_idx = image_probs.argmax().item()
            best_prob = image_probs[best_idx].item()

            # Only assign a category if confidence exceeds threshold
            if best_prob >= threshold:
                results[str(path)] = labels[best_idx]
            else:  # image doesn't clearly match any category
                results[str(path)] = None

    return results
