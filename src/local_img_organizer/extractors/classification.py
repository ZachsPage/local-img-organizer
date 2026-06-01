"""Image classification"""

import subprocess
import time
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, override

import torch
from PIL import Image
from pydantic import field_validator
from transformers import CLIPModel, CLIPProcessor

from local_img_organizer.config import parse_operations
from local_img_organizer.interfaces import Extractor, Journal, Operation
from local_img_organizer.utils import get_logger

_log = get_logger(__name__)

type ImgToClass = dict[Path, str | None]


@dataclass
class Classification(Extractor):
    """Extracts image classifications to be used in Operations"""

    class Cfg(Extractor.Cfg):
        """Classification extractor configuration"""

        categories: dict[str, list[dict[str, Any]]]
        threshold: float = 0.95
        device: Literal["cuda", "cpu"] = "cuda"
        batch_size: int = 16
        debug: bool = False

        @field_validator("categories", mode="before")
        @classmethod
        def _flatten_categories(cls, v: Any) -> Any:  # noqa: ANN401
            """YAML represents categories as a list-of-dicts; flatten to a single dict"""
            if isinstance(v, list):
                result: dict[str, Any] = {}
                for item in v:
                    if isinstance(item, dict):
                        result.update(item)
                return result
            return v

    cfg: Cfg
    categories_to_ops: dict[str, list[Operation]]

    @classmethod
    def from_cfg(cls, data: dict[str, Any]) -> "Classification":
        """Build a Classification extractor from raw YAML config data"""
        cfg = cls.Cfg.model_validate(data)
        cats_to_ops = {cat: parse_operations(op_list) for cat, op_list in cfg.categories.items()}
        return cls(cfg=cfg, categories_to_ops=cats_to_ops)

    @override
    def run(self, img_dir: Path, *, is_dry: bool) -> Generator[Callable[[], Journal.Entry]]:
        cfg = self.cfg
        _log.info(f"Loading model using {cfg.device}...")
        model, processor = _load_model(cfg.device)
        categories = list(self.categories_to_ops.keys())
        _log.info(f"Classifying images into {len(categories)} categories: {categories}...")
        start_ns = time.time_ns()
        path_to_cats = _classify_folder(
            folder=img_dir,
            labels=categories,
            model=model,
            processor=processor,
            threshold=cfg.threshold,
            batch_size=cfg.batch_size,
            device=cfg.device,
        )
        elapsed_s = (time.time_ns() - start_ns) / 1e9
        num_with_classes = len([x for x in path_to_cats.values() if x])
        _log.info(f"Classified {num_with_classes}/{len(path_to_cats)} images in {elapsed_s:.2f}s")
        if cfg.debug:
            self._debug(path_to_cats)
            return
        for path, category in path_to_cats.items():
            if category is None:
                continue
            for op in self.categories_to_ops[category]:
                yield op.prepare(
                    Operation.Data(src=path, is_dry=is_dry), ext_data={"category": category}
                )

    def _debug(self, path_to_cats: ImgToClass) -> None:
        """Interactively display images grouped by category for manual verification"""
        categorized = defaultdict(list)
        for path, category in path_to_cats.items():
            categorized[category if category else "[no match]"].append(path)
        for category in sorted(categorized.keys()):
            images = categorized[category]
            _log.info(f"\nShowing {len(images)} imgs classified as '{category}'")
            try:
                for path in images:
                    try:
                        subprocess.run(
                            ["xdg-open", str(path)],
                            check=True,
                            stderr=subprocess.DEVNULL,
                        )
                    except subprocess.CalledProcessError as e:
                        _log.error(f"  Unable to open {path}: {e}")
            except KeyboardInterrupt:
                _log.info("\n  Skipping to next category...")
                continue


def _load_model(device: str) -> tuple[CLIPModel, CLIPProcessor]:
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
    # - This disables features only needed during training like dropout (randomly zeros some values
    #   to prevent overfitting during training)
    model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=False)
    return model, processor


# TODO - would likely be cleaner as a generator to reduce memory usage
def _classify_folder(
    folder: Path,
    *,
    labels: list[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    threshold: float,
    batch_size: int,
    device: str,
) -> ImgToClass:
    """Return a dict mapping image paths (as strings) to their category (or None)

    Args:
        folder: Path to folder containing images
        labels: List of text descriptions for categories
            Tip: phrases like "a photo of a receipt" often work better than just "receipt"
        model: The loaded CLIP model
        processor: The loaded CLIP processor
        threshold: Minimum confidence to assign a label (0-1)
            - Too low: images get incorrectly categorized
            - Too high: too many images marked as None
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
            results[path] = labels[best_idx] if best_prob >= threshold else None

    return results
