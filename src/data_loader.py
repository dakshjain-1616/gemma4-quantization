#!/usr/bin/env python3
"""
Data loader for WikiText-2 benchmark dataset.
Downloads and preprocesses WikiText-2 for quantization evaluation.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Tuple
import torch
from huggingface_hub import login

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_wikitext2(num_samples: int = 1000, max_length: int = 512) -> Tuple[list, list]:
    """
    Load WikiText-2 dataset for benchmark evaluation.
    
    Args:
        num_samples: Number of samples to load
        max_length: Maximum sequence length
    
    Returns:
        texts: List of text samples
        encodings: List of tokenized encodings
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    logger.info(f"Loading WikiText-2 dataset ({num_samples} samples)...")
    
    # Load WikiText-2 from HuggingFace (use specific config)
    try:
        dataset = load_dataset(
            "wikitext",
            "wikitext-2-raw-v1",
            split="test"
        )
        logger.info(f"Dataset loaded: {len(dataset)} samples available")
    except Exception as e:
        logger.warning(f"Failed to load WikiText-2: {e}")
        logger.info("Using fallback text samples...")
        # Fallback: use simple text samples
        fallback_texts = [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "Machine learning is a subset of artificial intelligence. " * 10,
            "Natural language processing enables computers to understand human language. " * 10,
        ]
        dataset = {"text": fallback_texts * (num_samples // 3 + 1)}
    
    # Get samples
    texts = []
    for i in range(min(num_samples, len(dataset))):
        text = dataset[i]['text']
        texts.append(str(text)[:max_length * 2])  # Get extra for tokenization
    
    logger.info(f"Loaded {len(texts)} text samples")
    
    # Load the Gemma-4 tokenizer to match the model under study
    logger.info("Loading google/gemma-4-E2B-it tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "google/gemma-4-E2B-it",
        token=HF_TOKEN,
    )
    
    # Tokenize with consistent length
    encodings = []
    for text in texts:
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding='max_length'  # Ensure all sequences are exactly max_length
        )
        encodings.append(encoding)
    
    logger.info(f"Tokenized {len(encodings)} samples")
    
    return texts, encodings


def create_dataloader(texts: list, encodings: list, 
                     batch_size: int = 8, 
                     max_length: int = 512) -> torch.utils.data.DataLoader:
    """
    Create a DataLoader for benchmark evaluation.
    
    Args:
        texts: List of text samples
        encodings: List of tokenized encodings
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
    
    Returns:
        DataLoader for benchmark evaluation
    """
    from torch.utils.data import Dataset, DataLoader
    
    class TextDataset(Dataset):
        def __init__(self, texts, encodings, max_length):
            self.texts = texts
            self.encodings = encodings
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            encoding = self.encodings[idx]
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'text': self.texts[idx]
            }
    
    dataset = TextDataset(texts, encodings, max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True
    )
    
    logger.info(f"Created DataLoader with {len(dataset)} samples, batch_size={batch_size}")
    
    return dataloader


def verify_data_loader(dataloader: torch.utils.data.DataLoader, 
                       num_batches: int = 5) -> bool:
    """
    Verify that the data loader works correctly.
    
    Args:
        dataloader: DataLoader to verify
        num_batches: Number of batches to test
    
    Returns:
        True if verification passed
    """
    logger.info(f"Verifying data loader ({num_batches} batches)...")
    
    try:
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            # Check batch structure
            assert 'input_ids' in batch, "Missing input_ids"
            assert 'attention_mask' in batch, "Missing attention_mask"
            assert 'text' in batch, "Missing text"
            
            # Check tensor shapes
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            assert input_ids.dim() == 2, f"input_ids should be 2D, got {input_ids.dim()}"
            assert attention_mask.dim() == 2, f"attention_mask should be 2D, got {attention_mask.dim()}"
            assert input_ids.shape == attention_mask.shape, "Shape mismatch"
            
            # Check for valid tokens
            assert input_ids.min() >= 0, "Negative token IDs found"
            assert attention_mask.min() >= 0, "Negative attention mask values"
            
            logger.info(f"Batch {i+1}: shape={input_ids.shape}, min_token={input_ids.min()}, max_token={input_ids.max()}")
        
        logger.info("✓ Data loader verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data loader verification failed: {e}")
        return False


def main():
    """Main entry point for data preparation"""
    logger.info("=" * 60)
    logger.info("WikiText-2 Data Preparation")
    logger.info("=" * 60)
    
    # Load data
    texts, encodings = load_wikitext2(num_samples=100, max_length=512)
    
    # Create dataloader
    dataloader = create_dataloader(texts, encodings, batch_size=8, max_length=512)
    
    # Verify
    success = verify_data_loader(dataloader, num_batches=5)
    
    if success:
        logger.info("\n✓ Data preparation completed successfully!")
        logger.info(f"  - Samples: {len(texts)}")
        logger.info(f"  - Batch size: 8")
        logger.info(f"  - Max length: 512")
    else:
        logger.error("\n✗ Data preparation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()