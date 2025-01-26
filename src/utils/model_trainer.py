import logging
from typing import List, Dict, Any, Optional
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Manages local model training and fine-tuning."""
    
    def __init__(
        self,
        model_name: str,
        output_dir: str = "trained_models"
    ):
        """
        Initialize model trainer.
        
        Args:
            model_name: Base model name
            output_dir: Directory to save trained models
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    async def prepare_training_data(
        self,
        documents: List[Dict[str, Any]],
        search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Prepare training data from documents and search results."""
        training_data = []
        
        # Process uploaded documents
        for doc in documents:
            training_data.append({
                "input": f"Context: {doc['content']}\nQuestion: What is this document about?",
                "output": doc.get('title', 'Document summary not available')
            })
        
        # Process search results
        for result in search_results:
            if result.get('snippet'):
                training_data.append({
                    "input": f"Context: {result['snippet']}\nQuestion: What is this about?",
                    "output": result['title']
                })
        
        return training_data
    
    async def train_model(
        self,
        training_data: List[Dict[str, str]],
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the model on prepared training data.
        
        Args:
            training_data: List of input-output pairs
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimization
        """
        try:
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            model.to(self.device)
            model.train()
            
            # Prepare optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate
            )
            
            # Training loop
            for epoch in range(num_epochs):
                total_loss = 0
                
                for i in range(0, len(training_data), batch_size):
                    batch = training_data[i:i + batch_size]
                    
                    # Prepare batch
                    inputs = tokenizer(
                        [item["input"] for item in batch],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    labels = tokenizer(
                        [item["output"] for item in batch],
                        padding=True,
                        truncation=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        labels=labels["input_ids"]
                    )
                    
                    loss = outputs.loss
                    total_loss += loss.item()
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                
                logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")
            
            # Save trained model
            output_path = os.path.join(
                self.output_dir,
                f"{self.model_name}_trained_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Model saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise