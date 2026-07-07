#!/usr/bin/env python3
"""
Script for training a model using SFT on conversation data.
Supports multi-turn conversations in the format:
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    ...
]
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb


@dataclass
class ModelConfig:
    """Model configuration."""
    model_name: str
    tokenizer_name: Optional[str] = None
    use_flash_attention: bool = False


@dataclass
class DataConfig:
    """Data configuration."""
    data_path: str
    val_data_path: Optional[str] = None
    max_length: int = 2048


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./outputs/sft_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    bf16: bool = False
    fp16: bool = False
    gradient_checkpointing: bool = False
    seed: int = 42


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None


@dataclass
class HuggingFaceHubConfig:
    """HuggingFace Hub configuration."""
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None  # e.g., "username/model-name"
    hub_private: bool = False
    merge_adapters: bool = True  # Whether to merge LoRA adapters before pushing


@dataclass
class Config:
    """Main configuration."""
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    lora: LoRAConfig
    wandb: WandbConfig
    hf_hub: HuggingFaceHubConfig


def load_conversation_data(data_path: str, split: Optional[str] = None) -> List[List[Dict[str, str]]]:
    """Load conversation data from JSONL file or Hugging Face dataset.
    
    Args:
        data_path: Path to local JSONL file or Hugging Face dataset repo (e.g., "user/dataset")
        split: Dataset split to load (e.g., "train", "test"). Only used for HF datasets.
    
    Returns:
        List of conversations, where each conversation is a list of message dicts
    """
    conversations = []
    
    # Check if it's a HuggingFace dataset (contains '/' and doesn't end with .jsonl/.json)
    is_hf_dataset = '/' in data_path and not data_path.endswith(('.jsonl', '.json'))
    
    if is_hf_dataset:
        print(f"Loading from Hugging Face dataset: {data_path}, split: {split}")
        dataset = load_dataset(data_path, split=split)
        
        for item in dataset:
            # Handle different formats
            if 'messages' in item:
                conversations.append(item['messages'])
            elif 'conversation' in item:
                conversations.append(item['conversation'])
            elif isinstance(item, dict):
                # If it's a single message dict, wrap it in a list
                conversations.append([item])
            else:
                conversations.append(item)
    else:
        # Load from local JSONL file
        print(f"Loading from local file: {data_path}")
        with open(data_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    # Handle both direct list format and dict with 'messages' key
                    if isinstance(data, list):
                        conversations.append(data)
                    elif isinstance(data, dict) and 'messages' in data:
                        conversations.append(data['messages'])
                    elif isinstance(data, dict) and 'conversation' in data:
                        conversations.append(data['conversation'])
                    else:
                        # Assume it's a dict with role/content keys
                        conversations.append([data])
    
    return conversations




def format_conversation(conversation: List[Dict[str, str]], tokenizer) -> str:
    """Format conversation using the tokenizer's chat template."""
    # Use the tokenizer's chat template if available
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        formatted = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback to simple formatting
        formatted = ""
        for message in conversation:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'system':
                formatted += f"System: {content}\n\n"
            elif role == 'user':
                formatted += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n\n"
        formatted += tokenizer.eos_token if tokenizer.eos_token else "</s>"
    
    return formatted


def create_dataset(conversations: List[List[Dict[str, str]]]) -> Dataset:
    """Create a dataset from conversations."""
    # SFTTrainer expects 'messages' field with conversation format
    dataset = Dataset.from_dict({'messages': conversations})
    return dataset


def setup_model_and_tokenizer(cfg: Config):
    """Setup model and tokenizer."""
    # Load tokenizer
    tokenizer_name = cfg.model.tokenizer_name if cfg.model.tokenizer_name else cfg.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if cfg.training.bf16 else (torch.float16 if cfg.training.fp16 else torch.float32),
    }
    
    if cfg.model.use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name,
        **model_kwargs
    )
    
    # Setup LoRA if specified
    if cfg.lora.use_lora:
        print("Setting up LoRA...")
        model = prepare_model_for_kbit_training(model)
        
        target_modules = OmegaConf.to_container(
            cfg.lora.lora_target_modules, resolve=True
        )

        lora_config = LoraConfig(
            r=int(cfg.lora.lora_r),
            lora_alpha=int(cfg.lora.lora_alpha),
            target_modules=target_modules,
            lora_dropout=float(cfg.lora.lora_dropout),
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    if cfg.training.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model, tokenizer


@hydra.main(version_base=None, config_path="configs", config_name="train_sft")
def main(cfg: DictConfig):
    # Convert to structured config
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    set_seed(cfg.training.seed)
    
    # Initialize wandb if specified
    if cfg.wandb.wandb_project:
        wandb.init(
            project=cfg.wandb.wandb_project,
            name=cfg.wandb.wandb_run_name,
            config=OmegaConf.to_container(cfg, resolve=True)
        )
    
    print(f"Loading model and tokenizer from {cfg.model.model_name}...")
    model, tokenizer = setup_model_and_tokenizer(cfg)
    
    print(f"Loading training data from {cfg.data.data_path}...")
    train_conversations = load_conversation_data(cfg.data.data_path, split=cfg.data.split)
    print(f"Loaded {len(train_conversations)} training conversations")
    
    train_dataset = create_dataset(train_conversations)#.select(range(8))
    
    # Load validation data if provided
    eval_dataset = None
    if cfg.data.val_data_path:
        print(f"Loading validation data from {cfg.data.val_data_path}...")
        val_conversations = load_conversation_data(cfg.data.val_data_path)
        print(f"Loaded {len(val_conversations)} validation conversations")
        eval_dataset = create_dataset(val_conversations)
    
    # Setup training arguments with SFTConfig
    training_args = SFTConfig(
        output_dir=cfg.training.output_dir,
        num_train_epochs=cfg.training.num_train_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        save_steps=cfg.training.save_steps,
        eval_steps=cfg.training.eval_steps if eval_dataset else None,
        save_strategy="epoch", # if eval_dataset else "no",
        save_total_limit=cfg.training.save_total_limit,
        bf16=cfg.training.bf16,
        fp16=cfg.training.fp16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to="wandb" if cfg.wandb.wandb_project else "none",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        greater_is_better=False,
        save_safetensors=True,
        seed=cfg.training.seed,
        assistant_only_loss=True
    )
    
    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {cfg.training.output_dir}/final")
    trainer.save_model(f"{cfg.training.output_dir}/final")
    tokenizer.save_pretrained(f"{cfg.training.output_dir}/final")
    
    print("Training complete!")
    
    # Merge adapter weights if LoRA was used and push model to HuggingFace Hub
    if cfg.lora.use_lora and cfg.hf_hub.push_to_hub:
        if cfg.hf_hub.merge_adapters:
            print("Merging LoRA adapter weights with base model...")
            # Merge and unload to get the full model
            model = model.merge_and_unload()
            
            # Save merged model
            merged_output_dir = f"{cfg.training.output_dir}/merged"
            print(f"Saving merged model to {merged_output_dir}")
            model.save_pretrained(merged_output_dir)
            tokenizer.save_pretrained(merged_output_dir)
            
            if cfg.hf_hub.hub_model_id:
                print(f"Pushing merged model to HuggingFace Hub: {cfg.hf_hub.hub_model_id}")
                model.push_to_hub(
                    cfg.hf_hub.hub_model_id,
                    private=cfg.hf_hub.hub_private,
                    safe_serialization=True
                )
                tokenizer.push_to_hub(
                    cfg.hf_hub.hub_model_id,
                    private=cfg.hf_hub.hub_private
                )
                print(f"Model successfully pushed to https://huggingface.co/{cfg.hf_hub.hub_model_id}")
            else:
                print("Warning: hub_model_id not specified, skipping HuggingFace Hub push")
        else:
            # Push adapters only (not merged)
            if cfg.hf_hub.hub_model_id:
                print(f"Pushing LoRA adapters to HuggingFace Hub: {cfg.hf_hub.hub_model_id}")
                model.push_to_hub(
                    cfg.hf_hub.hub_model_id,
                    private=cfg.hf_hub.hub_private,
                    safe_serialization=True
                )
                tokenizer.push_to_hub(
                    cfg.hf_hub.hub_model_id,
                    private=cfg.hf_hub.hub_private
                )
                print(f"Adapters successfully pushed to https://huggingface.co/{cfg.hf_hub.hub_model_id}")
            else:
                print("Warning: hub_model_id not specified, skipping HuggingFace Hub push")
    
    if cfg.wandb.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
