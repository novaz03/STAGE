"""Trainer configuration utilities."""

from __future__ import annotations

import logging

from transformers import (
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .config import PipelineConfig

LOGGER = logging.getLogger(__name__)


def create_trainer(
    config: PipelineConfig,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset,
) -> Trainer:
    """Configure the Hugging Face Trainer."""
    training_args = TrainingArguments(
        output_dir=str(config.checkpoints_dir()),
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        dataloader_num_workers=config.dataloader_num_workers,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        optim="adamw_torch_fused",
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    LOGGER.info("Initialising trainer with dataset of %s blocks", len(dataset))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    return trainer
