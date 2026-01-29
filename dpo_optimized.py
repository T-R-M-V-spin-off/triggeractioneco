import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# Spostiamo tutto dentro una funzione main
def main():
    # ==================================================
    # CONFIGURAZIONE
    # ==================================================
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    OUTPUT_DIR = "./eco_dpo_model"
    MAX_LEN = 128
    EPOCHS = 1
    LR = 2e-5

    # ==================================================
    # CARICAMENTO DATASET
    # ==================================================
    print("Caricamento dataset...")
    dataset = load_dataset("csv", data_files="output_rules_modified.csv")["train"]

    def format_example(example):
        return {
            "prompt": example["rule"],
            "chosen": example["eco_rule"],
            "rejected": example["non_eco_rule"],
        }

    # IMPORTANTE: num_proc=1 su Windows per evitare ArrowInvalid/OSError
    dataset = dataset.map(format_example, remove_columns=dataset.column_names, num_proc=1)

    # ==================================================
    # TOKENIZER E MODELLO
    # ==================================================
    print("Caricamento tokenizer e modello...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # ==================================================
    # LORA
    # ==================================================
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # ==================================================
    # DPO CONFIG
    # ==================================================
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        max_length=MAX_LEN,
        max_prompt_length=MAX_LEN // 2,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        report_to="none",
    )

    # ==================================================
    # DPO TRAINER
    # ==================================================
    print("\n🚀 Inizializzazione Trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n🔥 Starting DPO training...\n")
    trainer.train()

    print("\nSalvataggio finale...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("\n✅ Completato!")

# Questo blocco è OBBLIGATORIO su Windows per il multiprocessing
if __name__ == "__main__":
    main()