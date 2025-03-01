import argparse
import os

from transformers import AutoModelForSequenceClassification

import vec2text.experiments as experiments
from vec2text import analyze_utils
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--embedder", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--hidden", action="store_true")
parser.add_argument("--from_gradients", action="store_true")
parser.add_argument("--tinydata", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default="dia_saves_copy")
parser.add_argument("--do_eval", action="store_true", default=False)
args = parser.parse_args()

cais = "compute-permanent-node" in os.uname().nodename
use_less_data = 1000 if args.tinydata else -1
embeddings_from_layer_n = 4 if args.hidden else None  # 13 layers in gpt2
model = args.model
print("model", flush=True)
resume_from_checkpoint = args.resume

if cais:
    print("Running on CAIS")
    batch_size = 128
else:
    batch_size = 64

hidden_data_suffix = "__hidden" if embeddings_from_layer_n is not None else ""
tiny_data_suffix = "__tinydata" if use_less_data > 0 else ""
from_gradients_suffix = "__from_gradients" if args.from_gradients else ""

# model_args = ModelArguments(
#     model_type=model,
#     embedder_model_name=args.embedder,
#     use_frozen_embeddings_as_input=False,
#     embeddings_from_layer_n=embeddings_from_layer_n,
#     embedder_no_grad=(not args.from_gradients),
# )

# data_args = DataArguments(
#     dataset_name="one_million_instructions",
#     use_less_data=use_less_data,
# )


# training_args = TrainingArguments(
#     output_dir=f"{args.output_dir}/{model}__{args.embedder.replace('/', '__')}__one-million-instructions{hidden_data_suffix}{tiny_data_suffix}{from_gradients_suffix}",
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=0,
#     experiment=("inversion_from_gradients" if args.from_gradients else "inversion_from_logits"),
#     use_wandb=args.wandb,
# )


experiment, trainer = analyze_utils.load_experiment_and_trainer_from_pretrained(
     "/share/u/koyena/rep-inverse/vec2text/dia_saves_copy/t5-small__EleutherAI__pythia-14m__one-million-instructions__from_gradients/checkpoint-420000", use_less_data=250000, use_wandb=args.wandb
)
train_datasets = experiment._load_train_dataset_uncached(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    embedder_tokenizer=trainer.embedder_tokenizer
)
val_datasets = experiment._load_val_datasets_uncached(
    model=trainer.model,
    tokenizer=trainer.tokenizer,
    embedder_tokenizer=trainer.embedder_tokenizer
)


# print(train_datasets, flush=True)
trainer.args.per_device_eval_batch_size = 16
trainer.sequence_beam_width = 1
trainer.num_gen_recursive_steps = 20
trainer.evaluate(
    eval_dataset=val_datasets["person_finder"]
)


# experiment = experiments.experiment_from_args(model_args, data_args, training_args)

# trainer = experiment.load_trainer()
# print("Created trainer", flush=True)
# trainer.model.to(training_args.device)
# trainer.train(resume_from_checkpoint=resume_from_checkpoint)
# trainer.evaluate()