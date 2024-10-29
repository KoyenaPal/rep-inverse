import argparse
import os

import vec2text.experiments as experiments
from vec2text.run_args import DataArguments, ModelArguments, TrainingArguments

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--embedder", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--hidden", action="store_true")
parser.add_argument("--from_gradients", action="store_true")
parser.add_argument("--tinydata", action="store_true")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--output_dir", type=str, default="dia_saves")
args = parser.parse_args()

cais = "compute-permanent-node" in os.uname().nodename
use_less_data = 1000 if args.tinydata else -1
embeddings_from_layer_n = 4 if args.hidden else None  # 13 layers in gpt2
model = args.model
resume_from_checkpoint = args.resume

if cais:
    print("Running on CAIS")
    batch_size = 128
else:
    batch_size = 64

hidden_data_suffix = "__hidden" if embeddings_from_layer_n is not None else ""
tiny_data_suffix = "__tinydata" if use_less_data > 0 else ""
from_gradients_suffix = "__from_gradients" if args.from_gradients else ""

model_args = ModelArguments(
    model_type=model,
    embedder_model_name=args.embedder,
    use_frozen_embeddings_as_input=False,
    embeddings_from_layer_n=embeddings_from_layer_n,
    embedder_no_grad=(not args.from_gradients),
)

data_args = DataArguments(
    dataset_name="one_million_instructions",
    use_less_data=use_less_data,
)

training_args = TrainingArguments(
    output_dir=f"{args.output_dir}/{model}__{args.embedder.replace('/', '__')}__one-million-instructions{hidden_data_suffix}{tiny_data_suffix}{from_gradients_suffix}",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    experiment=("inversion_from_gradients" if args.from_gradients else "inversion_from_logits"),
    use_wandb=args.wandb,
)

experiment = experiments.experiment_from_args(model_args, data_args, training_args)

trainer = experiment.load_trainer()
trainer.model.to(training_args.device)

trainer.train(resume_from_checkpoint=resume_from_checkpoint)
