# rome-detector
For ROME-Detector Project with Bau Lab


## Execution notes
### 1
To test things out: `python run.py --model t5-small --embedder EleutherAI/pythia-14m --from_gradients --tinydata`. Once you're ready to run the full thing, remove `--tinydata` and add `--wandb`.

### 2
You'll have to change `[your conda environment]/lib/python3.11/site-packages/transformers/modeling_attn_mask_utils.py` line 279. Replace `elif (is_training or not is_tracing) and torch.all(attention_mask == 1):` with `elif False`.

Vague memory of why: to get per-sample gradients, we need to use grad and vmap from functorch. They don't support the kind of dynamic branching we see in that line. I think this should be fine bc effectively what we're doing is just disabling the flash attention kernel, making things slower, but still correct.
