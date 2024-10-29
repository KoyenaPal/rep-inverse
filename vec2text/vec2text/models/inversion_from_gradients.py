import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor as T
from torch.func import functional_call, grad, vmap

from vec2text.models.config import InversionConfig
from vec2text.models.inversion import InversionModel


def zero_embedding_except_topk(
    embeddings: torch.Tensor, vocab_size: int, k: torch.Tensor, default_val: float
) -> torch.Tensor:
    # return embeddings
    topk = embeddings[:, :vocab_size].topk(k=k, dim=1)
    new_emb = torch.zeros_like(embeddings, device=embeddings.device) + default_val
    return new_emb.scatter_add(1, topk.indices, topk.values)


def compute_loss(model):
    loss_fn = nn.CrossEntropyLoss()

    def compute_loss_(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)
        model_output = functional_call(
            model,
            (params, buffers),
            (batch, targets),
        )
        return loss_fn(
            model_output.logits.view(-1, model_output.logits.shape[-1]),
            targets.view(-1),
        )

    return compute_loss_


class InversionFromFullGradientsModel(InversionModel):
    def __init__(self, config: InversionConfig):
        super().__init__(config=config)
        self.encoder_hidden_dim = self.encoder_decoder.config.hidden_size
        self.embedder_is_decoder = True

        self.embedder_params = {k: v.detach().to("cuda") for k, v in self.embedder.named_parameters()}
        self.embedder_buffers = {k: v.detach().to("cuda") for k, v in self.embedder.named_buffers()}

    def call_embedding_model(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        inputs_str = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        emb_input_ids = self.embedder_tokenizer(
            inputs_str,
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(next(self.parameters()).device)

        # pad input_ids to match the shape of emb_input_ids
        input_ids = nn.functional.pad(input_ids, (0, emb_input_ids.input_ids.shape[1] - input_ids.shape[1]))

        ft_compute_grad = grad(compute_loss(self.embedder))
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(
            self.embedder_params,
            self.embedder_buffers,
            emb_input_ids.input_ids,
            input_ids,
        )

        return ft_per_sample_grads

    def embed_and_project(
        self,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        frozen_embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grads = self.call_embedding_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        Us = []
        for g in grads.values():
            # hacky way to make the "left" vector the smaller one
            if len(g.shape) == 2:
                g = g.unsqueeze(-1)
            elif g.shape[1] < g.shape[2]:
                g = g.transpose(1, 2)

            U, _, _ = torch.svd_lowrank(g, q=1)

            if len(U.shape) == 3:
                Us.append(U.squeeze(-1))
            else:
                Us.append(U)

        reduced_grad = torch.cat(Us, dim=-1)

        # pad with zeros to be divisible by 768
        num_zeros_to_add = self.encoder_hidden_dim - (reduced_grad.shape[-1] % self.encoder_hidden_dim)
        reduced_grad = nn.functional.pad(reduced_grad, (0, num_zeros_to_add))

        reduced_grad = reduced_grad.view(reduced_grad.shape[0], -1, self.encoder_hidden_dim)

        attention_mask = torch.ones((reduced_grad.shape[0], reduced_grad.shape[1]), device=reduced_grad.device)

        assert reduced_grad.shape == (
            attention_mask.shape[0],
            attention_mask.shape[1],
            self.encoder_hidden_dim,
        )
        return reduced_grad, attention_mask

    def generate(
        self,
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        generation_kwargs = copy.copy(generation_kwargs)  # make a copy so we can edit
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            frozen_embeddings=inputs.get("frozen_embeddings"),
        )

        if "decoder_input_ids" in inputs:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                decoder_input_ids=inputs["decoder_input_ids"],
                # decoder_attention_mask=inputs["decoder_attention_mask"],
                **generation_kwargs,
            )
        else:
            return self.encoder_decoder.generate(
                # required: input embeddings
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                # optional: input IDs (for starting generation).
                # typically not set unless generating prefixes for
                # reranking.
                **generation_kwargs,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        frozen_embeddings: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        inputs_embeds, attention_mask = self.embed_and_project(
            input_ids=input_ids,
            attention_mask=attention_mask,
            frozen_embeddings=frozen_embeddings,
        )

        return self.encoder_decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values,
        )
