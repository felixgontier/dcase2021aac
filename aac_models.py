import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartForConditionalGeneration

class BARTAAC(nn.Module):
    def __init__(self, settings, bart_config):
        super().__init__()
        
        # Parameters
        audio_emb_size = settings['data']['audio_emb_size']
        lm_emb_size = 1024 if 'bart-large' in settings['lm']['name'] else 768
        bart_name = settings['lm']['name']
        pretrained_lm = settings['lm']['pretrained']
        n_adapt_layers = settings['adapt']['nb_layers']
        
        self.token_conditioning = settings['lm']['token_conditioning']
        self.combine_tokens_audio = settings['lm']['combine_tokens_audio']
        
        self.audio_adapt = None
        if n_adapt_layers >= 1:
            audio_adapt_list = [nn.Linear(audio_emb_size, lm_emb_size)]
            for i_adapt in range(n_adapt_layers-1):
                audio_adapt_list.append(nn.ReLU(inplace=True))
                audio_adapt_list.append(nn.Linear(lm_emb_size, lm_emb_size))
            
            self.audio_adapt = nn.Sequential(*audio_adapt_list)
        
        if self.combine_tokens_audio == 'cat':
            self.tok_audio_adapt = nn.Linear(bart_config.d_model+(bart_config.d_model if self.audio_adapt is not None else audio_emb_size), lm_emb_size)
        
        if pretrained_lm:
            self.bart_lm = BartForConditionalGeneration.from_pretrained(bart_name)
        else:
            self.bart_lm = BartForConditionalGeneration(bart_config)
        
    def forward(self,
                audio_features=None,
                cond_tokens=None,
                input_ids=None,
                attention_mask=None,
                audio_attention_mask=None,
                decoder_input_ids=None,
                decoder_attention_mask=None,
                head_mask=None,
                decoder_head_mask=None,
                encoder_outputs=None,
                past_key_values=None,
                inputs_embeds=None,
                decoder_inputs_embeds=None,
                labels=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True,
        ):
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_features)
        else:
            audio_embs = audio_features
        
        if self.token_conditioning:
            if self.combine_tokens_audio == 'cat':
                lm_embs = self.tok_audio_adapt(torch.cat(((self.bart_lm.model.encoder.embed_tokens(cond_tokens) * self.bart_lm.model.encoder.embed_scale), audio_embs), dim=2))
            elif self.combine_tokens_audio == 'add': # Addition
                lm_embs = audio_embs + (self.bart_lm.model.encoder.embed_tokens(cond_tokens) * self.bart_lm.model.encoder.embed_scale)
            else: # None: bypass audio features, still computes adapt. layers if they exist, adapt.nb_layers should be set to 0 in this case
                lm_embs = (self.bart_lm.model.encoder.embed_tokens(cond_tokens) * self.bart_lm.model.encoder.embed_scale)
        else:
            lm_embs = audio_embs
        
        encoder_outputs = self.bart_lm.model.encoder(
                    input_ids=None,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    inputs_embeds=lm_embs,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True)['last_hidden_state']
        
        encoder_outputs = [encoder_outputs]
        
        # Only decoder is computed here
        outputs = self.bart_lm(input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    head_mask=head_mask,
                    decoder_head_mask=decoder_head_mask,
                    encoder_outputs=encoder_outputs,
                    past_key_values=past_key_values,
                    inputs_embeds=None,
                    decoder_inputs_embeds=decoder_inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=True,
          )
        
        return outputs['loss'], outputs['logits']
    
    def generate_lm(self,
                audio_features=None,
                cond_tokens=None,
                attention_mask=None,
                audio_attention_mask=None,
                inputs_embeds=None,
                labels=None,
        ):
        
        # First pass with audio embeddings, does not work with _prepare_inputs_for_generation because it uses input_ids, not input_embs
        self.bart_lm.force_bos_token_to_be_generated=True
        
        if self.audio_adapt is not None:
            audio_embs = self.audio_adapt(audio_features)
        else:
            audio_embs = audio_features
        
        encoder_outputs = None
        if self.token_conditioning:
            if self.combine_tokens_audio == 'cat':
                lm_embs = self.tok_audio_adapt(torch.cat(((self.bart_lm.model.encoder.embed_tokens(cond_tokens) * self.bart_lm.model.encoder.embed_scale), audio_embs), dim=2))
            elif self.combine_tokens_audio == 'add':
                lm_embs = audio_embs + (self.bart_lm.model.encoder.embed_tokens(cond_tokens) * self.bart_lm.model.encoder.embed_scale)
            else:
                lm_embs = (self.bart_lm.model.encoder.embed_tokens(cond_tokens) * self.bart_lm.model.encoder.embed_scale)
        else:
            lm_embs = audio_embs
        
        encoder_outputs = self.bart_lm.model.encoder(
                input_ids=None,
                attention_mask=attention_mask,
                head_mask=None,
                inputs_embeds=lm_embs,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=True)
        
        input_ids = torch.zeros((audio_embs.size(0),1)).long().cuda()
        input_ids[:, 0] = self.bart_lm.config.decoder_start_token_id
        
        outputs = self.bart_lm.generate(input_ids=None,
                    attention_mask=attention_mask,
                    decoder_input_ids=input_ids,
                    decoder_attention_mask=None,
                    encoder_outputs=encoder_outputs,
                    head_mask=None,
                    decoder_head_mask=None,
                    inputs_embeds=None,
                    decoder_inputs_embeds=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=None,
          )
        print(outputs)
        return outputs
        
