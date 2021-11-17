import torch
from pathlib import Path
import argparse

from data_loader import get_clotho_dataset, get_audiocaps_dataset, default_data_collator

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

from aac_models import *
from trainer import *
from metrics import aac_metrics

from torch.nn import Linear, LayerNorm
from transformers.models.gpt2.modeling_gpt2 import Attention, MLP
from transformers.models.bart.modeling_bart import BartAttention
import yaml

def main(config):
    # Settings
    with Path('./exp_settings/', config.exp+'.yaml').open('r') as f:
        settings = yaml.safe_load(f)
    print(settings)
    
    if 'seed' in settings['training'].keys():
        torch.manual_seed(settings['training']['seed'])
    else:
        torch.manual_seed(0)
    
    if settings['workflow']['evaluate'] and not settings['workflow']['train']:
        if settings['data']['cond_tok_class_sel'] == 'sample':
            settings['data']['cond_tok_class_sel'] = 'max'
    
    training_args = TrainingArguments(output_dir='./outputs/'+config.exp+'_out', learning_rate=settings['training']['lr'], label_smoothing_factor=settings['training']['label_smoothing_factor'] if 'label_smoothing_factor' in settings['training'].keys() else 0.0)
    training_args.per_device_train_batch_size = settings['data']['batch_size']
    training_args.gradient_accumulation_steps = settings['training']['gradient_accumulation_steps']
    training_args.dataloader_num_workers = settings['data']['num_workers']
    training_args.save_steps = settings['training']['save_steps']
    training_args.num_train_epochs = float(settings['training']['nb_epochs']) # Has to be a float for logging
    print(training_args)
    
    lm_config = AutoConfig.from_pretrained(settings['lm']['name'])
    print(lm_config)
    
    tokenizer = AutoTokenizer.from_pretrained(settings['lm']['name'], use_fast=True)
    data_train = None
    data_eval = None
    if settings['workflow']['train']:
        if 'clotho' in settings['data']['root_dir']:
            if 'end_to_end' in settings['lm'].keys() and settings['lm']['end_to_end']:
                data_train, data_eval = get_clotho_tag_dataset('development', settings, tokenizer)
            else:
                data_train, data_eval = get_clotho_dataset('development', settings, tokenizer)
        else: # Audiocaps
            if 'end_to_end' in settings['lm'].keys() and settings['lm']['end_to_end']:
                data_train, data_eval = get_audiocaps_tag_dataset('development', settings, tokenizer)
            else:
                data_train, data_eval = get_audiocaps_dataset('development', settings, tokenizer)
        print('Loaded development dataset.')
    
    if settings['workflow']['validate']:
        from transformers.trainer_utils import EvaluationStrategy
        training_args.evaluation_strategy = EvaluationStrategy.STEPS
        #training_args.evaluation_strategy = EvaluationStrategy.EPOCH
        training_args.eval_steps = settings['training']['eval_steps']
    
    if 'gpt2' in settings['lm']['name']:
        model = CondGPT2AAC(settings, lm_config, vocab_size=len(data_train.tokenizer))
        
    elif 'bart' in settings['lm']['name']:
        if 'end_to_end' in settings['lm'].keys() and settings['lm']['end_to_end']:
            model = BARTTagAAC(settings, lm_config)
        else:
            model = BARTAAC(settings, lm_config)
    
    if 'custom_pretrained_ckpt' in settings['lm'].keys():
        if settings['lm']['custom_pretrained_ckpt']: # Not None or False
            model.load_state_dict(torch.load(settings['lm']['custom_pretrained_ckpt']))
        
    
    print(model)
    
    if settings['adapt']['pretrained'] and settings['workflow']['train']:
        audio_lm_dict = model.audio_lm.state_dict()
        pretrained_dict = torch.load(settings['adapt']['pretrained_path'], map_location='cpu')
        pretrained_dict = {k.replace('audio_lm.', ''): v for k, v in pretrained_dict.items() if 'audio_lm' in k}
        audio_lm_dict.update(pretrained_dict)
        model.audio_lm.load_state_dict(audio_lm_dict)
    
    # Freezing
    if 'bart' in settings['lm']['name']:
        if settings['lm']['freeze_all']:
            for p in model.bart_lm.parameters():
                p.requires_grad = False
            for p in model.bart_lm.model.encoder.embed_positions.parameters():
                p.requires_grad = True
            for p in model.bart_lm.model.encoder.layers[0].self_attn.parameters():
                p.requires_grad = True
            
        if settings['lm']['freeze_dec']:
            for p in model.bart_lm.model.shared.parameters():
                p.requires_grad = False
            for p in model.bart_lm.model.decoder.parameters():
                p.requires_grad = False
            for p in model.bart_lm.lm_head.parameters():
                p.requires_grad = False
        if settings['lm']['freeze_enc']:
            for p in model.bart_lm.model.encoder.parameters():
                p.requires_grad = False
        if settings['lm']['freeze_attn']:
            for l in model.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze_mlp']:
            for l in model.bart_lm.modules():
                if isinstance(l, Linear):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze_dec_attn']:
            for l in model.bart_lm.model.decoder.modules():
                if isinstance(l, BartAttention):
                    for p in l.parameters():
                        p.requires_grad = False
        if settings['lm']['freeze_dec_mlp']:
            for l in model.bart_lm.model.decoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze_dec_self_attn']:
            for l in model.bart_lm.model.decoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze_enc_mlp']:
            for l in model.bart_lm.model.encoder.layers:
                for p in l.fc1.parameters():
                    p.requires_grad = False
                for p in l.fc2.parameters():
                    p.requires_grad = False
        if settings['lm']['freeze_enc_attn']:
            for l in model.bart_lm.model.encoder.layers:
                for p in l.self_attn.parameters():
                    p.requires_grad = False
    if 'end_to_end' in settings['lm'].keys() and settings['lm']['end_to_end']:
        if settings['lm']['freeze_tagger']:
            for p in model.audio_tagger.parameters():
                p.requires_grad = False
    
    print('Num parameters: {}'.format(sum(p.numel() for p in model.parameters())))
    print('Num trainable parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad==True)))
    
    if 'bart' in settings['lm']['name']:
        trainer = BARTAACTrainer(model, args=training_args, data_collator=default_data_collator, train_dataset=data_train, eval_dataset=data_eval)
    
    if settings['workflow']['train']:
        trainer.train()
    
    if settings['workflow']['evaluate']:
        pretrained_dict = torch.load('./outputs/'+config.exp+'_out/'+'checkpoint-'+str(settings['lm']['checkpoint_eval'])+'/pytorch_model.bin')
        # Retro compatibility
        for key in list(pretrained_dict.keys()):
            pretrained_dict[key.replace('audio_lm.', 'audio_adapt.')] = pretrained_dict.pop(key)
        model.load_state_dict(pretrained_dict)
        
        if 'bart-large-cnn' in settings['lm']['name']:
            model.bart_lm.config.task_specific_params['summarization']['min_length'] = 5
            model.bart_lm.config.length_penalty = 1.0 # From 2.0
            model.bart_lm.config.task_specific_params['summarization']['length_penalty'] = 1.0
        elif 'bart-large-xsum' in settings['lm']['name']:
            model.bart_lm.config.num_beams = 4 # From 6
        if 'bart' in settings['lm']['name']:
            model.bart_lm.config.min_length = 5 # From 11/56
            model.bart_lm.config.force_bos_token_to_be_generated = True
        print(model.bart_lm.config)
        if 'clotho' in settings['data']['root_dir']:
            if 'end_to_end' in settings['lm'].keys() and settings['lm']['end_to_end']:
                data_eval, _ = get_clotho_tag_dataset('evaluation', settings, tokenizer)
            else:
                data_eval, _ = get_clotho_dataset('evaluation', settings, tokenizer)
        else: # Audiocaps
            if 'end_to_end' in settings['lm'].keys() and settings['lm']['end_to_end']:
                data_eval, _ = get_audiocaps_tag_dataset('test', settings, tokenizer)
            else:
                data_eval, _ = get_audiocaps_dataset('test', settings, tokenizer)
        print('Loaded development dataset.')
        
        trainer.ar_generate(data_eval)
        
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--exp', type=str, default='exp001', help='Experience settings YAML')
    
    config = parser.parse_args()
    main(config)

