import torch
from tqdm import tqdm
import csv
import numpy as np

from transformers import Trainer
from torch.nn import CrossEntropyLoss

from transformers.trainer_pt_utils import nested_detach
from metrics import aac_metrics
from eval_metrics import write_json
from pathlib import Path

class BARTAACTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if 'file_name' in inputs.keys():
            file_name = inputs.pop('file_name')
        if 'cond_text' in inputs.keys():
            cond_text = inputs.pop('cond_text')
        len_audio_embs = inputs.pop('len_audio_embs')
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs['labels']
        else:
            labels = None
        
        loss, outputs = model(**inputs)
        
        if labels is not None:
            loss = self.label_smoother({'logits': outputs}, labels)
            
        tqdm.write(str(loss.item()))
        
        return (loss, outputs) if return_outputs else loss # Do not return past key values
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                else:
                    logits = outputs[1:]
                
            else:
                loss = None
                if self.use_amp:
                    with autocast():
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                else:
                    logits = outputs
                if self.args.past_index >= 0:
                    self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
            
        logits = nested_detach(logits)
        
        if len(logits) == 1:
            logits = logits[0]
        logits = logits[0].argmax(2)
        
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None
        
        return (loss, logits, labels)
        
    def ar_generate(self, eval_dataset):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        model = self.model
        
        model.eval()
        
        all_labels = []
        all_preds = []
        all_filenames = []
        
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(eval_dataloader)):
                inputs = self._prepare_inputs(inputs)
                labels = inputs.pop('labels')
                if 'len_audio_embs' in inputs.keys():
                    len_audio_embs = inputs.pop('len_audio_embs')
                if 'loss_mask' in inputs.keys():
                    loss_mask = inputs.pop('loss_mask')
                if 'decoder_input_ids' in inputs.keys():
                    decoder_input_ids = inputs.pop('decoder_input_ids')
                if 'decoder_attention_mask' in inputs.keys():
                    decoder_attention_mask = inputs.pop('decoder_attention_mask')
                if 'file_name' in inputs.keys():
                    file_name = inputs.pop('file_name')
                if 'cond_text' in inputs.keys():
                    cond_text = inputs.pop('cond_text')
                    for c in cond_text:
                        print(c)
                outputs = model.generate_lm(**inputs)
                
                all_labels.append(labels.cpu())
                all_preds.append(F.pad(outputs, (0,labels.size(1)-outputs.size(1)),'constant', 1).cpu())
                all_filenames.extend(file_name)
                
        all_labels = torch.cat(all_labels, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()
        print(all_labels)
        print(all_preds)
        metrics, all_gt_captions, all_pred_captions = aac_metrics({'predictions': all_preds, 'label_ids': all_labels, 'filenames': all_filenames})
        
        write_json(metrics, Path(self.args.output_dir).joinpath('metrics_coco_beam.json'))
        with open(Path(self.args.output_dir).joinpath('generated_captions_beam.txt'), 'w') as f:
            for i_file in range(len(all_pred_captions)):
                f.write('----- File {} -----\n'.format(i_file))
                f.write('GT:   '+'\n')
                for i_gt in range(len(all_gt_captions[i_file])):
                    f.write('      '+all_gt_captions[i_file][i_gt]+'\n')
                f.write('Pred: '+all_pred_captions[i_file]+'\n')
        
        return metrics


