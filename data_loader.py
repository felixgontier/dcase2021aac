from pathlib import Path
from typing import Any, Dict, List, NewType
from torch.utils.data import DataLoader
from pathlib import Path
import torch.utils.data
import numpy as np
from tqdm import tqdm
import csv
from transformers import GPT2TokenizerFast, BartTokenizerFast

class ClothoDatasetBART(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, input_field_name, output_field_name, tokenizer, validation=None, normalize=False, cond_tok_field_name='', cond_tok_class_sel=None, cond_tok_time_sel=None, cond_tok_separator=None, max_audio_len:int=None, max_ctext_len:int=None, align_tokens_audio=None, kept_percent=100):
        """Initialization of a Clotho dataset object.
        
        :param data_dir: Data directory with Clotho dataset files.
        :type data_dir: pathlib.Path
        :param split: The split to use (`development`, `validation`)
        :type split: str
        :param input_field_name: Field name for the input values
        :type input_field_name: str
        :param output_field_name: Field name for the output (target) values.
        :type output_field_name: str
        """
        super(ClothoDatasetBART, self).__init__()
        the_dir = data_dir.joinpath(split)
        
        self.examples = sorted(the_dir.iterdir())
        if validation == False:
            self.examples = self.examples[:int(np.floor(len(self.examples)*0.9/5))*5]
        elif validation == True:
            self.examples = self.examples[int(np.floor(len(self.examples)*0.9/5))*5:]
        elif validation is None:
            pass # Whole dataset
        self.examples = self.examples[:int(np.round(len(self.examples)*kept_percent/100))]
        
        self.normalize = normalize
        self.max_audio_len = max_audio_len
        self.max_ctext_len = max_ctext_len
        self.align_tokens_audio = align_tokens_audio
        self.input_name = input_field_name
        self.output_name = output_field_name
        self.tok_name = cond_tok_field_name
        if self.tok_name is not None and 'logits' in self.tok_name:
            with open('./'+self.tok_name.replace('_logits','')+'_class_map.csv', 'r') as f:
                num_classes = sum(1 for l in f)-1
                self.class_map = [[] for l in range(num_classes)]
                f.seek(0)
                reader = csv.reader(f, delimiter=',')
                _ = next(reader)
                for r in reader:
                    c_ = ' '.join(r[2:])
                    c_ = c_.replace('etc.', '')
                    if '(' in c_:
                        c_ = c_[:c_.index('(')-1]
                    assert '.' not in c_
                    self.class_map[int(r[0])] = c_
                    
        self.cond_tok_class_sel = cond_tok_class_sel
        self.cond_tok_time_sel = cond_tok_time_sel
        self.cond_tok_separator = cond_tok_separator
        
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, item):
        ex = self.examples[item]
        ex = np.load(str(ex), allow_pickle=True)
        
        # ----- Labels/Decoder inputs -----
        ou_e = ex[self.output_name].item()
        
        tok_e = self.tokenizer(ou_e, max_length=64, return_tensors='pt', padding='max_length')
        if tok_e['input_ids'].size(1) > 64:
            tok_e['input_ids'] = tok_e['input_ids'][:,:64]
            tok_e['attention_mask'] = tok_e['attention_mask'][:,:64]
        
        # ----- Audio conditioning -----
        if isinstance(self.input_name, list): # Concat. embeddings
            in_e_temp = {inp_name: ex[inp_name].item() for inp_name in self.input_name}
            
            if all([in_e_temp[k].shape[0]==in_e_temp[list(in_e_temp.keys())[0]].shape[0] for k in in_e_temp.keys()]): # All same length: concat.
                pass
            elif sum([in_e_temp[k].shape[0]!=1 for k in in_e_temp.keys()])==1: # Only one does not have length of 1: replicate others to match
                max_length = np.amax([in_e_temp[k].shape[0] for k in in_e_temp.keys()])
                in_e_temp = {k: (np.repeat(in_e_temp[k], max_length, axis=0) if in_e_temp[k].shape[0]==1 else in_e_temp[k]) for k in in_e_temp.keys()}
            else:
                raise NotImplementedError
            
            in_e = torch.Tensor(np.concatenate(([in_e_temp[k] for k in in_e_temp.keys()]), axis=1)).float()
        else:
            in_e = ex[self.input_name].item()
            
            in_e = torch.Tensor(in_e).float().unsqueeze(0)
            if self.normalize:
                in_e = (in_e-in_e.mean())/in_e.std()
            # For fixed length audio features, append zeros as padding to reach the desired number of timesteps
            in_e = in_e.squeeze()
        
        if len(list(in_e.size())) == 1:
            in_e = in_e.unsqueeze(0)
        
        # ----- Conditioning inputs -----
        cond_text = None
        if self.tok_name is not None:
            if 'logits' in self.tok_name:
                cond_tok_logits = torch.Tensor(ex[self.tok_name].item()).float()
                # ----- Tag sampling -----
                if 'top' in self.cond_tok_time_sel: # Eg: 'top5'
                    if cond_tok_logits.size(0) == 1 and in_e.size(0) != 1:
                        cond_tok_logits = cond_tok_logits.repeat(in_e.size(0), 1)
                    cond_tok_logits = torch.mean(cond_tok_logits, dim=0) # Average along time
                    cond_tok_class = torch.argsort(cond_tok_logits, descending=True)[:int(self.cond_tok_time_sel[3:])]
                else: # Num tags = num logits
                    if in_e.size(0) != cond_tok_logits.size(0) and cond_tok_logits.size(0) == 1: # Eg.: panns logits with vggish embeddings
                        cond_tok_logits = cond_tok_logits.repeat(in_e.size(0), 1)
                    
                    if self.cond_tok_class_sel == 'max':
                        cond_tok_class = torch.argmax(cond_tok_logits, dim=1)
                    else: # Sample
                        cond_tok_class = torch.multinomial(cond_tok_logits, 1)
                len_cond_tok = cond_tok_logits.size(0)
                # ----- Reformat text inputs -----
                if self.cond_tok_time_sel == 'unroll' and in_e.size(0) == cond_tok_class.size(0)-1: # Some 9.5s files, vggish cuts whereas yamnet pads
                    cond_tok_class = cond_tok_class[:in_e.size(0)]
                cond_text = self.cond_tok_separator.join([self.class_map[int(ic)] for ic in cond_tok_class])+'.'
            else:
                cond_tok_text = ex[self.tok_name].item() # Assume array of strings
                if cond_tok_text == []:
                    cond_tok_text = [' ']
                len_cond_tok = len([ic for ic in cond_tok_text if ic != '[]'])
                cond_text = (self.cond_tok_separator.join([ic for ic in cond_tok_text if ic != '[]'])+'.').replace('_', ' ')
            cond_tokens = self.tokenizer(cond_text, max_length=self.max_ctext_len, return_tensors='pt', padding='max_length') # Reduce to 128 on AudioCaps for speedup
            att_mask = cond_tokens['attention_mask'].squeeze()
            cond_tokens = cond_tokens['input_ids'].squeeze().long()
        else:
            cond_tokens = None
            att_mask = None
        
        # ----- Reformat audio inputs -----
        if self.tok_name is None or not self.align_tokens_audio: # No token cond. or separate audio encoding: do not align audio and text inputs
            audio_att_mask = torch.zeros((self.max_audio_len,)).long()
            
            len_in_e = in_e.size(0)
            
            audio_att_mask[:len_in_e] = 1
            if in_e.size(0) > self.max_audio_len:
                in_e = in_e[:self.max_audio_len, :]
            elif in_e.size(0) < self.max_audio_len:
                in_e = torch.cat([in_e, torch.zeros(self.max_audio_len - in_e.size(0), in_e.size(1)).float()]) # BART encoder max_length = 1024 ?
        else:
            audio_att_mask = None
            if 'top' in self.cond_tok_time_sel: # Eg: 'top5'
                if in_e.size(0) != 1:
                    in_e = in_e.mean(dim=0, keepdim=True)
                in_e = in_e[0, :].repeat(cond_tokens.size(0), 1)
            else: # Num tags = num logits
                if in_e.size(0) != len_cond_tok and in_e.size(0) == 1: # Eg.: yamnet logits with panns embeddings
                    in_e = in_e.repeat(len_cond_tok, 1)
                if in_e.size(0) != len_cond_tok and int(np.ceil(len_cond_tok/10.)/in_e.size(0))==1: # Panns for Clotho
                    in_e = in_e.repeat_interleave(10, dim=0)
                    in_e = in_e[:len_cond_tok, :]
                    if in_e.size(0) < len_cond_tok: # e.g. 30 < 31 in some cases
                        in_e = torch.cat((in_e, in_e[-1,:].repeat(len_cond_tok-in_e.size(0), 1)), dim=0)
                if in_e.size(0) != len_cond_tok and in_e.size(0) == 2:
                    in_e = in_e[0,:].repeat(len_cond_tok, 1)
                assert in_e.size(0) == len_cond_tok, 'Audio embeddings ({}) and tags ({}) dimensions do not match for file {}.'.format(in_e.size(0), len_cond_tok, ex['file_name'].item())
                sep_token = self.tokenizer.encode('and'+self.cond_tok_separator, return_tensors='pt', add_special_tokens=False)[0,1]
                audio_features = torch.zeros((att_mask.size(0), in_e.size(1)))
                i_frame = 0
                for i_tok in range(att_mask.size(0)):
                    if cond_tokens[i_tok] == 1 or cond_tokens[i_tok] == 0 or cond_tokens[i_tok] == 2:
                        pass # BOS, EOS, PAD
                    elif cond_tokens[i_tok] == sep_token: # separator
                        i_frame += 1
                    else:
                        audio_features[i_tok, :] = in_e[i_frame, :]
                in_e = audio_features
        
        return {'audio_features': in_e,
                'attention_mask': att_mask if att_mask is not None else audio_att_mask,
                'audio_attention_mask': audio_att_mask,
                'decoder_attention_mask': tok_e['attention_mask'].squeeze(),
                'len_audio_embs': in_e.size(0),
                'file_name': ex['file_name'].item(),
                'labels': tok_e['input_ids'].squeeze().long(),
                'cond_tokens': cond_tokens,
                'cond_text': cond_text}


InputDataClass = NewType("InputDataClass", Any)
def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - ``label``: handles a single value (int or float) per object
        - ``label_ids``: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        elif k not in ("label", "label_ids") and v is not None: # str
            batch[k] = [f[k] for f in features]
    
    return batch

    
def get_clotho_dataset(split, settings, tokenizer):
    """Gets the data loader.

    :param split: Split to be used.
    :type split: str
    :param is_training: Is training data?
    :type is_training: bool
    :param settings: Settings.
    :type settings: dict
    :return: Data loader.
    :rtype: torch.utils.data.DataLoader
    """
    data_dir = Path(settings['data']['root_dir'], settings['data']['features_dir'])
    
    if split == 'development' and settings['workflow']['validate']:
        return ClothoDatasetBART(
                data_dir=data_dir,
                split=split,
                input_field_name=settings['data']['input_field_name'],
                output_field_name=settings['data']['output_field_name'],
                cond_tok_field_name=settings['data']['cond_tok_field_name'],
                cond_tok_class_sel=settings['data']['cond_tok_class_sel'],
                cond_tok_time_sel=settings['data']['cond_tok_time_sel'],
                cond_tok_separator=settings['data']['cond_tok_separator'],
                tokenizer=tokenizer,
                normalize=settings['data']['normalize'],
                max_audio_len=settings['data']['max_audio_len'],
                max_ctext_len=settings['data']['max_ctext_len'],
                align_tokens_audio=settings['data']['align_tokens_audio'],
                validation=False),  ClothoDatasetBART(
                data_dir=data_dir,
                split=split,
                input_field_name=settings['data']['input_field_name'],
                output_field_name=settings['data']['output_field_name'],
                cond_tok_field_name=settings['data']['cond_tok_field_name'],
                cond_tok_class_sel=settings['data']['cond_tok_class_sel'],
                cond_tok_time_sel=settings['data']['cond_tok_time_sel'],
                cond_tok_separator=settings['data']['cond_tok_separator'],
                tokenizer=tokenizer,
                normalize=settings['data']['normalize'],
                max_audio_len=settings['data']['max_audio_len'],
                max_ctext_len=settings['data']['max_ctext_len'],
                align_tokens_audio=settings['data']['align_tokens_audio'],
                validation=True)
    else:
        return ClothoDatasetBART(
                data_dir=data_dir,
                split=split,
                input_field_name=settings['data']['input_field_name'],
                output_field_name=settings['data']['output_field_name'],
                cond_tok_field_name=settings['data']['cond_tok_field_name'],
                cond_tok_class_sel=settings['data']['cond_tok_class_sel'],
                cond_tok_time_sel=settings['data']['cond_tok_time_sel'],
                cond_tok_separator=settings['data']['cond_tok_separator'],
                tokenizer=tokenizer,
                normalize=settings['data']['normalize'],
                max_audio_len=settings['data']['max_audio_len'],
                max_ctext_len=settings['data']['max_ctext_len'],
                align_tokens_audio=settings['data']['align_tokens_audio'],
                validation=None), None
                    
def get_audiocaps_dataset(split, settings, tokenizer):
    data_dir = Path(settings['data']['root_dir'], settings['data']['features_dir'])
    if split == 'development' and settings['workflow']['validate']:
        return ClothoDatasetBART(
                data_dir=data_dir,
                split='train',
                input_field_name=settings['data']['input_field_name'],
                output_field_name=settings['data']['output_field_name'],
                cond_tok_field_name=settings['data']['cond_tok_field_name'],
                cond_tok_class_sel=settings['data']['cond_tok_class_sel'],
                cond_tok_time_sel=settings['data']['cond_tok_time_sel'],
                cond_tok_separator=settings['data']['cond_tok_separator'],
                tokenizer=tokenizer,
                normalize=settings['data']['normalize'],
                max_audio_len=settings['data']['max_audio_len'],
                max_ctext_len=settings['data']['max_ctext_len'],
                align_tokens_audio=settings['data']['align_tokens_audio'],
                kept_percent=settings['data']['kept_percent'],
                validation=None),  ClothoDatasetBART(
                data_dir=data_dir,
                split='val',
                input_field_name=settings['data']['input_field_name'],
                output_field_name=settings['data']['output_field_name'],
                cond_tok_field_name=settings['data']['cond_tok_field_name'],
                cond_tok_class_sel=settings['data']['cond_tok_class_sel'],
                cond_tok_time_sel=settings['data']['cond_tok_time_sel'],
                cond_tok_separator=settings['data']['cond_tok_separator'],
                tokenizer=tokenizer,
                normalize=settings['data']['normalize'],
                max_audio_len=settings['data']['max_audio_len'],
                max_ctext_len=settings['data']['max_ctext_len'],
                align_tokens_audio=settings['data']['align_tokens_audio'],
                validation=None)
    else:
        return ClothoDatasetBART(
                data_dir=data_dir,
                split=split,
                input_field_name=settings['data']['input_field_name'],
                output_field_name=settings['data']['output_field_name'],
                cond_tok_field_name=settings['data']['cond_tok_field_name'],
                cond_tok_class_sel=settings['data']['cond_tok_class_sel'],
                cond_tok_time_sel=settings['data']['cond_tok_time_sel'],
                cond_tok_separator=settings['data']['cond_tok_separator'],
                tokenizer=tokenizer,
                normalize=settings['data']['normalize'],
                max_audio_len=settings['data']['max_audio_len'],
                max_ctext_len=settings['data']['max_ctext_len'],
                align_tokens_audio=settings['data']['align_tokens_audio'],
                validation=None), None

