import numpy as np
import csv
from tqdm import tqdm
from pathlib import Path
from input_features import yamnet_classify, panns_infer
import torch

def create_dataset():
    data_path = Path('data/audiocaps')
    vggish_model = torch.hub.load('harritaylor/torchvggish', 'vggish')
    vggish_model.eval()
    vggish_model.postprocess = False
    vggish_model.embeddings[5] = torch.nn.Sequential()
    
    splits = ['train', 'val', 'test']
    for split in splits:
        print('Split '+split+'.')
        out_path = Path('audiocaps_vggish_yamnet_panns/'+split)
        out_path.mkdir(parents=True, exist_ok=True)
        
        in_path = data_path.joinpath(split)
        file_list = [fname for fname in in_path.iterdir() if fname.suffix == '.wav']
        
        example_list = []
        with open(split+'.csv', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)
            for r in reader:
                example_list.append(r)        
        
        for ex in tqdm(example_list):
            # Audio file name format: Y<youtube_id>_<t_start>_<t_end>.wav
            file_name = 'Y'+ex[1]+'_'+ex[2]+'.000_'+str(int(ex[2])+10)+'.000.wav'
            if in_path.joinpath(file_name) in file_list:
                # Get caption
                caption = ex[3]
                
                # Compute VGGish embeddings and YAMNet logits
                yamnet_logits = yamnet_classify(str(in_path.joinpath(file_name)))
                vggish_embeddings = vggish_model.forward(str(in_path.joinpath(file_name))).detach().numpy()
                
                # Get PANNs logits and embeddings (global 10s)
                panns_logits, panns_embeddings = panns_infer(str(in_path.joinpath(file_name)))
                panns_logits = panns_logits.numpy()
                panns_embeddings = panns_embeddings.numpy()
                
                # Create recarray
                np_rec_array = np.rec.array(np.array(
                    (ex[1], vggish_embeddings, caption, yamnet_logits[0::2,:], panns_logits, panns_embeddings),
                    dtype=[
                        ('file_name', 'U{}'.format(len(ex[1]))),
                        ('vggish_embeddings', np.dtype(object)),
                        ('caption', 'U{}'.format(len(caption))),
                        ('yamnet_logits', np.dtype(object)),
                        ('panns_logits', np.dtype(object)),
                        ('panns_embeddings', np.dtype(object))
                    ]
                ))

                # Save recarray
                np.save(str(out_path.joinpath(
                        'audiocaps_{audio_file_name}_{caption_index}.npy'.format(
                        audio_file_name=ex[1], caption_index=ex[0]))), np_rec_array)
                
                
if __name__ == '__main__':
    create_dataset()
    
    
