import sys
import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import yamnet.params as yamnet_params
import yamnet.yamnet as yamnet_model
import torch
import librosa as lr
import panns.models as panns_models
import panns.config as panns_config

params = yamnet_params.Params()
yamnet = yamnet_model.yamnet_frames_model(params)
yamnet.load_weights('yamnet/yamnet.h5')

panns_wavegram_cnn = panns_models.Wavegram_Logmel_Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=panns_config.classes_num)
panns_wavegram_cnn.load_state_dict(torch.load('Wavegram_Logmel_Cnn14_mAP=0.439.pth', map_location=torch.device('cpu'))['model'])
panns_wavegram_cnn.eval()

def panns_infer(file_name):
    audio_data_full, _ = lr.core.load(file_name, sr=32000, mono=True)
    
    f_len = 320000
    n_frames = int(np.ceil(audio_data_full.shape[0]/f_len))
    panns_output = None
    for i_frame in range(n_frames):
        audio_data = audio_data_full[i_frame*f_len:np.minimum((i_frame+1)*f_len, audio_data_full.shape[0])]
        if audio_data.shape[0] < f_len:
            audio_data = np.concatenate((audio_data, np.zeros((f_len-audio_data.shape[0],))), axis=0)
        if audio_data.shape[0] > f_len:
            raise ValueError
            audio_data = audio_data[:f_len]
        audio_data = torch.Tensor(audio_data[None,:])
        with torch.no_grad():
            panns_output_temp = panns_wavegram_cnn(audio_data)
        if panns_output is None:
            panns_output = panns_output_temp
        else:
            panns_output['clipwise_output'] = torch.cat((panns_output['clipwise_output'], panns_output_temp['clipwise_output']), dim=0)
            panns_output['embedding'] = torch.cat((panns_output['embedding'], panns_output_temp['embedding']), dim=0)
    return panns_output['clipwise_output'], panns_output['embedding']

def yamnet_classify(file_name):
    wav_data, sr = sf.read(file_name, dtype=np.int16)
    assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    
    waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    waveform = waveform.astype('float32')
    
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)
    scores, embeddings, spectrogram = yamnet(waveform)
    
    return scores.numpy()

