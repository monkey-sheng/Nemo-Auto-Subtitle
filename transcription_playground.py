import nemo.collections.asr as nemo_asr
import librosa
import numpy as np
import torch
from glob import glob
import re


model = nemo_asr.models.EncDecCTCModelBPE.restore_from('models/stt_en_citrinet_1024.nemo')
# We will store transcriptions here
model.preprocessor.featurizer.dither = 0.0
model.preprocessor.featurizer.pad_to = 0
# Switch model to evaluation mode
model.eval()
# Freeze the encoder and decoder modules
model.encoder.freeze()
model.decoder.freeze()


@torch.no_grad()
def transcribe(audio, audio_length):
    hypotheses = []
    #audio = torch.tensor(audio, device='cuda')
    #audio_length = torch.tensor()
    logits, logits_len, greedy_predictions = model.forward(input_signal=audio, input_signal_length=audio_length)
    current_hypotheses = model._wer.ctc_decoder_predictions_tensor(greedy_predictions, predictions_len=logits_len)
    hypotheses += current_hypotheses
    return hypotheses


# audio1, _ = librosa.load('plda_data/voice/6.wav', sr=16000)
# audio2, _ = librosa.load('plda_data/voice/16.wav', sr=16000)
# print('shape of audios:', audio1.shape, audio2.shape)
# audio_tensor1 = torch.tensor([audio1], device='cuda')
# length_tensor1 = torch.tensor([len(audio1)], device='cuda')
# audio_tensor2 = torch.tensor([audio2], device='cuda')
# length_tensor2 = torch.tensor([len(audio2)], device='cuda')
#
# result1 = transcribe(audio_tensor1, length_tensor1)
# result2 = transcribe(audio_tensor2, length_tensor2)
# print(result1, result2)

files = glob('plda_data/background/*.wav')
results = []
for file in files:
    audio, _ = librosa.load(file, sr=16000)
    for i in range(0, len(audio), 4000):
        audio_segment = audio[i:i+12000]
        if len(audio_segment) < 4800:
            print('skipping segments too short (<300ms)')
            continue
        audio_tensor = torch.tensor([audio_segment], device='cuda')
        length_tensor = torch.tensor([len(audio_segment)], device='cuda')
        results.append((file, i, transcribe(audio_tensor, length_tensor)[0]))
print(results)

