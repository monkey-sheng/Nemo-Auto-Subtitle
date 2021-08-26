import gc
import pickle
import re
import librosa
import numpy as np
import torch
from plda import Classifier
from glob import glob
import nemo.collections.asr as nemo_asr
import nemo.collections.nlp as nemo_nlp

# the diarization model also works (actually a subclass of verification model), not sure about accuracy though
model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from('models/speakerdiarization_speakernet.nemo')
#model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from('models/speakerverification_speakernet.nemo')
model.eval()


@torch.no_grad()
def get_embedding(audio):
    # if isinstance(audio, str):
    #     audio, sr = librosa.load(audio, sr=16000)
    audio_length = audio.shape[0]
    audio_signal = torch.tensor([audio], device='cuda')
    audio_signal_len = torch.tensor([audio_length], device='cuda')
    something, embs = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return embs.cpu().numpy()[0]


@torch.no_grad()
def batch_get_embedding(audio_batch):
    audio_length = audio_batch.shape[1]
    audio_signal = torch.tensor(audio_batch, device='cuda')
    audio_signal_len = torch.tensor([audio_length] * audio_batch.shape[0], device='cuda')
    something, embs = model.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
    return embs.cpu().numpy()


def prepare_PLDA(data, labels):
    classifier = Classifier()
    classifier.fit_model(data, labels)
    # the returned classifier can be used to `predict(data, normalize_logps=True)`
    # optionally get normalized posterior, aka model certainties
    return classifier


# voices_embs = np.array(list(map(get_embedding, glob('data/voices/16k/*.wav'))))
# bg_embs = np.array(list(map(get_embedding, glob('data/background/16k/*.wav'))))
#
# data = np.concatenate((voices_embs, bg_embs))
# labels = [1] * 20 + [0] * 20
# cfr = prepare_PLDA(data, labels)
#
# WINDOW_SIZE = 0.3
# SR = 16000
# WINDOW_SAMPLE_SIZE = int(SR * WINDOW_SIZE)  # n samples per window
# STRIDE_SIZE = int(SR * 0.1)  # 100ms stride/step
#
# FILE_NAME = 'data/test.wav'
# wf, _ = librosa.load(FILE_NAME, sr=16000)
# predictions = []
#
# for i in range(0, len(wf), STRIDE_SIZE):
#     signal = wf[i:i+WINDOW_SAMPLE_SIZE]
#     emb = get_embedding(signal)
#     pred = cfr.predict(emb)[0]
#     predictions.append(pred)
#
# print(predictions)


def get_plda_data_embeddings(FILES):
    # VOICE_FILES = glob('plda_data/voice/*.wav')
    embs = []
    for file in FILES:
        voice, _ = librosa.load(file, sr=16000)
        for i in range(0, len(voice), 1600):  # 100ms step size
            voice_seg = voice[i:i+4800]  # 300ms window size
            if len(voice_seg) < 3200:  # < 200ms, too short, skip
                continue
            embs.append(get_embedding(voice_seg))
    return embs


# VOICE_FILES = glob('plda_data/voice/*.wav')
# BG_FILES = glob('plda_data/background/*.wav')
# voice_embs, bg_embs = get_plda_data_embeddings(VOICE_FILES), get_plda_data_embeddings(BG_FILES)
try:
    cfr = pickle.load(open('classifier.pkl', 'rb'))
except:
    print('cannot load from pickle file')
    VOICE_FILES = glob('plda_data/voice/*.wav')
    BG_FILES = glob('plda_data/background/*.wav')
    voice_embs, bg_embs = get_plda_data_embeddings(VOICE_FILES), get_plda_data_embeddings(BG_FILES)
    cfr = prepare_PLDA(np.concatenate((bg_embs, voice_embs)), [0]*len(bg_embs)+[1]*len(voice_embs))
    pickle.dump(cfr, open('classifier.pkl', 'wb'))

STEP_SIZE = 1600  # 100ms step size
WINDOW_SIZE = 4800  # 300ms window size
CFR_BATCH_SIZE = 1024  # batch size to send to get embeddings

TEST_FILE = 'data/oblivion.m4a'
test_wav, _ = librosa.load(TEST_FILE, sr=16000)
test_results = []
# TODO: see if I can use batches here
for i in range(0, len(test_wav), STEP_SIZE * CFR_BATCH_SIZE):
    # FIXME: resize/pad when needed
    # batch = [test_wav[j:j+WINDOW_SIZE] for j in range(i, i + STEP_SIZE*CFR_BATCH_SIZE, STEP_SIZE)]
    batch = []
    for j in range(i, i + STEP_SIZE * CFR_BATCH_SIZE, STEP_SIZE):
        seg = test_wav[j:j + WINDOW_SIZE]
        batch.append(np.pad(test_wav[j:j + WINDOW_SIZE], (0, WINDOW_SIZE-len(seg))))
    batch = np.asarray(batch)
    print('batch shape', batch.shape)
    embs = batch_get_embedding(batch)
    assert embs.shape == (CFR_BATCH_SIZE, 256)
    batch_predictions = [cfr.predict(e) for e in embs]
    test_results += batch_predictions
# for i in range(0, len(test_wav), STEP_SIZE):  # 100ms step size
#     test_seg = test_wav[i:i+WINDOW_SIZE]  # 300ms window size
#     prediction = cfr.predict(get_embedding(test_seg))  # , normalize_logps=True
#     test_results.append(prediction)

print(test_results)


def format_to_time(sample: int) -> str:
    """
    :param sample: The sample index
    :return: an ASS compatible time stamp
    """
    rounded_time = round(sample / 16000, 2)
    hour = int(rounded_time // 3600)
    minute = int((rounded_time - 3600 * hour) // 60)
    sec = rounded_time - 3600 * hour - 60 * minute
    return '{}:{:02d}:{}'.format(hour, minute, '{:05.2f}'.format(sec))
    # str(round(sec, 2)).rjust(5, '0')


def format_tuple_to_time(t):
    return format_to_time(t[0]), format_to_time(t[1])

# TODO:
#  use a window and a step size to move across the whole audio
#  join segments if the resultant segment is NOT too long;
#  use the log probs to decide where to cut when a segment is too long,
#  also


with open('ass-template.ass', 'r') as template:
    ASS_TEMPLATE = template.read()
SPEECH_STATE = False
START_SAMPLE_INDEX, END_SAMPLE_INDEX = None, None
sample_timed_result = []
# Using 100ms step here, so 1600 samples per step
for i in range(len(test_results)):
    test_result = test_results[i][0]  # it's a tuple of (pred, [prob0, prob1])
    if not SPEECH_STATE and test_result:  # non-speech to speech
        START_SAMPLE_INDEX = i * STEP_SIZE
        SPEECH_STATE = True
    elif SPEECH_STATE and not test_result:  # speech to non-speech
        END_SAMPLE_INDEX = i * STEP_SIZE
        SPEECH_STATE = False
        sample_timed_result.append((START_SAMPLE_INDEX, END_SAMPLE_INDEX))

#print('sample segmentation results', sample_timed_result)
# combine if segments are too close AND not too long
processed_segmentation_result = []
MIN_SPEECH_LENGTH = 4800  # speech should be longer than 300ms
CONCAT_THRESHOLD = 4800  # concat segments that are shorter than this
MAX_SEG_LENGTH = 5 * 16000  # max segment length (5s), don't concat if result is longer than this
# i = 0
# while i < len(timed_results)-1:
#     this_result, next_result = timed_results[i], timed_results[i+1]
#     if next_result[0] - this_result[1] <= 3 and next_result[1] - this_result[0] <= 50:
#         # less than 300ms between them, and after combining less than 5s
#         new_results.append((this_result[0], next_result[1]))
#         if i == len(timed_results)-3:
#             new_results.append(timed_results[-1])
#             print('appended the last one to new results')
#         i += 2
#     else:  # normal case, did not combine
#         new_results.append(this_result)
#         if i == len(timed_results)-2:
#             new_results.append(next_result)
#             print('appended the last one to new results')
#         i += 1
if len(sample_timed_result) == 0:
    raise RuntimeError('no voice detected')

this_seg = sample_timed_result.pop(0)
for next_seg in sample_timed_result:
    # TODO: if segment is too long, pick a suitable point to cut it apart, using prediction prob
    #  see test_results, elements of it should be in the form (pred, [prob0, prob1])
    #  also shift the timings by 0.5 * WINDOWS_SIZE for better accuracy?

    # if 2 segments are very close
    if next_seg[0] - this_seg[1] <= CONCAT_THRESHOLD:
        # if not too long after concat, then join onto this_seg
        if next_seg[1] - this_seg[0] <= MAX_SEG_LENGTH:
            this_seg = (this_seg[0], next_seg[1])
        else:  # will be too long to concat, don't join, adjust this_seg's end timing to leave no gap
            this_seg = (this_seg[0], next_seg[0])
            processed_segmentation_result.append(this_seg)
            this_seg = next_seg
    else:  # cannot concat, append and assign to this_seg
        if this_seg[1] - this_seg[0] <= MIN_SPEECH_LENGTH:
            # too short and can't concat, consider as non-speech, skip it
            continue
        processed_segmentation_result.append(this_seg)
        this_seg = next_seg
processed_segmentation_result.append(this_seg)


#print('processed segmentation results', processed_segmentation_result)
sample_timed_result = list(map(format_tuple_to_time, processed_segmentation_result))
#print('sample timed result', sample_timed_result)
# with open('result.ass', 'w') as result_ass:
#     result_ass.write(ASS_TEMPLATE)
#     for t in sample_timed_result:
#         ass_line = f'\nDialogue: 0,{t[0]},{t[1]},侦探华生,,0,0,0,,'
#         result_ass.write(ass_line)

del model  # free memory
gc.collect()

model = nemo_asr.models.EncDecCTCModelBPE.restore_from('models/stt_en_citrinet_1024.nemo')
model.preprocessor.featurizer.dither = 0.0
model.preprocessor.featurizer.pad_to = 0
# Switch model to evaluation mode
model.eval()
# Freeze the encoder and decoder modules
model.encoder.freeze()
model.decoder.freeze()


@torch.no_grad()
def transcribe(audio, audio_length):
    # hypotheses = []
    # audio = torch.tensor(audio, device='cuda')
    # audio_length = torch.tensor()
    logits, logits_len, greedy_predictions = model.forward(input_signal=audio, input_signal_length=audio_length)
    current_hypotheses = model._wer.ctc_decoder_predictions_tensor(greedy_predictions, predictions_len=logits_len)
    # hypotheses += current_hypotheses
    # return hypotheses
    return ''.join(current_hypotheses)  # should be just one element, but just in case


transcriptions = []
for t in processed_segmentation_result:
    audio_batch = test_wav[t[0]:t[1]]
    audio_tensor = torch.tensor([audio_batch], device='cuda')
    length_tensor = torch.tensor([len(audio_batch)], device='cuda')
    transcriptions.append(transcribe(audio_tensor, length_tensor))

# prune out some stuff, make possible gibberish string empty, skip when writing to file
# TODO: potential problem: align it with processed segmentation result
_to_prune = ['^$', '\W+$', 'mm+', 'm\W+$', 'hm+\W*$', 'h\W+$', 'yeah\W+$', 'yeah\W*mm+', '\w\W*$']
patterns_to_prune = map(re.compile, _to_prune)
print('transcriptions length b4 pruning:', len(transcriptions))
transcriptions = list(map(lambda s: '' if any(map(lambda p: p.match(s), patterns_to_prune)) else s, transcriptions))
print('after:', len(transcriptions))

del model
gc.collect()

model = nemo_nlp.models.machine_translation.MTEncDecModel.restore_from('models/nmt_en_zh_transformer6x6.nemo').cuda()
model.eval()
translations = []
for batch in range(0, len(transcriptions), 64):
    transcription_batch = transcriptions[batch:batch+64]
    translations += model.translate(transcription_batch)


with open('result.ass', 'w') as result_ass:
    result_ass.write(ASS_TEMPLATE)
    for t, transcription, translation in zip(sample_timed_result, transcriptions, translations):
        if transcription == '':  # pruned stuff, or empty all along
            continue
        ass_line = f'\nDialogue: 0,{t[0]},{t[1]},侦探华生,,0,0,0,,{translation}; {transcription}'
        result_ass.write(ass_line)
