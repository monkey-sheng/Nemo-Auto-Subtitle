import pickle
from dataclasses import dataclass
from os import listdir
from datetime import timedelta
from subprocess import run

import pysubs2
from google.cloud import storage, speech

files = listdir()
videos = [f for f in files if f.endswith('.mp4')]
subs = [f for f in files if f.endswith('.ass')]
if len(videos) == 0 or len(subs) == 0:
    from time import sleep
    from sys import exit

    print('no .mp4 and/or .ass found. Quitting')
    sleep(5)
    exit(1)

video, ass = videos[0], subs[0]

# genereate wav file from video
run(['ffmpeg', '-y', '-i', video, '-ac', '1', 'out.wav'])

# upload to GCS
BUCKET_NAME = "subtitle-files"
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob("out.wav")
blob.upload_from_filename("out.wav")

AUDIO_FILE_URI = "gs://subtitle-files/out.wav"  # use in stt API

speech_client = speech.SpeechClient()
audio = speech.RecognitionAudio(uri=AUDIO_FILE_URI)
# maybe try use_enhanced=True below
config = speech.RecognitionConfig(language_code='en-US', enable_word_time_offsets=True, enable_automatic_punctuation=True, model='video')
operation = speech_client.long_running_recognize(config=config, audio=audio)

print("Waiting for google tts to complete...")
# Each result is for a consecutive portion of the audio. Iterate through
# them to get the transcripts for the entire audio file.
results = operation.result().results

# CAN'T PICKLE THIS
###pickle.dump(results, open('result-dump', 'wb'))

# all_words = []
# word_dict_list = []
#
# for result in results:
#     for word in result.alternatives[0].words:
#         all_words.append(word)
#
# for word in all_words:
#     # does not have 'confidence' in response
#     word_dict = {'start_time': word.start_time, "end_time": word.end_time, "word": word.word, "speaker_tag": word.speaker_tag}
#     word_dict_list.append(word_dict)
# pickle.dump(word_dict_list, open('result-dict-dump', 'wb'))


###results = pickle.load(open('result-dict-dump', 'rb'))


@dataclass
class Word:
    start: int
    end: int
    word: str


def post_proc(word):
    start = word['start_time'] // timedelta(milliseconds=1)
    end = word['end_time'] // timedelta(milliseconds=1)
    return Word(start, end, word['word'])


def marshall_words_from_results(results):
    words = []

    def to_word(w):
        return Word(w.start_time // timedelta(milliseconds=1),
                    w.end_time // timedelta(milliseconds=1),
                    w.word)
    for result in results:
        words.extend(map(to_word, result.alternatives[0].words))
    return words


words = marshall_words_from_results(results)
# dump the words list for debug purposes
pickle.dump(words, open('words-dump', 'wb'))

###words = list(map(post_proc, results))
sub = pysubs2.load(ass)

word_idx = 0

for line in sub:
    words_this_line = []
    while word_idx < len(words):
        word = words[word_idx]
        word_duration = word.end - word.start

        # word ends within line, check word's start time
        if word.end < line.end:
            # 50% of word within the line?
            # somehow there can be words starting and ending at the same time
            if word_duration == 0 or (word.end - line.start) / word_duration > 0.5:
                words_this_line.append(word.word)
            # else the word is mostly before the line started, should be handled by
            # the previous line, Or might be transcribed by mistake, ignore,
            # and increment index regardless
            word_idx += 1

        # word ends after the line, check word's start time
        else:
            # check word is mostly inside the line
            if word_duration != 0 and (line.end - word.start) / word_duration > 0.5:
                words_this_line.append(word.word)
            # word ends after the line and is mostly outside, signifies last word of line
            # don't include this word in this line, break loop go for next line
            else:
                break
            word_idx += 1

        line.text = " ".join(words_this_line)

sub.save('processed.ass')
