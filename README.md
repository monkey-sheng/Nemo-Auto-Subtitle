# Nemo-Auto-Subtitle
Using Nvidia NeMo to auto create subtitles, many thanks to @[RaviSoji](https://github.com/RaviSoji) for the PLDA!

### How it works

Basically, do VAD and find the timings, perform STT then NMT. However, the VAD from Nemo doesn’t seem to give satisfactory results. Instead a diarization model is used to extract the embeddings, which then calls for the use of PLDA to predict speech segments from those.

**Need to put relevant data into `plda_data/voice`and `plda_data/background`.**



Note: Nemo doesn’t work well on Windows due to pyannote’s multiprocessing. Modify the \__init__.py of pyannote.metrics (commented out this line). Seems to work without issue.

```python
# manager_ = Manager()
```

### Usage:

TODO, whole working thing is in `speakernet_playground.py` at the moment.

