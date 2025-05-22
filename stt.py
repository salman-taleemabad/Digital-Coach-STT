from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("ihanif/whisper-medium-urdu")
model = AutoModelForSpeechSeq2Seq.from_pretrained("ihanif/whisper-medium-urdu")