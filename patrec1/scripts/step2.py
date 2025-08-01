import os
import librosa
import re


def text_to_number(text):
 
  number_map = { 'zero': 0,'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

  if  text.lower() in number_map:
    return number_map[text.lower()]
  else:
    return None

def data_parser(data_dir = '/home/arismarkog/patrec1/pr_lab2_data/digits'):
    wav_data = []
    speakers = []
    digits = []

    # Regular expression to capture speaker and digit from filename
    pattern = re.compile("([a-zA-Z]+)([0-9]+)")

    for filename in os.listdir(data_dir):

        if filename.endswith('.wav'):

            filepath = os.path.join(data_dir, filename)

            # Load the audio file with librosa
            wav, _ = librosa.load(filepath, sr=None)
            wav_data.append(wav)

            match = pattern.match(filename)
            if match:
                speaker = match.group(2)
                digit = match.group(1)
                speakers.append(speaker)
                digits.append(text_to_number(digit))

    return wav_data, speakers, digits
