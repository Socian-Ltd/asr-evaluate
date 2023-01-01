#!/usr/bin/env python
# coding: utf-8
  
import os
import sys
import glob
import requests
import Levenshtein as Lev
from tqdm import tqdm
import argparse
from diff_generator import diff_match_patch

dmp = diff_match_patch()

import speech_recognition as sr
r = sr.Recognizer()


input_dir = '/home/tareq/Downloads/testset/testset1_noisy_smartphone_single_speaker'
audio_extension = 'mp3'
url = 'http://alap.centralindia.cloudapp.azure.com:8084/transcribe/form/output'




def calculate_wer(hypothesis, ground_truth):
        """
        Computes the Word Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to words.
        Arguments:
            hypothesis (string): space-separated sentence
            ground_truth (string): space-separated sentence
        """
        # build mapping of words to integers
        b = set(hypothesis.split() + ground_truth.split())
        word2char = dict(zip(b, range(len(b))))

        # map the words to a char array (Levenshtein packages only accepts
        # strings)
        w1 = [chr(word2char[w]) for w in hypothesis.split()]
        w2 = [chr(word2char[w]) for w in ground_truth.split()]

        distance = Lev.distance(''.join(w1), ''.join(w2))
        num_words = len(ground_truth.split())
        return float(distance) / num_words

from data import data
def read_transcript(wav_path, audio_ext):
    text = ""
    file_name = wav_path.split(".")[0]
    file_name = file_name.split("/")[-1]
    try:
        text = data[file_name]
    except Exception as e:
        print(e)
    return text

def write_predicted_transcript(wav_path, text, audio_ext):
    transcript_path = wav_path.replace('.'+audio_ext, '-predicted.txt')
    with open(transcript_path, mode='w', encoding='utf8') as f:
        f.write(text)
    

def write_html_report(results, avg_wer,asr_from="Test"):
    base_dir = os.path.dirname(input_dir)
    html_path = os.path.join(base_dir, 'report-'+os.path.basename(input_dir)+'.html')
    print('Writing report to {}', html_path)
    print('Please open this file in your browser.')
    with open(html_path, mode='w', encoding='utf8') as f:
        f.write('<h3>{} Average WER on {} files: {:.2f}</h3><br><br>'.format(asr_from,len(results), avg_wer))
        for result in results:
            f.write("Audio: {} <br>".format(result['audio']))
            f.write("WER: {:.2f} <br>".format(result['WER']))
            f.write("Transcript: {} <br>".format(result['diff_html']))
            f.write('<hr style="height:2px;border-width:0;color:gray;background-color:gray"><br>')

def parser_socian_asr(file_path):
    post_file = {"file": open(file_path, 'rb')}
    payload = {'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c'}
    r = requests.post(url, files=post_file,data=payload)
    return r.json()['transcript']

def parser_google_asr(file_path):
    text = ""
    with sr.AudioFile(file_path) as source:
        audio_text = r.listen(source)
        try:
            text = r.recognize_google(audio_text, language="bn-BD")
            print(text)
        except Exception as e:
            print('Sorry.. run again...', str(e))

    return  text
if __name__ == '__main__':

    input_dir = "/home/tamzid/Desktop/socian/asr-evaluate/test/audio"#params.input_dir
    audio_extension = "wav"#params.audio_extension
    url = "https://devs-beta.socian.ai:8085/transcribe/form/output"#params.api_endpoint
    key = "Google" #google or socian

    audios = glob.glob(os.path.join(input_dir, '*.' + audio_extension))
    print("Total audio files found: ", len(audios))

    if len(audios) < 1:
        print('No audio found in dir: ', input_dir)
        print('Have you specified the correct directory? Or try setting --audio-extension param if your audios are not in wav format')
        sys.exit()

    total_wer = 0
    results = []
    for audio in tqdm(audios):
        try:
            print('\nProcessing file: ', audio)
            #change here
            if key == "Google":
                hyp_text = parser_google_asr(audio)
            else:
                key = "Socian"
                hyp_text = parser_socian_asr(audio)


            ground_truth = read_transcript(audio, audio_extension)
            write_predicted_transcript(audio, hyp_text, audio_extension)
            wer = calculate_wer(hyp_text, ground_truth)
            diff = dmp.diff_main(ground_truth, hyp_text)
            total_wer += wer
            results.append({
                'audio': os.path.basename(audio),
                'WER': wer,
                'ground_truth': ground_truth,
                'predicted': hyp_text,
                'diff_html': dmp.diff_prettyHtml(diff)
            })
            print('Predicted: ', hyp_text)
            print('Ground Truth: ', ground_truth)
            print('WER: ', wer)

        except Exception as e:
            print(e)
            print('Connection lost. Failed to transcribe ', audio)
        # break
    avg_wer = total_wer / len(audios)
    print("Avg WER on all files: ",avg_wer)

    write_html_report(results, avg_wer,asr_from=key)
