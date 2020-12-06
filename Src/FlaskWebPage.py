from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
import secrets, script
import os, sys, json
import azure.cognitiveservices.speech as speechsdk
import moviepy.editor as mp
import time

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
speech_key, service_region = secrets.speech_key, "westus"  # ADD THIS TO SECRETS!!
speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.request_word_level_timestamps()
speech_config.speech_recognition_language = "en-US"
speech_config.output_format = speechsdk.OutputFormat(1)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
audio_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "theaudio.wav")
resulting_text = ""
word_timestamps = []
videoFilePath = ""


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    global videoFilePath
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        videoFile = request.files['file']
        if videoFile.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if videoFile and allowed_file(videoFile.filename):
            filename = secure_filename(videoFile.filename)
            videoFilePath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            videoFile.save(videoFilePath)
            mp.VideoFileClip(videoFilePath).audio.write_audiofile(audio_filepath)
            return redirect(url_for('video_table_of_contents'))
    return 'ERROR'


@app.route('/videotableofcontents')
def video_table_of_contents():
    audio_input = speechsdk.audio.AudioConfig(filename=audio_filepath)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_input)
    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    def got_text(evt):
        global resulting_text
        global word_timestamps
        result = json.loads(evt.result.json)
        print('RECOGNIZEDOFFSET: {}'.format(evt.offset))
        print('RECOGNIZEDTEXT: {}'.format(evt.result.text))
        print('RECOGNIZEDJSON: {}'.format(result['NBest'][0]))
        for offset_obj in result['NBest'][0]['Words']:
            word_timestamps.append(offset_obj['Offset'])
        resulting_text += evt.result.text

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(got_text)
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)

    speech_recognizer.stop_continuous_recognition()
    result_tuple = script.main(resulting_text, word_timestamps)
    return render_template('videotableofcontents.html', result=result_tuple)


@app.route('/')
def home():
    return render_template('home.html')


app.run()
