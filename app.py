from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import sys
from gtts import gTTS
import uuid
import shutil
import subprocess
import random
import pandas as pd
import parselmouth
from praatio import textgrid
from comparison import compare_infos
app = Flask(__name__)


LEXICON = "japanese_mfa"
MFA_MODEL = "japanese_mfa"
UPLOAD_FOLDER = "/tmp/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/health")
def health_check():
    return jsonify({"status": "healthy"}), 200

vocab_df = pd.read_csv("genki1vocab.csv")

@app.route("/debug_tmp/<session_id>")
def debug_tmp(session_id):
    files = os.listdir(f"/tmp/uploads/{session_id}")
    return {"tmp_files": files}

@app.route("/random_word")
def random_word():
    row = vocab_df.sample(1).iloc[0]
    display_word = row["kanji"] if pd.notna(row["kanji"]) and row["kanji"].strip() != "" else row["hiragana"]

    return jsonify({
        "hiragana": row["hiragana"],
        "kanji_or_hiragana": display_word,
        "meaning": row["meaning"],
        "chapter": int(row["chapter"])
    })


@app.route("/align", methods=["POST"])
def align():

    audio_file = request.files["audio"]
    target_word = request.form.get("target_word")

    session_id = str(uuid.uuid4())[:8]
    session_path = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(session_path, exist_ok=True)

    user_wav = os.path.join(session_path, "user.wav")
    convert_to_wav(audio_file, user_wav)

    user_lab = os.path.join(session_path, "user.lab")
    make_lab(target_word, user_lab)

    user_textgrid = os.path.join(session_path, "user.TextGrid")


    ref_wav = os.path.join(session_path, "ref.wav")
    text_to_wav(target_word, ref_wav)


    ref_lab = os.path.join(session_path, "ref.lab")
    make_lab(target_word, ref_lab)

    ref_textgrid = os.path.join(session_path, "ref.TextGrid")

    run_mfa(session_path, LEXICON, MFA_MODEL, session_path)

    diff_summary = compare_infos(user_wav, user_textgrid, ref_wav, ref_textgrid)

    random_score = random.randint(1, 100)
    feedback = diff_summary

    shutil.rmtree(session_path, ignore_errors=True)


    return jsonify({
        "status": "ok",
        "feedback": feedback
    })

@app.route("/download/<session_id>/<filename>")
def download_file(session_id, filename):
    file_path = os.path.join(UPLOAD_FOLDER, session_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404


def text_to_wav(text, output_file):
    tts = gTTS(text, lang="ja")
    tts.save(output_file)


def make_lab(word, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(word + "\n")


def run_mfa(corpus_dir, dict_path, model_path, output_dir):

    mfa_cmd = [
        "conda", "run", "-n", "mfa", "mfa", "align",
        corpus_dir,
        dict_path,
        model_path,
        output_dir
    ]

    print(f"Executing MFA command: {' '.join(mfa_cmd)}", file=sys.stdout) # Diagnostic line

    try:
        result = subprocess.run(
            mfa_cmd,
            check=True,  
            capture_output=True,
            text=True
        )

        print("\n--- MFA STDOUT (Success) ---\n", result.stdout, file=sys.stdout)
        return True

    except subprocess.CalledProcessError as e:
        print("\n--- MFA SUBPROCESS FAILED! ---", file=sys.stderr)
        print(f"Command: {e.cmd}", file=sys.stderr)
        print(f"Return Code: {e.returncode}", file=sys.stderr)

        print("\n--- MFA STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- MFA STDERR ---", file=sys.stderr)
        print(e.stderr, file=sys.stderr)



def convert_to_wav(file_storage, output_path):
    temp_input = output_path + ".tmp"
    file_storage.save(temp_input)

    command = [
        "ffmpeg",
        "-y",
        "-i", temp_input,
        "-ac", "1",
        "-ar", "16000",
        "-sample_fmt", "s16",
        output_path
    ]

    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode != 0:
        print("FFmpeg error:", process.stderr)
        raise RuntimeError("Audio conversion failed")

    os.remove(temp_input)
    return output_path



















