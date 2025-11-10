FROM continuumio/miniconda3

RUN conda create -y -n mfa -c conda-forge python=3.9 montreal-forced-aligner ffmpeg cmake make pkg-config spacy sudachipy sudachidict-core

RUN conda create -y -n app -c conda-forge python=3.9 ffmpeg

RUN conda run -n mfa mfa model download acoustic japanese_mfa
RUN conda run -n mfa mfa model download dictionary japanese_mfa

RUN conda run -n app pip install \
    flask \
    gunicorn \
    gtts \
    pandas \
    praatio \
    praat-parselmouth

WORKDIR /app
COPY . /app

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "app", \
     "gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--timeout", "3600", "app:app"]










