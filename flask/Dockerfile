FROM python:3.6-jessie

MAINTAINER Lewis Brown "github.com/ramity"

ADD . /var/flask
WORKDIR /var/prototypes

RUN apt-get update
RUN apt-get install -y cmake

RUN git clone https://github.com/danbloomberg/leptonica /var/leptonica
RUN cd /var/leptonica && ./autogen.sh && ./configure && make && make install && ldconfig

RUN apt-get install -y ghostscript libtool libpng-dev libjpeg62-turbo-dev libtiff5-dev zlib1g-dev libicu-dev libpango1.0-dev libcairo2-dev

RUN git clone https://github.com/tesseract-ocr/tesseract.git /var/tesseract-ocr
RUN cd /var/tesseract-ocr && ./autogen.sh && ./configure && make && make install && ldconfig
RUN cd /usr/local/share && git clone https://github.com/tesseract-ocr/tessdata_best.git tessdata-best

RUN python -m pip install -r requirements.txt

# FOR TRAINING IN THE FUTURE: https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract-4.00

CMD tail -f /dev/null
