FROM tensorflow/tensorflow:2.13.0-gpu
WORKDIR /

COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    rm /requirements.txt

COPY builder/model_fetcher.py /model_fetcher.py
RUN python /model_fetcher.py
RUN rm /model_fetcher.py

ADD src .

CMD [ "python", "-u", "/rp_handler.py" ]