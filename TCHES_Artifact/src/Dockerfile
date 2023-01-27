FROM python:3.10

WORKDIR /home/tches/TCHES_Artifact/src/

# COPY requirements.txt .

# RUN pip install -r requirements.txt

ADD . .

WORKDIR /home/tches/TCHES_Artifact/src/Distinguish_Class_Pairs/

# COPY ./Distinguish_Class_Pairs/Collect_inference_timings.py .

RUN python Collect_inference_timings.py -m custom_cnn