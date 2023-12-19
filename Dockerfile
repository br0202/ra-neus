FROM nvcr.io/nvidia/pytorch:22.08-py3
WORKDIR /app
ARG DEBIAN_FRONTEND=noninteractive
RUN #apt-get -y install sudo dialog apt-utils
RUN echo 'debconf debconf/frontend select Noninteractive' |\
debconf-set-selections
RUN #apt-get install -y -q gedit
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install pytorch-lightning==1.8.6 \
    && pip install omegaconf \
    && pip install "opencv-python-headless<4.3" \
    && pip install torchtyping \
#    && pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
#    && !pip install "opencv-python-headless<4.3" \
#    && pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \


#    && pip install "opencv-python-headless<4.3" \
