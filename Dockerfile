FROM pytorch/pytorch:latest

# Update
RUN apt-get update \
     && apt-get install -y libgl1-mesa-glx libx11-xcb1 \
     && apt-get clean all \
     && rm -r /var/lib/apt/lists/*

# Install Python libraries
RUN /opt/conda/bin/conda install --yes \
    numpy=1.20.1 \
    matplotlib \
    scikit-learn \
    gensim \
    pillow \
    h5py

# Install OpenJDK-11 to compute evaluation metrics
RUN apt-get update && \
    apt-get install -y openjdk-11-jre-headless && \
    apt-get clean;