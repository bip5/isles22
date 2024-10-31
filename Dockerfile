FROM nvcr.io/nvidia/pytorch:22.06-py3
#Above sets the base image for Docker image

#Create a new group called algorithm, useradd -m to create home directory,no-log-init to disable login initialisation, -r create system user,-g algorithm assigns user to algorithm, algorithm is the username
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

#create directories and allocate ownership to 'algorithm' group's 'algorithm' user
RUN mkdir -p /opt/algorithm /input /output /opt/algorithm/weights \
    && chown algorithm:algorithm /opt/algorithm /input /output /opt/algorithm/weights

# Rest of the commands will run as 'algorithm' is the user
USER algorithm

# change the working directory to opt/algorithm
WORKDIR /opt/algorithm

# add .local/bin directory to system's PATH variable
ENV PATH="/home/algorithm/.local/bin:${PATH}"

# Run pip in python install it for the user and upgrade 
RUN python -m pip install --user -U pip

RUN wget -O /opt/algorithm/weights/2024-10-20SegResNetDS_j7953765_ts0 "https://filebin.net/a52m71y2fpx32jd8"

COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
RUN python -m pip install --user -r /opt/algorithm/requirements.txt

COPY --chown=algorithm:algorithm isles22.py /opt/algorithm/



ENTRYPOINT ["python", "-m", "isles22"]
