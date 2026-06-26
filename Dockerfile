FROM nvidia/cuda:13.2.1-cudnn-runtime-ubuntu24.04

COPY . /riskliver
WORKDIR /riskliver

RUN apt-get update && \
apt-get install -y curl tar bzip2 libgomp1 libxrender1 libxext6 libexpat1 unzip xz-utils && \
mkdir MATLAB_inst && cd MATLAB_inst && \
unzip /riskliver/MATLAB_Runtime_R2022a_Update_9_glnxa64.zip && \
./install -mode silent -agreeToLicense yes && \
cd /riskliver && \
tar -xf /riskliver/OPERA2.9_CL_Par.tar.xz && \
OPERA2.9_CL_Par/OPERA2_CL_Par/OPERA_P_2.9_mcr_Installer.install -mode silent -agreeToLicense yes && \
cd /root && \
curl -Ls https://micro.mamba.pm/api/micromamba/$(uname)-$(uname -m)/latest | tar -xvj bin/micromamba && \
chmod +x bin/micromamba && \
bin/micromamba env create -f /riskliver/env.yml -y

# mkdir -p /var/run/sshd 
# mkdir /run/sshd && \
# ssh-keygen -A && \
# sed -i 's/#Port 22/Port 8022/' /etc/ssh/sshd_config

EXPOSE 6310
# ENTRYPOINT ["blrec", "--host", "0.0.0.0", "--no-progress"]
CMD /root/bin/micromamba run -n riskliver --cwd /riskliver python riskliver_server.py