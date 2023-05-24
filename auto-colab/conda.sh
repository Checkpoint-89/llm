# Run this script to install miniconda on a colab VM
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local/miniconda
rm Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
export PATH=/usr/local/miniconda/bin:$PATH