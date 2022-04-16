module purge all

module load python/anaconda3.6

conda create --name celltrack python=3.7.11 --yes

source activate celltrack

conda install -y -c anaconda git
conda install -y -c conda-forge screen
conda install -y -c conda-forge htop
conda install -y -c conda-forge matplotlib=3.5.0
conda install -y -c conda-forge pyyaml=6.0
conda install -y -c conda-forge scikit-learn=1.0.1
conda install -y -c conda-forge tqdm=4.62.3
conda install -y -c fastai opencv-python-headless=4.5.3.56
conda install -y pandas=1.3.4

pip install --no-input pyradiomics==3.0.1
pip install --no-input scikit-image==0.19.1
