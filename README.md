# fragmentomics_tools


## Install

```
conda create -n test_fragmentomics_tools
conda activate test_fragmentomics_tools
conda install mamba
mamba install jupyterlab pytest
```

# install fragments_h5
```
git clone git@github.com:nboley/fragments_h5.git
cd fragments_h5
pip install -e .
```

# install data manifest
```
git clone git@github.com:nboley/datamanifest.git
cd datamanifest
pip install -e .
```

# install fragmentomics_tools
```
git clone git@github.com:nboley/fragmentomics_tools.git
cd fragmentomics_tools
mamba install --verbose --override-channels -c pytorch -c conda-forge -c bioconda --file requirements.in
python setup.py develop
```
