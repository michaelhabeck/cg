# cg #

This Python package implements methods for Bayesian coarse-graining of biomolecular structures. 

### Installation ###

* Local installation
```
python setup.py install --prefix=${HOME}
```

* Global installation
```
sudo python setup.py install
```

### Dependencies ###

* numpy: [download](https://pypi.python.org/pypi/numpy)

* scipy: [download](https://pypi.python.org/pypi/scipy)

* csb:   [download](https://pypi.python.org/pypi/csb)

Install with
```
pip install numpy scipy csb
```

Optional (used in only test and application scripts)

* matplotlib: [download](http://matplotlib.org)

Install with
```
pip install matplotlib
```

### Usage ###

Examples for running Bayesian coarse-graining on PDB structure can
be found in the *scripts/* folder.

Execute the script *cg_pdb_hmc.py*. 
