# SP-BAND
Implementation of Spectral Parameterization for the Broadband Analysis of Neural Data (SP-BAND), a work presented at the Asilomar Conference on Signals, Systems, and Computers on October 28, 2024.

This work is an extension of the spectral parameterization algorithm proposed by Donoghue _et al._ (2020) "Parameterizing neural power spectra into periodic and aperiodic components``. _Nature Neuroscience_, 23, 1655-1665. DOI: [10.1038/s41593-020-00744-x](https://doi.org/10.1038/s41593-020-00744-x).

The ```ParamSpectra``` class is built off the ```FOOOF``` class developed by Donoghue _et al._ which is available at [fooof-tools.github.io](https://fooof-tools.github.io/). The work in this ``SP-BAND`` package is not affiliated with the ```FOOOF``` package.

This work offers a data-driven approach to spectral parameterization that incorporates prior knowledge of expected periodic activity. As a result, this improves fit to real-world broadband data and improves the interpretability of the parameterized components.

## Installation
To use the SP-BAND package, first clone the repository:
```bash
git clone git@github.com:aroyphillips/SP-BAND.git
```

Then, navigate to the directory and install the conda environment:
```bash
cd SP-BAND
conda env create -f environment.yml
```

This will create the spband environment. To activate the environment, run:
```bash
conda activate spband
```

Finally, install the package:
```bash
pip install -e .
```

Now, you are ready to use the SP-BAND package!

## Usage
To use the SP-BAND, visit the [notebooks](notebooks) directory for a demonstration of the package.


## Repository Structure
The repository is structured as follows:
```
SP-BAND/
│
├── notebooks/  # Jupyter notebooks demonstrating the SP-BAND package
|   ├── 1.0.0-SP-BAND-Demonstration.ipynb
│
├── SPBAND/  # Source code for the SP-BAND package
│   ├── __init__.py
│   ├── spband.py
│   ├── utils.py
│
|-- data/  # Data used in the notebooks
│
|-- environment.yml  # Conda environment file
|
|-- LICENSE  # MIT License
|
|-- project.toml  # Project file
│
|-- README.md  # This README file
```
