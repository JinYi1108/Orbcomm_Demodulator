# Orbcomm_Demodulator 
[![PyPI version](https://badge.fury.io/py/orbdemod.svg)](https://badge.fury.io/py/orbdemod)
[![DOI](https://zenodo.org/badge/1130552672.svg)](https://doi.org/10.5281/zenodo.18214211)

**Orbcomm_Demodulator** is a Python-based toolkit for demodulating **ORBCOMM** satellite downlink signals.  
This project was originally developed to process **raw voltage data captured by the 21CMA (21 Centimeter Array)** radio telescope, and aims to provide a **complete, standardized processing pipeline** from raw samples to decoded ORBCOMM data packets.

---

## Core Pipeline Features
The demodulation chain includes:

* **Digital Down Conversion (DDC)** for raw high-rate samples.
* **Symbol Timing Recovery** using Gardner TED and Farrow interpolator.
* **Carrier Phase Recovery** via a 2nd-order Costas Loop.
* **Differential Decoding** and **Frame Synchronization**.
* **Fletcher-16 Checksum** with 1-bit Error Correction (ECC).

---

## Installation

####  From PyPI (Recommended)
You can install the stable version directly via `pip`:
```bash
pip install orbdemod
```

#### From GitHub
You can install the package directly from GitHub using:

```bash
pip install git+https://github.com/JinYi1108/Orbcomm_Demodulator.git
```

#### From Source (For Development)
For development and debugging, an editable installation is recommended:
```bash
git clone https://github.com/JinYi1108/Orbcomm_Demodulator.git
cd Orbcomm_Demodulator
pip install -e .
```
---
## Examples and Test Data

Example scripts are provided in the `examples/` directory. Due to storage limits, large raw datasets are hosted on Zenodo.

#### 1. Raw Voltage Data (21CMA)
- **File**: `20250415-1940-0.dat` (approx. 8 GB)
- **Source**: Raw voltage data recorded by the **21CMA** radio telescope.
- **Download**: [Available on Zenodo (DOI: 10.5281/zenodo.18213739)](https://zenodo.org/records/18213739)

#### 2. DDC Processed IQ Data (21CMA)
- **File**: `raw_data/iq_data_0.dat`
- **Description**: This is the IQ data generated from the raw 21CMA `.dat` file after Digital Down Conversion (DDC). It is provided for quick testing of the backend stages (STR, Costas, etc.).

#### 3. Test Data (from ORBCOMM-receiver)
- **File**: `raw_data/1552071892p6.dat`
- **Description**: Derived from the original `1552071892p6.mat` from Frank Bieberly's project. It has been downsampled and filtered to a baseband IQ format.


![alt text](image.png)

---

## Acknowledgements

This project is developed based on **ORBCOMM-receiver** project by **Frank Bieberly** (https://github.com/fbieberly/ORBCOMM-receiver).  
