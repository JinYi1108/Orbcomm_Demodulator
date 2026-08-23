# LFdemod
[![DOI](https://zenodo.org/badge/1130552672.svg)](https://doi.org/10.5281/zenodo.18214211)

**LFdemod** is a Python toolkit for offline demodulation and diagnostic
processing of low-frequency raw-voltage data. It began as an **ORBCOMM**
satellite downlink demodulator for **21CMA (21 Centimeter Array)** recordings
and is being extended with a broadcast-FM processing path.

The project name is written `LFdemod`. Its terminal command and new public
Python package are both lowercase: `lfdemod`. The historical `orbdemod`
package remains available as a compatibility layer for existing ORBCOMM code.

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

#### Legacy ORBCOMM release from PyPI
The earlier ORBCOMM-only release remains available as:
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

After the editable install, inspect the current command interface with:

```bash
lfdemod --help
lfdemod fm --help
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

---

## Explicit DDC filtering and decimation

The shared DDC filter stages are specified by measurable passband, stopband,
ripple, and attenuation requirements instead of fixed cutoff ratios. The
ORBCOMM-specific plan and complete voltage-to-IQ pipeline live in
`orbdemod.orbcomm.orbcomm_ddc`. The historical top-level
`ddc(data, freq_lo, fs_in, fs_mid, fs_out)` alias remains valid.

At the default ORBCOMM rates, the anti-alias path is now:

```text
480 MS/s -> 2.4 MS/s -> 96 kS/s -> 9.6 kS/s
     /200          /25          /10
```

This replaces the old direct 250-fold second-stage reduction, whose fixed
201-tap FIR did not adequately suppress the first alias band. The default
ORBCOMM passband is 4.0 kHz, the final stopband begins at 4.8 kHz, and the
requested stopband attenuation is 60 dB. These acquisition-band parameters
should be re-evaluated against real Doppler and receiver-frequency error data.

Legacy `decimate_iir` and `decimate_fir` helpers remain available for existing
callers, but new code should use `DecimationStageConfig` together with
`filter_and_decimate`.

Butterworth and Kaiser filter generation defaults to `verify=False`, so normal
data processing does not repeatedly calculate a dense frequency response. Set
`verify=True` when a newly generated filter should be checked against its
configured passband loss/ripple and stopband attenuation. IIR pole stability
and finite coefficients are checked as well, and a failed check raises a
`ValueError`:

```python
from orbdemod.ddc import (
    design_kaiser_lowpass,
    format_filter_report,
    validate_filter_response,
)
from orbdemod.orbcomm.orbcomm_ddc import make_orbcomm_decimation_stages

stage = make_orbcomm_decimation_stages()[-1]
taps = design_kaiser_lowpass(stage)
report = validate_filter_response(taps, stage)
print(format_filter_report(report))
```

When only pass/fail enforcement is needed, use
`design_kaiser_lowpass(stage, verify=True)` instead.

When a Butterworth stage is applied with zero-phase `sosfiltfilt`, verification
uses `processing_passes=2`, so the reported requirements describe the actual
two-pass response rather than a single forward pass.

---

## Experimental broadcast-FM PSD path

For a parameter-by-parameter explanation, automatic output naming rules, and
Python API examples, see [FM 功能使用说明](docs/FM使用说明.md).

The `fm-dev` branch contains an offline, deliberately shallow FM path:

```text
480 MS/s int16 real voltage
  -> channel-selecting DDC
  -> 240 kS/s complex IQ
  -> quadrature FM discriminator
  -> FM composite (MPX) PSD
```

It marks the 0--15 kHz audio band, 19 kHz stereo pilot, 23--53 kHz
stereo-difference band (centred on the suppressed 38 kHz subcarrier), and
57 kHz RDS region. It does not automatically classify a window as broadcast
FM.

After an editable install, analyze one window with the lowercase command:

```bash
lfdemod fm \
  --input /path/to/20250415-1940-0.dat \
  --rf-frequency 100.1e6 \
  --start 0.0 \
  --duration 0.5 \
  --label direct_100p1
```

Without `--output-dir`, this run is saved automatically under:

```text
results/20250415-1940-0/100.1MHz/0_0.5/
```

Use `--output-root /another/root` to change only the automatic root. Use
`--output-dir /exact/path` to supply the output-directory base explicitly.

The third plot shows 50 ms from the centre of the selected analysis window by
default. To choose it explicitly, use `--waveform-start` (relative to the
selected analysis window, not to the raw file) and `--waveform-duration`:

```bash
lfdemod fm \
  --input /path/to/20250415-1940-0.dat \
  --rf-frequency 100.1e6 \
  --start 3.0 \
  --duration 0.5 \
  --waveform-start 0.12 \
  --waveform-duration 0.03 \
  --output-root results
```

This plots 120--150 ms within the 0.5 s analysis window, corresponding to
3.120--3.150 s in the original file. The first two PSD plots still use the
whole 0.5 s analysis window.

Alternatively, select a window by file fraction (the two window modes are
mutually exclusive):

```bash
lfdemod fm \
  --input /path/to/20250415-1940-0.dat \
  --rf-frequency 100.1e6 \
  --start-fraction 0.25 \
  --stop-fraction 0.30
```

Automatic names use the resolved start and duration in seconds. Numeric names
drop unnecessary trailing zeros, so 98.3 MHz and 98.33 MHz become `98.3MHz`
and `98.33MHz`, while a 30-second start and 3-second duration become `30_3`.

Existing directories are never silently overwritten. A repeated run appends
`_run02`, then `_run03`, and so on. Pass `--overwrite` only when the fixed
result files in the selected base directory should be replaced. This collision
rule also applies when `--output-dir` is supplied explicitly.

The default high-rate processing block is 10,000,000 input samples and remains
configurable with `--chunk-samples`. Real-data benchmarks at 2M, 5M, 10M, and
20M samples should be used before fixing a long-term default.

The output directory contains `fm_psd.png`, `fm_arrays.npz`, `summary.json`,
and `run_config.json`.

`python examples/analyze_fm_psd.py ...` remains as a thin compatibility
wrapper and accepts the same FM options. New usage should prefer `lfdemod fm`.
