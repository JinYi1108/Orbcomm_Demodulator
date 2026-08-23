"""单文件、顺序执行的广播 FM 下变频、鉴频和功率谱绘图代码。

使用方法：只修改“用户参数”区域，然后直接运行：

    python examples/fm_psd_one_file.py

这个文件不定义自建函数或类，也不依赖 orbdemod.fm 子模块。
"""

from pathlib import Path
import json
import os
import tempfile

import numpy as np
from scipy import signal


# =============================================================================
# 用户参数：真实数据测试时主要修改这里
# =============================================================================

# 原始电压文件。请改成服务器上 20250415-1940-0.dat 的真实路径。
INPUT_FILE = Path("/path/to/20250415-1940-0.dat")

# 需要下变频到零频的目标 FM 电台频率，例如 98.3 MHz。
RF_FREQUENCY_HZ = 98.3e6

# 窗口选择方式："seconds" 按秒，"fraction" 按文件总长度比例。
WINDOW_MODE = "seconds"

# 按秒模式：从文件第几秒开始，以及处理多长时间。
START_SECONDS = 0.0
DURATION_SECONDS = 0.2

# 按比例模式：例如 0.25--0.30 表示读取文件的 25% 到 30%。
START_FRACTION = 0.25
STOP_FRACTION = 0.30

# 输出位置。None 表示在 OUTPUT_ROOT 下自动按“文件名/频率/时间段”命名；
# 也可以填写 Path("/指定/文件夹")，直接使用自己给出的路径作为基准。
OUTPUT_DIR = None
OUTPUT_ROOT = Path("./results")
OVERWRITE = False
LABEL = "FM 98.3 MHz direct test"

# 21CMA 原始电压格式和采样率。
INPUT_DTYPE = "<i2"       # 小端 int16
FS_IN = 480e6             # 480 MS/s 原始实电压
FS_MID = 2.4e6            # 第一级降到 2.4 MS/s
FS_OUT = 240e3            # 最终复数 IQ 为 240 kS/s

# FM 信道滤波要求。
PASSBAND_HZ = 90e3
STOPBAND_HZ = 120e3
STOPBAND_ATTENUATION_DB = 60.0
FIRST_STAGE_PASSBAND_RIPPLE_DB = 0.25

# 分块大小和滤波边缘留白。
CHUNK_SAMPLES = 10_000_000
PADDING_SECONDS = 0.020

# 第三张图显示分析时间段中的哪一小段。
# None 表示自动截取分析时间段正中央；也可以填 0.10 表示从分析段内
# 第 0.10 秒开始画。这个起点不是原始文件中的绝对时间。
WAVEFORM_START_SECONDS = None
WAVEFORM_DURATION_SECONDS = 0.050


# =============================================================================
# 检查参数并读取指定时间段
# =============================================================================

INPUT_FILE = INPUT_FILE.expanduser().resolve()

if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        "找不到输入文件，请修改 INPUT_FILE：{}".format(INPUT_FILE)
    )
if not 0 < RF_FREQUENCY_HZ < FS_IN / 2.0:
    raise ValueError("RF_FREQUENCY_HZ 必须位于 0 和 FS_IN/2 之间。")
if not 0 < PASSBAND_HZ < STOPBAND_HZ <= FS_OUT / 2.0:
    raise ValueError("需要满足 0 < PASSBAND_HZ < STOPBAND_HZ <= FS_OUT/2。")
if WAVEFORM_START_SECONDS is not None and WAVEFORM_START_SECONDS < 0:
    raise ValueError("WAVEFORM_START_SECONDS 不能为负。")
if WAVEFORM_DURATION_SECONDS <= 0:
    raise ValueError("WAVEFORM_DURATION_SECONDS 必须大于0。")

DECIMATION_1 = int(round(FS_IN / FS_MID))
DECIMATION_2 = int(round(FS_MID / FS_OUT))
if not np.isclose(FS_IN / FS_MID, DECIMATION_1, rtol=0.0, atol=1e-9):
    raise ValueError("FS_IN / FS_MID 必须是整数。")
if not np.isclose(FS_MID / FS_OUT, DECIMATION_2, rtol=0.0, atol=1e-9):
    raise ValueError("FS_MID / FS_OUT 必须是整数。")

dtype = np.dtype(INPUT_DTYPE)
raw = np.memmap(INPUT_FILE, dtype=dtype, mode="r")
file_duration_seconds = len(raw) / FS_IN
if WINDOW_MODE == "seconds":
    if START_SECONDS < 0 or DURATION_SECONDS <= 0:
        raise ValueError("START_SECONDS 不能为负，DURATION_SECONDS 必须大于0。")
    requested_start_seconds = START_SECONDS
    requested_duration_seconds = DURATION_SECONDS
elif WINDOW_MODE == "fraction":
    if not 0.0 <= START_FRACTION < STOP_FRACTION <= 1.0:
        raise ValueError("需要满足 0 <= START_FRACTION < STOP_FRACTION <= 1。")
    requested_start_seconds = START_FRACTION * file_duration_seconds
    requested_duration_seconds = (
        STOP_FRACTION - START_FRACTION
    ) * file_duration_seconds
else:
    raise ValueError('WINDOW_MODE 必须是 "seconds" 或 "fraction"。')

requested_stop_seconds = requested_start_seconds + requested_duration_seconds
if requested_stop_seconds > file_duration_seconds:
    raise ValueError(
        "请求处理到 {:.6f} 秒，但文件总时长只有 {:.6f} 秒。".format(
            requested_stop_seconds,
            file_duration_seconds,
        )
    )

# 自动目录示例：results/20250415-1940-0/98.3MHz/30_3/。
# 数字最多保留六位小数，并去掉没有意义的末尾零。
if OUTPUT_DIR is None:
    frequency_name = (
        "{:.6f}".format(RF_FREQUENCY_HZ / 1e6).rstrip("0").rstrip(".")
        + "MHz"
    )
    start_name = (
        "{:.6f}".format(requested_start_seconds).rstrip("0").rstrip(".")
    )
    duration_name = (
        "{:.6f}".format(requested_duration_seconds).rstrip("0").rstrip(".")
    )
    output_base_dir = (
        OUTPUT_ROOT.expanduser()
        / INPUT_FILE.stem
        / frequency_name
        / "{}_{}".format(start_name, duration_name)
    ).resolve()
else:
    output_base_dir = Path(OUTPUT_DIR).expanduser().resolve()

OUTPUT_DIR = output_base_dir
if OUTPUT_DIR.exists() and not OVERWRITE:
    run_number = 2
    while True:
        candidate = output_base_dir.with_name(
            "{}_run{:02d}".format(output_base_dir.name, run_number)
        )
        if not candidate.exists():
            OUTPUT_DIR = candidate
            break
        run_number += 1

OUTPUT_DIR.mkdir(parents=True, exist_ok=OVERWRITE)

padded_start_seconds = max(0.0, requested_start_seconds - PADDING_SECONDS)
padded_stop_seconds = min(
    file_duration_seconds,
    requested_stop_seconds + PADDING_SECONDS,
)
start_sample = int(round(padded_start_seconds * FS_IN))
stop_sample = int(round(padded_stop_seconds * FS_IN))
raw_window = raw[start_sample:stop_sample]

print("输入文件：", INPUT_FILE)
print("输出目录：", OUTPUT_DIR)
print("文件总时长：{:.6f} s".format(file_duration_seconds))
print("目标频率：{:.6f} MHz".format(RF_FREQUENCY_HZ / 1e6))
print(
    "处理时间：{:.6f}--{:.6f} s".format(
        requested_start_seconds,
        requested_stop_seconds,
    )
)
print("原始采样点数（含padding）：", len(raw_window))


# =============================================================================
# 第一级 DDC：混频到零频、低通滤波、480 MS/s -> 2.4 MS/s
# =============================================================================

first_order, first_critical_hz = signal.buttord(
    wp=PASSBAND_HZ,
    ws=FS_MID / 2.0,
    gpass=FIRST_STAGE_PASSBAND_RIPPLE_DB,
    gstop=STOPBAND_ATTENUATION_DB,
    fs=FS_IN,
)
first_sos = signal.butter(
    first_order,
    first_critical_hz,
    btype="lowpass",
    fs=FS_IN,
    output="sos",
)
first_filter_state = np.zeros((first_sos.shape[0], 2), dtype=np.complex128)

# 让本振相位对应原始文件中的绝对起始采样点。
phase_step = 2.0 * np.pi * RF_FREQUENCY_HZ / FS_IN
phase = np.remainder(phase_step * start_sample + np.pi, 2.0 * np.pi) - np.pi
input_count = 0
stage_1_blocks = []

for block_start in range(0, len(raw_window), CHUNK_SAMPLES):
    block_stop = min(block_start + CHUNK_SAMPLES, len(raw_window))
    # 转换为浮点数，但保留原始ADC计数尺度。
    voltage = np.asarray(raw_window[block_start:block_stop], dtype=np.float32)

    sample_index = np.arange(len(voltage), dtype=np.float64)
    block_phase = phase + phase_step * sample_index
    local_oscillator = np.exp(-1j * block_phase)
    mixed = voltage * local_oscillator
    phase = np.remainder(
        phase + phase_step * len(voltage) + np.pi,
        2.0 * np.pi,
    ) - np.pi

    filtered, first_filter_state = signal.sosfilt(
        first_sos,
        mixed,
        zi=first_filter_state,
    )

    # 保证分块边界前后的抽取网格连续。
    first_output_index = (-input_count) % DECIMATION_1
    stage_1_blocks.append(filtered[first_output_index::DECIMATION_1])
    input_count += len(voltage)

stage_1_iq = np.concatenate(stage_1_blocks)
print("第一级输出：{} samples @ {:.3f} MS/s".format(len(stage_1_iq), FS_MID / 1e6))


# =============================================================================
# 第二级 DDC：90--120 kHz Kaiser FIR，2.4 MS/s -> 240 kS/s
# =============================================================================

normalized_transition_width = (STOPBAND_HZ - PASSBAND_HZ) / (FS_MID / 2.0)
design_attenuation_db = STOPBAND_ATTENUATION_DB + 3.0
second_num_taps, second_beta = signal.kaiserord(
    design_attenuation_db,
    normalized_transition_width,
)
second_num_taps = max(3, int(second_num_taps))
if second_num_taps % 2 == 0:
    second_num_taps += 1

second_cutoff_hz = 0.5 * (PASSBAND_HZ + STOPBAND_HZ)
second_fir = signal.firwin(
    second_num_taps,
    second_cutoff_hz,
    window=("kaiser", second_beta),
    fs=FS_MID,
)

iq_padded = signal.resample_poly(
    stage_1_iq,
    up=1,
    down=DECIMATION_2,
    window=second_fir,
)
iq_padded = np.asarray(iq_padded, dtype=np.complex64)

# 去掉为了滤波边缘而多读取的padding，只保留用户指定的时间段。
trim_start = int(round((requested_start_seconds - padded_start_seconds) * FS_OUT))
requested_output_samples = int(round(requested_duration_seconds * FS_OUT))
trim_stop = min(trim_start + requested_output_samples, len(iq_padded))
iq = iq_padded[trim_start:trim_stop]
if len(iq) < 256:
    raise ValueError("降采样后的数据太短，无法计算稳定的功率谱。")

print("最终 IQ：{} samples @ {:.3f} kS/s".format(len(iq), FS_OUT / 1e3))
print("第二级 FIR taps：", second_num_taps)


# =============================================================================
# FM相位差鉴频：复数 IQ -> FM复合基带 MPX
# =============================================================================

phase_difference = np.angle(iq[1:] * np.conj(iq[:-1]))
instantaneous_frequency_hz = phase_difference * FS_OUT / (2.0 * np.pi)
carrier_offset_hz = float(np.mean(instantaneous_frequency_hz, dtype=np.float64))
mpx_hz = instantaneous_frequency_hz - carrier_offset_hz
mpx_hz = np.asarray(mpx_hz, dtype=np.float32)


# =============================================================================
# 计算鉴频后FM复合基带的功率谱密度
# =============================================================================

mpx_nperseg = min(len(mpx_hz), max(2048, int(round(0.050 * FS_OUT))))
mpx_frequencies, mpx_psd = signal.welch(
    mpx_hz,
    fs=FS_OUT,
    window="hann",
    nperseg=mpx_nperseg,
    noverlap=mpx_nperseg // 2,
    detrend="constant",
    scaling="density",
)
mpx_psd = np.maximum(mpx_psd, np.finfo(np.float64).tiny)

# 这里只报告19 kHz附近的局部峰，不进行“是否为FM”的自动分类。
pilot_mask = (mpx_frequencies >= 18.8e3) & (mpx_frequencies <= 19.2e3)
pilot_noise_mask = (
    ((mpx_frequencies >= 18.0e3) & (mpx_frequencies <= 18.7e3))
    | ((mpx_frequencies >= 19.3e3) & (mpx_frequencies <= 20.0e3))
)
if np.any(pilot_mask) and np.any(pilot_noise_mask):
    pilot_indices = np.flatnonzero(pilot_mask)
    pilot_peak_index = int(pilot_indices[np.argmax(mpx_psd[pilot_mask])])
    pilot_peak_hz = float(mpx_frequencies[pilot_peak_index])
    pilot_noise_median = float(np.median(mpx_psd[pilot_noise_mask]))
    pilot_local_contrast_db = float(
        10.0
        * np.log10(
            float(mpx_psd[pilot_peak_index])
            / max(pilot_noise_median, np.finfo(float).tiny)
        )
    )
else:
    pilot_peak_hz = None
    pilot_local_contrast_db = None


# =============================================================================
# 计算下变频后复数IQ的RF功率谱
# =============================================================================

rf_nperseg = min(len(iq), max(2048, int(round(0.020 * FS_OUT))))
rf_frequencies, rf_psd = signal.welch(
    iq,
    fs=FS_OUT,
    window="hann",
    nperseg=rf_nperseg,
    noverlap=rf_nperseg // 2,
    detrend=False,
    return_onesided=False,
    scaling="density",
)
rf_order = np.argsort(rf_frequencies)
rf_frequencies = rf_frequencies[rf_order]
rf_psd = np.maximum(np.real(rf_psd[rf_order]), np.finfo(float).tiny)


# =============================================================================
# 绘图：RF信道、FM复合基带PSD、指定时间段的鉴频波形
# =============================================================================

cache_root = Path(tempfile.gettempdir()) / "orbdemod-fm-one-file-cache"
(cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
(cache_root / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

figure, axes = plt.subplots(3, 1, figsize=(12, 11))

axes[0].plot(rf_frequencies / 1e3, 10.0 * np.log10(rf_psd), linewidth=0.8)
axes[0].set_xlim(-FS_OUT / 2e3, FS_OUT / 2e3)
axes[0].set_xlabel("Frequency offset from tuned station (kHz)")
axes[0].set_ylabel("PSD (dB/Hz)")
axes[0].set_title(
    "{}: channel IQ centred at {:.6f} MHz".format(
        LABEL,
        RF_FREQUENCY_HZ / 1e6,
    )
)
axes[0].grid(alpha=0.25)

display_limit_hz = min(90e3, FS_OUT / 2.0)
display_mask = mpx_frequencies <= display_limit_hz
axes[1].plot(
    mpx_frequencies[display_mask] / 1e3,
    10.0 * np.log10(mpx_psd[display_mask]),
    linewidth=0.9,
    color="black",
)
axes[1].axvspan(0.1, 15.0, color="tab:blue", alpha=0.10, label="0-15 kHz audio (L+R)")
axes[1].axvspan(18.8, 19.2, color="tab:red", alpha=0.18, label="19 kHz stereo pilot")
axes[1].axvspan(23.0, 53.0, color="tab:orange", alpha=0.10, label="23-53 kHz stereo (L-R)")
axes[1].axvline(38.0, color="tab:orange", linestyle="--", linewidth=1.0, label="38 kHz suppressed centre")
axes[1].axvspan(56.5, 57.5, color="tab:green", alpha=0.18, label="57 kHz RDS")
axes[1].set_xlim(0.0, display_limit_hz / 1e3)
axes[1].set_xlabel("FM composite (MPX) frequency (kHz)")
axes[1].set_ylabel("PSD (dB/Hz)")
axes[1].set_title("FM discriminator output: four broadcast-FM regions")
axes[1].legend(loc="best", fontsize=8, ncol=2)
axes[1].grid(alpha=0.25)

available_waveform_seconds = len(mpx_hz) / FS_OUT
waveform_duration_seconds = min(
    WAVEFORM_DURATION_SECONDS,
    available_waveform_seconds,
)
if WAVEFORM_START_SECONDS is None:
    waveform_start_seconds = 0.5 * (
        available_waveform_seconds - waveform_duration_seconds
    )
else:
    waveform_start_seconds = WAVEFORM_START_SECONDS
    if waveform_start_seconds >= available_waveform_seconds:
        raise ValueError("WAVEFORM_START_SECONDS 必须位于所选分析时间段内部。")
    waveform_duration_seconds = min(
        waveform_duration_seconds,
        available_waveform_seconds - waveform_start_seconds,
    )

waveform_start_sample = int(round(waveform_start_seconds * FS_OUT))
waveform_start_sample = min(waveform_start_sample, len(mpx_hz) - 1)
waveform_sample_count = max(
    1,
    int(round(waveform_duration_seconds * FS_OUT)),
)
waveform_stop_sample = min(
    waveform_start_sample + waveform_sample_count,
    len(mpx_hz),
)
waveform_time_ms = (
    np.arange(waveform_start_sample, waveform_stop_sample, dtype=np.float64)
    / FS_OUT
    * 1e3
)
axes[2].plot(
    waveform_time_ms,
    mpx_hz[waveform_start_sample:waveform_stop_sample] / 1e3,
    linewidth=0.7,
)
axes[2].set_xlabel("Time within selected analysis window (ms)")
axes[2].set_ylabel("Instantaneous frequency deviation (kHz)")
axes[2].set_title(
    "FM composite waveform: {:.1f}--{:.1f} ms".format(
        waveform_start_sample / FS_OUT * 1e3,
        waveform_stop_sample / FS_OUT * 1e3,
    )
)
axes[2].grid(alpha=0.25)

figure.tight_layout()
figure_path = OUTPUT_DIR / "fm_psd.png"
figure.savefig(figure_path, dpi=180)
plt.close(figure)


# =============================================================================
# 保存中间数组和本次运行信息
# =============================================================================

np.savez_compressed(
    OUTPUT_DIR / "fm_arrays.npz",
    iq=iq,
    mpx_hz=mpx_hz,
    mpx_frequencies_hz=mpx_frequencies,
    mpx_psd=mpx_psd,
    sample_rate_hz=np.array(FS_OUT),
)

summary = {
    "input_file": str(INPUT_FILE),
    "output_dir": str(OUTPUT_DIR),
    "label": LABEL,
    "rf_frequency_hz": RF_FREQUENCY_HZ,
    "selection_mode": WINDOW_MODE,
    "start_seconds": requested_start_seconds,
    "duration_seconds": requested_duration_seconds,
    "start_fraction": START_FRACTION if WINDOW_MODE == "fraction" else None,
    "stop_fraction": STOP_FRACTION if WINDOW_MODE == "fraction" else None,
    "input_file_duration_seconds": file_duration_seconds,
    "input_dtype": dtype.str,
    "iq_sample_rate_hz": FS_OUT,
    "waveform_start_seconds": waveform_start_sample / FS_OUT,
    "waveform_duration_seconds": (
        waveform_stop_sample - waveform_start_sample
    ) / FS_OUT,
    "carrier_offset_hz": carrier_offset_hz,
    "pilot_peak_hz": pilot_peak_hz,
    "pilot_local_contrast_db": pilot_local_contrast_db,
}
with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as output_file:
    json.dump(summary, output_file, ensure_ascii=False, indent=2, allow_nan=False)

print("处理完成。")
print("PSD图：", figure_path)
print("数组：", OUTPUT_DIR / "fm_arrays.npz")
print("摘要：", OUTPUT_DIR / "summary.json")
print(json.dumps(summary, ensure_ascii=False, indent=2))
