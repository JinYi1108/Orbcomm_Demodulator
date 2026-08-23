# FM 功能使用说明

这份文档集中记录广播 FM 后处理功能的入口、参数、自动命名规则和输出文件。项目展示名写作 `LFdemod`；终端命令和公开 Python 包统一使用小写 `lfdemod`。以后忘记参数时，优先查看这里；命令行也可以运行：

```bash
lfdemod fm --help
```

## 当前功能边界

当前处理链为：

```text
480 MS/s 原始实电压
→ 目标频率数字下变频（DDC）
→ 滤波并降采样到 240 kS/s 复数 IQ
→ 相邻 IQ 相位差鉴频
→ FM 复合基带（MPX）
→ MPX 功率谱和诊断图
```

当前功能用于检查候选信号中是否出现广播 FM 的 MPX 结构。它还没有完成左右声道恢复、音频文件输出或自动 FM 分类。

## 推荐入口

完成可编辑安装后：

```bash
python -m pip install -e .
```

查看顶层命令和 FM 子命令：

```bash
lfdemod --help
lfdemod fm --help
```

运行一个频率和时间窗口：

```bash
lfdemod fm \
  --input /data/20250415-1940-0.dat \
  --rf-frequency 98.33e6 \
  --start 30 \
  --duration 3
```

没有指定输出目录时，结果自动保存到：

```text
results/20250415-1940-0/98.33MHz/30_3/
```

## 文件窗口参数

两种模式只能选择一种。

### 按秒选择

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `--start` | 相对于原始文件开头的起始秒数 | `30` |
| `--duration` | 处理时长，单位秒 | `3` |

```bash
--start 30 --duration 3
```

表示处理原始文件的 30–33 秒。

### 按文件比例选择

| 参数 | 含义 | 示例 |
| --- | --- | --- |
| `--start-fraction` | 文件起始比例，范围0–1 | `0.25` |
| `--stop-fraction` | 文件结束比例，范围0–1 | `0.30` |

```bash
--start-fraction 0.25 --stop-fraction 0.30
```

表示处理文件总时长的25%–30%。自动目录仍使用换算后的实际起点秒数和时长；原始比例记录在 `run_config.json` 和 `summary.json` 中。

## 常用信号和数据参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--rf-frequency` | 必填 | 目标绝对频率，单位Hz，例如 `98.3e6` |
| `--dtype` | `<i2` | 小端有符号16位原始样本 |
| `--sample-rate` | `480e6` | 原始实电压采样率 |
| `--intermediate-rate` | `2.4e6` | 第一级降采样后的采样率 |
| `--channel-rate` | `240e3` | 最终复数IQ采样率 |
| `--passband` | `90e3` | 目标信道单边有效通带 |
| `--stopband` | `120e3` | 最终滤波器单边阻带起点 |
| `--stopband-attenuation` | `60` | 阻带衰减要求，单位dB |
| `--padding` | `0.020` | 分析窗口两侧额外读取的滤波留白，单位秒 |
| `--chunk-samples` | `10000000` | 高采样率数据每个处理块的样点数 |
| `--label` | `direct_fm_test` | 图标题和摘要中的人为标签 |

`chunk-samples` 只影响内存占用和运行效率，不改变用户选择的分析时长。

## 第三张图参数

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--waveform-start` | 不指定 | 第三张图在所选分析窗口内的起点；不指定表示居中 |
| `--waveform-duration` | `0.050` | 第三张图显示时长，单位秒 |

例如完整分析原始文件的 30–33 秒，但第三张图只显示这段数据内部第1.2秒开始的20 ms：

```bash
lfdemod fm \
  --input /data/20250415-1940-0.dat \
  --rf-frequency 98.3e6 \
  --start 30 \
  --duration 3 \
  --waveform-start 1.2 \
  --waveform-duration 0.02
```

第三张图对应原始文件的 31.2–31.22 秒。第一张和第二张功率谱仍使用完整的30–33秒数据。

不要把以下三个时间混淆：

- `duration`：从原始文件中分析多长时间；
- `compute_mpx_psd(segment_seconds=0.050)`：Welch PSD每个分段的长度；
- `waveform-duration`：第三张时域图显示多长时间。

## 自动输出目录和防覆盖

未填写 `--output-dir` 时采用：

```text
输出根目录/原始文件名/频率MHz/开始秒数_处理秒数/
```

例如：

```text
results/20250415-1940-0/98.3MHz/30_3/
results/20250415-1940-0/98.33MHz/30_3/
results/20250415-1940-0/98.33MHz/30.5_0.05/
```

数字最多保留六位小数，并去掉没有意义的末尾零。因此 `98.30` 写成 `98.3MHz`，但 `98.33` 不会变成 `98.3MHz`。

如果基础目录已经存在，默认不覆盖，而是依次生成：

```text
30_3/
30_3_run02/
30_3_run03/
```

### 改变自动根目录

```bash
--output-root /home/yangyanbin/FM_results
```

这只改变最外层根目录，内部仍自动按文件名、频率和时间命名。

### 主动指定输出目录

```bash
--output-dir /home/yangyanbin/my_test
```

此时不再生成文件名、频率和时间层级，以用户给出的路径作为基础目录。如果它已存在且没有允许覆盖，则创建 `my_test_run02`。

### 明确允许覆盖

```bash
--output-dir /home/yangyanbin/my_test --overwrite
```

只有给出 `--overwrite` 时，程序才会替换基础目录里的同名结果文件。

## 输出文件

每个结果目录包含：

| 文件 | 内容 |
| --- | --- |
| `fm_psd.png` | IQ频谱、MPX频谱、局部瞬时频偏三联图 |
| `fm_arrays.npz` | 完整分析窗口的IQ、MPX和MPX PSD数组 |
| `summary.json` | 实际输出目录、实际窗口和简单测量结果 |
| `run_config.json` | 用户请求的完整FM和DDC配置 |

读取保存数组：

```python
import numpy as np

data = np.load("results/.../fm_arrays.npz")
iq = data["iq"]
mpx_hz = data["mpx_hz"]
frequencies = data["mpx_frequencies_hz"]
psd = data["mpx_psd"]
fs = float(data["sample_rate_hz"])
```

## 四个FM模块分别负责什么

| 文件 | 主要职责 | 通常是否由用户直接调用 |
| --- | --- | --- |
| `fm/fm_ddc.py` | 原始实电压到目标频道复数IQ | 做单独DDC实验时调用 |
| `fm/demod.py` | IQ鉴频、MPX PSD、19 kHz候选测量 | 做中间结果研究时调用 |
| `fm/pipeline.py` | 文件读取、时间选择、串联处理和保存 | 推荐的完整程序入口 |
| `fm/plotting.py` | 生成三联诊断图 | 通常由pipeline调用 |

程序化调用完整 pipeline：

```python
from lfdemod.fm import FMDDCConfig, FMPSDConfig, analyze_fm_psd_file

ddc = FMDDCConfig()
config = FMPSDConfig(
    rf_frequency_hz=98.3e6,
    start_seconds=30.0,
    duration_seconds=3.0,
    waveform_start_seconds=None,
    waveform_duration_seconds=0.050,
    ddc=ddc,
)

summary = analyze_fm_psd_file(
    "/data/20250415-1940-0.dat",
    None,
    config,
    output_root="results",
    overwrite=False,
)

print(summary["output_dir"])
```

查看代码内置说明：

```python
from lfdemod.fm import analyze_fm_psd_file, downconvert_fm_voltage

help(analyze_fm_psd_file)
help(downconvert_fm_voltage)
```

## 单文件学习版本

`examples/fm_psd_one_file.py` 把全部步骤按执行顺序写在一个文件中，适合阅读和临时调试。正常真实数据测试优先使用 `lfdemod fm`；`examples/analyze_fm_psd.py` 只是调用同一个 CLI 的兼容包装，不再复制参数解析和 pipeline 逻辑。
