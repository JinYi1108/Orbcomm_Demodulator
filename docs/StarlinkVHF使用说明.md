# Starlink VHF 解调功能使用说明

`lfdemod starlink-vhf` 用于对一个已知 RF 频率和已知文件时间窗执行：

```text
实数原始电压
  -> LFdemod 通用 DDC 原语组成的 Starlink VHF 信道 DDC
  -> 240 kS/s 复数 IQ
  -> 约 83.333 kS/s（每 chip 两点）
  -> LoRa 前导检测、同步、解调、解码及 CRC 校验
  -> 按 payload 长度选择保守的 Starlink VHF profile
```

它不会搜索整段数据、推断 100 秒周期、计算 TLE 或把协议字段自动映射到某颗具名卫星。

## 安装

LoRa 后端是可选依赖；不使用本功能时不会影响 FM、航空 AM 和 ORBCOMM：

```bash
cd /home/yxp/LFdemod
python -m pip install -e '.[starlink-vhf]'
```

## 典型命令

```bash
lfdemod starlink-vhf \
  -i /home/data/202504152200/20250415-2200-0.dat \
  -f 137.055e6 \
  -s 5.15 \
  -d 1.4 \
  -o /home/yxp/FM/137p05_satellite_candidate/results/example \
  -iq
```

所有完整参数及简写可随时查看：

```bash
lfdemod starlink-vhf --help
```

核心输入为：

- `-i/--input`：实数原始电压文件；默认数据类型是小端 int16。
- `-f/--rf-frequency`：需要移到零频的已知 RF 中心，单位 Hz。
- `-s/--start`：相对单个文件开头的开始秒数。
- `-d/--duration`：搜索窗口长度，单位秒。

DDC 参数提供输入、中间和输出采样率，以及两级滤波器的通带、阻带、纹波、衰减和分块长度。LoRa 参数提供带宽、扩频因子、前导长度、同步字、header 模式、预期及可接受 payload 长度、编码率、CRC、同步容差以及是否强制 LDRO。默认值对应当前已验证的 81-byte Starlink VHF profile。

当前配置对外接受公开观测中出现过的以下 payload 长度：

```text
73, 81, 87, 104, 227 bytes
```

其中只有 81-byte profile 已用本项目两起 21CMA 真实事件验证并进行字段级解释。其他长度可以完成 LoRa PHY 解调、header 与 CRC 检查、二进制及结构化输出，但在取得真实样本前不会套用 81-byte 字段位置或虚构字段含义。默认仍为 81 bytes，可通过 `-eb/--expected-payload-bytes` 和 `-ab/--accepted-payload-bytes` 调整。

## 已验证的默认 PHY 参数

- `SF=8`，`BW=125000/3 Hz`。
- 15 个普通 preamble upchirps。实测 preamble 起点到 PHY data 起点为 19.25 symbols，与 `15 + 2 sync + 2.25 SFD` 一致。
- LoRa conventional sync byte 为 `0x12`；它只是 profile 证据之一，不能单独证明 Starlink 身份。
- explicit PHY header。两个真实包均解析出 payload length 81、CR 4/5、payload CRC enabled。
- 强制 `LDRO=True`。

默认强制 `LDRO=True` 是一项实测 profile 参数，不是按通用 LoRa 符号时长规则自动推断的结果。SF8、BW≈41.667 kHz 时符号时长为 6.144 ms，普通自动规则通常会关闭 LDRO；但两起独立的 21CMA 实测事件只有在启用 LDRO 时才都得到 148 个 PHY data symbols 且 payload CRC 正确。关闭 LDRO 后两者均只得到 113 个 symbols、payload 内容改变且 CRC 失败。该选项仍保留为可配置项，以便将来检验不同 packet profile。

这里的长度需要严格区分：

```text
LoRa explicit PHY header：由接收器解析为元数据，不在输出字节中
81-byte payload：Starlink profile 的协议数据
2-byte payload CRC：跟在 payload 后
83-byte decoded payload-and-CRC：81 + 2
```

因此旧兼容字段 `frame_bytes` 对当前事件是 83 bytes，但不包含 preamble、sync、SFD 或 explicit PHY header。新输出同时明确提供 `header_payload_length`、`coding_rate`、`crc_enabled`、`payload_binary` 和 `decoded_payload_and_crc_binary`。

## 输出

- `summary.json`：配置、包数、PHY header、CRC、CFO、名义样本时间、协议字段、诊断色标和所有输出路径。
- `frames.csv`：每个包一行的结构化摘要，包含 header 长度、CR、CRC状态和长度profile，便于跨文件汇总。
- `packet_NNN_payload.bin`：不含 LoRa CRC 的 payload。
- `packet_NNN_frame.bin`：包含末尾两字节 CRC 的完整解码帧。
- `diagnostic.png`：含 padding 的信道瀑布图、IQ 包络、选定包 symbols 和 dechirp concentration；橙色虚线标出用户请求窗口。
- `channel_iq.c64`：仅在指定 `-iq/--save-iq` 时产生的复数 IQ。

`starlink_profile_match=true` 表示 PHY header、编码率、CRC、已知 payload 长度、同步字，以及该长度下已有的内容特征检查共同吻合。81-byte profile 还要求当前固定字节特征通过；其他已知长度在没有真实帧前只具备 PHY/长度层面的形式支持。该结果支持“Starlink-profile VHF LoRa 帧”的判断，但单个条件或该布尔值本身都不等价于某个 NORAD 目标的身份认定。

时间字段是根据文件起点和名义采样率换算的 `nominal sample time`。与光学观测或轨道预报进行亚秒级对齐时，还需要单独核查文件 UTC 起点、采样钟误差和硬件延迟。
