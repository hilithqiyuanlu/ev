# EV Phase 1b：macOS GUI 客户端

## 当前状态

仓库开发版已经实现并通过自动测试和首轮真实麦克风验收：

- `ev engine serve` JSONL v1 协议和进程状态机；
- 菜单栏，以及实时、历史、Query 队列、声纹、模型和设置界面；
- 固定 Release 模型下载、SHA256 校验、安全解压和原子安装；
- SQLite v2、voice/manual query、WAV 回放与一致性删除；
- FunASR 四模型本地加载、macOS Debug 构建和真实 partial/final；
- 停止时当前段 flush、后台终稿/声纹/WAV/SQLite 处理；
- 8 段用户 profile 重启持久化。

尚未完成用户/非用户分别说 `EV` 的联合门控验收、声纹阈值标定、延迟/内存测量，
以及脱离仓库 `.venv` 的独立 arm64 `.app` 打包。签名和公证继续后置。

## 架构

EV macOS 客户端使用 SwiftUI。Python engine 继续拥有麦克风、FunASR、模型、WAV
和 SQLite，客户端不直接读取或修改数据库。

```text
SwiftUI App <- stdin/stdout JSONL v1 -> ev engine serve
                                           |
                      VAD / ASR / 声纹 / 模型下载 / SQLite
```

开发版从仓库 `.venv/bin/python` 启动 engine，数据默认保存在
`~/Library/Application Support/EV/`。独立 arm64 sidecar、签名和公证后置。

## Engine 协议

请求：

```json
{"version":1,"request_id":"uuid","command":"get_status","payload":{}}
```

事件：

```json
{"version":1,"request_id":"uuid","type":"engine_state","timestamp":"...","payload":{}}
```

stdout 只允许 JSONL 协议，日志写 stderr，并由开发客户端保存到应用日志目录。支持以下命令：

```text
get_status, list_devices
verify_models, download_models, cancel_download
start_listening, stop_listening
begin_enrollment, capture_enrollment_sample, cancel_enrollment
list_segments, delete_segment, delete_all_segments
submit_manual_query, delete_query, delete_all_queries
shutdown
```

实时事件包括 `capture_started`、音量、VAD 起止、partial、`segment_processing`、
`segment_failed`、终稿落库、声纹结果、模型下载进度和 query candidate。段级事件都带
`segment_id`；所有事件都带 UTC `timestamp`。

## 开发运行

```bash
uv sync
uv pip install funasr torch torchaudio
xcodebuild -project apps/macos/EV.xcodeproj -scheme EV \
  -configuration Debug -derivedDataPath /tmp/ev-derived CODE_SIGNING_ALLOWED=NO build
open /tmp/ev-derived/Build/Products/Debug/EV.app
```

也可以直接在 Xcode 中打开 `apps/macos/EV.xcodeproj`。开发版通过源文件路径定位仓库；
若移动构建产物，可设置 `EV_REPO_ROOT=/absolute/path/to/EV`。

## 当前边界

- App 包含菜单栏和实时、历史、声纹、模型、设置五个视图。
- 模型固定从 `models-v0.1.0` Release 下载，逐包校验 SHA256、验证目录结构后原子安装；
  下载失败或取消不会覆盖已有可用模型。
- 所有语音段都进入历史；只有 `EV + user` 进入 voice query。
- 手动 query 与 voice query 统一进入 SQLite `queries` 待处理队列。
- 删除语音段会同时删除 WAV 和关联 voice query；删除操作由 Python engine 统一执行。
- 本阶段不调用 LLM，不生成回复，不实现 TTS、iPhone 或正式签名分发。

## 数据目录

开发版默认使用 `~/Library/Application Support/EV/`：

```text
models/   固定 Release 模型
archive/  VAD 人声 WAV
ev.db     SQLite 元数据和 query 队列
logs/     engine stderr 日志
```

客户端通过 `EV_DATA_DIR` 和 `EV_MODEL_ROOT` 将目录传给 Python engine。模型下载使用
仓库内的 `src/ev/resources/models-v0.1.0.json` manifest，不静默升级版本。

## 验证顺序

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -p no:cacheprovider -q
CLANG_MODULE_CACHE_PATH=/tmp/ev-clang-module-cache \
  SWIFT_MODULE_CACHE_PATH=/tmp/ev-swift-module-cache \
  swift test --disable-sandbox --package-path apps/macos
xcodebuild -project apps/macos/EV.xcodeproj -scheme EV \
  -configuration Debug -derivedDataPath /tmp/ev-derived CODE_SIGNING_ALLOWED=NO build
```

自动测试覆盖流式 VAD/ASR 参数与事件顺序、后台 worker、协议、SQLite migration、query
写入与删除、模型 SHA256/安全解压和原子安装。真实麦克风已确认 partial/final、用户与
非用户归档、声纹分数、profile 持久化和停止 flush；`EV + user` 与非用户对照仍需现场完成。
