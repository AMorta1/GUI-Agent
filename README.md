# GUI-Agent

## 第四周：受控 GUI Agent 原型 v1.0

Week4 将任务规划、截图/EasyOCR、单步动作 grounding、PyAutoGUI、动作后重新感知和独立反馈校验串联。
真实任务覆盖 **5/5**，完整 E2E 成功 **0/5**；自动测试、API 接通、局部动作或安全探针通过不能替代完整任务验收。
详见 [五类任务测试报告](docs/week4-test-report.md) 和 [证据索引](docs/week4-evidence-index.md)。

### 源码与离线检查

- `scripts/week4_demo.py`：通用 CLI、本机搜索/消息测试服务及无危险文件夹具。
- `scripts/week4_focus_capability.py`：只读 UIA 诊断与保存证据解释；不授权输入，不再加载仓库外 Python 源码。
- `scripts/week4_browser_preflight.py`：纯多行 OCR 标签校验模块，仅使用各真实元素的现有 grounding，不捕获桌面或执行动作。
- `scripts/week4_search_focus_probe.py`：方案 B 的显式单次无 API 点击/写入拦截探针，已完成，不重复运行。
- `scripts/week4_search_e2e.py`：显式隔离 profile + localhost 固定词搜索模式，已保存的单次实验未通过，不重复运行。

以下命令不调用真实 API 或操作桌面：

```powershell
conda activate gui-agent
python scripts/week4_demo.py --help
python scripts/week4_search_e2e.py --help
python -m pytest tests/test_week4_focus_capability.py tests/test_week4_browser_preflight.py tests/test_week4_search_e2e.py tests/test_week4_search_focus_probe.py -q
python -m pytest -q
```

当前最终回归：相关测试 224 passed / 0 failed / 0 skipped；全量 734 passed / 0 failed / 0 skipped。
其中 2 项读取已有真实 UIA JSON 作离线回放；该证据不随 Git 分发，缺失时明确 skipped，不能计为通过。
纯合成与 mock 回归不依赖仓库外 Python 源码；实机诊断仍需 Windows UIA、已有隔离 profile、页面和缓存 OCR。

### 运行与安全边界

通用入口 `week4_demo.py run` 默认 dry-run；它仍可能截图/加载模型，API 路径即使 dry-run 也可能上传裁剪图，
不是“无副作用离线检查”。API 凭证只通过 `GUI_AGENT_API_BASE`、`GUI_AGENT_API_MODEL`、`GUI_AGENT_API_KEY` 等环境变量配置。
真实执行须另行批准任务、窗口、最终检查、截图上传及动作范围；参数说明见 `run --help`，不得凭历史记录复用坐标或 HWND。
默认严格模式阻断文字输入，原有计划/逐动作授权保留；监督式人工 Yes 本身不能可靠证明弹窗关闭后的键盘焦点，不单独据此开放输入。

方案 B 仅显式用于隔离 Edge profile + `http://127.0.0.1:8765/search`，固定词 `W4-E2E-SEARCH-001`。
模型生成计划后一次冻结授权；执行期间机器范围检查、runtime 原校验、UIA 实际焦点关联和最后 write 入口检查继续生效，
不向模型提供 UIA 信息，不使用 DOM/UIA 代替 GUI 操作。`selection_verified=false`，replace、地址栏输入和任意文字仍阻断。
无 API 探针只验证过一次真实 OCR 点击、两道焦点检查及强制写入拦截；尚未完成真实输入/提交闭环。

STOP 文件、Ctrl+C 和 PyAutoGUI FAILSAFE 保留。最后焦点检查与按键发送不是原子操作，STOP 不能撤销或保证中断已开始的输入。
禁止 Agent 外网导航不等于 Edge 进程零外联；当前没有操作系统级零外联保证。公网搜索、真实 IM、个人应用、
macOS/Linux、多显示器及第二分辨率完整闭环均未验证。不要自动重试或提高预算。

维护源码/测试在仓库内；历史一次性运行包装器、截图/OCR、API 响应、日志、profile 和哈希保留于仓库外 `../.test-tmp/`。
实验报告位于 `../Week4实验报告.md`。不要将这些运行产物、密钥或缓存纳入 Git；已有一次性结果目录禁止覆盖或重用。

## 第三周：数据集与规划 Agent

第三周新增统一 JSONL schema，以及 ScreenAgent、WebArena、Multimodal-Mind2Web
三个转换器。原始数据集和预处理结果保存在仓库外；每次转换都会在 `manifest.json`
和 `errors.jsonl` 中记录固定的源 revision、许可、获取日期、数量与错误。

```powershell
python scripts/prepare_datasets.py screenagent --source <split-dir> --split <train|test> --output <output-dir> --revision <commit> --source-acquired-at <date> --limit 20
python scripts/prepare_datasets.py webarena --source <config-dir> --output <output-dir> --revision <commit> --source-acquired-at <date> --limit 20
python scripts/prepare_datasets.py mind2web --cache-dir <dataset-cache> --output <output-dir> --revision <revision> --source-acquired-at <date> --limit 20
```

规划演示只读取已有静态图片并返回通过校验的计划 JSON，不会截取桌面、移动鼠标、
输入键盘或调用第二周控制模块。首次下载模型前需要先配置 Hugging Face 缓存；下面的
`--cache-dir` 指向项目级 `.model-cache` 下的 `huggingface/hub` 目录。

```powershell
python scripts/week3_demo.py --provider local --cache-dir <model-hub-cache> --offline --image <screenshot> --task "Describe the requested GUI task and produce a plan"
```

可通过 `--provider api`、`--base-url`、`--model` 和 `--api-key-env` 选择
OpenAI-compatible 端点。密钥只从指定环境变量读取。

两个模型客户端的构造函数均支持 `max_image_bytes` 和 `max_image_pixels`，默认值
分别由 `gui_agent.models.DEFAULT_MAX_IMAGE_BYTES`（10 MiB）和
`DEFAULT_MAX_IMAGE_PIXELS`（2000 万像素）统一定义。这是本项目的输入安全默认值，
不是模型厂商官方限制；仅接受可解码且扩展名匹配的 PNG/JPEG，等于上限允许通过。
调用方可传入正整数覆盖默认值。Qwen 的 `min_pixels/max_pixels` 是处理器缩放配置，
与原始图片输入安全限制分开。

API 客户端另支持 `max_retries`（默认 `0`，不重试）；例如传入 `2` 时最多请求三次，
只重试连接失败、连接超时和 HTTP 429/502/503/504，等待间隔为 0.5、1 秒。
读取/写入/连接池超时、认证错误、非法输入和非法响应不重试。项目异常保留原继承
关系，并提供 `category`、`retryable`、`status_code` 用于分类；`retryable` 描述错误
是否符合重试策略，不表示仍有剩余次数。`timeout` 默认 60 秒，作用于 HTTPX 各
网络阶段，不是整个调用的总时限。以上参数通过 Python 客户端构造函数配置，
`week3_demo.py` 使用默认值。

已使用智谱 OpenAI-compatible 端点和 `glm-4v-flash` 完成真实多模态 smoke
验证：图片请求、模型响应和 `TaskPlan` 结构化解析链路可用，凭证未写入仓库。
该结果只代表接口链路通过；实际计划仍可能遗漏清空旧文本、最终结果验证等语义步骤，
具体结果与限制见第三周实验报告。
基于多模态⼤模型的桌⾯ GUI 智能体

## 第二周功能

- 使用 mss 进行跨平台内存截图；
- 图像缩放及点/边界框的分辨率转换；
- 使用 EasyOCR 识别中英文文本元素；
- 使用 OpenCV 绘制文本边界框；
- 使用 PyAutoGUI 完成点击、ASCII 输入、滚动和拖拽；
- 通过剪贴板粘贴中文等 Unicode 文本；
- 使用 OpenCV 输出基础矩形 UI 候选框，不进行语义判断或自动点击。

## 开发环境

环境创建、CUDA/PyTorch 验证和基础依赖说明见 [开发环境配置文档](docs/environment-setup.md)。

```powershell
conda activate gui-agent
python scripts/verify_environment.py --require-cuda
```

## 测试

自动单元测试不会操作真实桌面：

```powershell
python -m pytest -q
```

以下命令会打开受控的 PyQt5 测试窗口。其中控制、中文输入与集成验证会真实移动鼠标并输入测试文本，运行期间不要操作鼠标键盘：

```powershell
python scripts/week2_demo.py --ocr-check
python scripts/week2_demo.py --control-check
python scripts/week2_demo.py --chinese-input-check
python scripts/week2_demo.py --candidate-check
python scripts/week2_demo.py --integration-check
```

中文输入验证会先提示并等待确认，继续后将覆盖当前文本剪贴板。候选框验证只截图、输出 bbox 并保存标注图，不会移动鼠标、点击候选区域或判断控件语义。

测试结果见 [第二周测试报告](docs/week2-test-report.md)。
