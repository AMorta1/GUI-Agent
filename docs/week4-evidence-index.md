# Week4 证据索引

此索引仅引用已存在的外部原始记录，不复制/修订原始截图、OCR、响应、日志、profile 或历史哈希。
根目录为 `E:\Projects\Microsoft\.test-tmp`；以下相对链接以当前仓库位置为基准，证据不随 Git 分发。
逐次结果与判定口径见 [测试报告](week4-test-report.md)。

## 五类真实任务

| 实验目录 | 主要证据/说明 |
|---|---|
| [week4-stage5-search-first-20261008](../../.test-tmp/week4-stage5-search-first-20261008/) | 首次搜索模型响应、截图/OCR、步骤反馈与 inconclusive |
| [week4-stage5-message-first-20261008](../../.test-tmp/week4-stage5-message-first-20261008/) | 无焦点输入提议，blocked，动作 0 |
| [week4-stage5-file-first-20261008](../../.test-tmp/week4-stage5-file-first-20261008/) | 3 次 File 点击、预算阻断；保留原菜单干扰记录 |
| [week4-stage5-file-retest-20261008](../../.test-tmp/week4-stage5-file-retest-20261008/) | cancelled、1 次点击、菜单仍可见、重复动作预设拒绝、前后源哈希 |
| [week4-stage5-close-first-20261008](../../.test-tmp/week4-stage5-close-first-20261008/) | 步骤证据失败，动作 0 |
| [week4-stage5-close-retest-20261008](../../.test-tmp/week4-stage5-close-retest-20261008/) | 原 window_closed 定义、独立窗口仍存在、源哈希不变；最终检查未执行 |
| [week4-stage5-browser-first-20261008-v2](../../.test-tmp/week4-stage5-browser-first-20261008-v2/) | 现场初始校验通过后焦点丢失，API 0、动作 0 |
| [week4-stage5-browser-retry-20261008-v2](../../.test-tmp/week4-stage5-browser-retry-20261008-v2/) | 多行 OCR 正确，text_present 步骤证据语义不符；API 2、动作 0 |
| [week4-search-e2e-once-20261008](../../.test-tmp/week4-search-e2e-once-20261008/) | 方案 B GLM 新实验：model-01/02、run-summary、safety.jsonl、service-before/after、source-hashes、observations |

[方案 B 搜索原始响应](../../.test-tmp/week4-search-e2e-once-20261008/model-02.json)、
[末帧截图](../../.test-tmp/week4-search-e2e-once-20261008/observations/obs-0005.png)、
[只读复核/原始文件哈希](../../.test-tmp/week4-search-e2e-review-20261008/review.json)。
截图用于人工检查原始状态，不提供给 Agent 作为任务完成 oracle。

## 独立安全证据

| 目录 | 结果与边界 |
|---|---|
| [week4-focus-deferred-desktop-menu-20261008-v2](../../.test-tmp/week4-focus-deferred-desktop-menu-20261008-v2/) | 无 API 菜单探针通过，非 E2E |
| [week4-focus-order-desktop-input_denial-20261008](../../.test-tmp/week4-focus-order-desktop-input_denial-20261008/) | 人工 No 立即停止；ASCII INPUT 未变由用户确认 |
| [浏览器完整只读 PASS](../../.test-tmp/week4-stage5-browser-prep-20261008-v2/desktop-preflight-multiline-once-20261008/) | preflight-summary、ocr-input.png、ocr-raw/normalized、observations；唯一三行标签、不授权点击 |
| [UIA 保存的能力诊断](../../.test-tmp/week4-focus-capability-once-20261008-v2/uia-capability.json) | 输入框/地址栏/按钮 FocusedElement；供 2 项可选离线回放 |
| [week4-search-focus-probe-once-20261008](../../.test-tmp/week4-search-focus-probe-once-20261008/) | probe-summary、两次焦点校验、强制 write veto、independent-field-after、服务记录；1 点击、0 写入 |
| [探针复核/原始文件哈希](../../.test-tmp/week4-search-focus-probe-review-20261008/review.json) | 无 API 探针原始 30 文件的只读复核 |
| [搜索旧固定尺寸失败](../../.test-tmp/week4-stage5-search-safety-once-20261008/) | preparation_blocked、API/动作均 0，保留原 1064x1020 限制失败，不计新真实任务 |
| [阶段 3 完整短闭环复验](../../.test-tmp/week4-stage3-desktop-completion-retest-20261007/) | 真实 CLICKED 反馈通过，但后续 BUTTON OK OCR 误识别，完整结果未通过 |

其他准备失败、版本备份、焦点抖动失败和一次性离线包装器继续保留在外部 `.test-tmp/` 的原有 Week4 目录，不覆盖或迁移。
历史浏览器 `run_once.py`/`retry_once.py` 不作为新的维护源码；可复用多行函数已提取到仓库，旧入口不改引用。

## 阶段 6 新证据

[week4-stage6-delivery-20261009](../../.test-tmp/week4-stage6-delivery-20261009/)：

- `related-tests.xml`：首次 223 passed / 1 failed / 0 skipped，新测试夹具问题。
- `related-tests-final.xml`：修正夹具后 224 passed / 0 failed / 0 skipped。
- `repository-tests.xml`：最终全量 734 passed / 0 failed / 0 skipped。
- `missing-evidence-tests.xml`：单独模拟缺失外部 JSON，0 passed / 0 failed / 2 skipped，不计通过。
- `preservation-before.json`：整理前 11,083 个受保护文件逐项 SHA-256 清单。
- `delivery-checks.json`：最终清单、回归计数、源码提取与保存证据检查、前后保护清单一致性、Git 只读检查结果。

保护范围包括 `gui_agent/*.py`、3 个不应修改的 Week4 入口和历史 Week4 文件。
排除本阶段新目录、`__pycache__` 与浏览器 profile 子目录；profile 不启动、不编辑，但不将可能由仍运行的浏览器改变的缓存冒充不可变证据。
Git 不接收上述运行产物；新交付报告为 `../Week4实验报告.md`，仓库内报告和此索引用于源码交付。
