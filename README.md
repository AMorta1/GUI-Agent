# GUI-Agent
基于多模态⼤模型的桌⾯ GUI 智能体

## 第二周功能

- 使用 mss 进行跨平台内存截图；
- 图像缩放及点/边界框的分辨率转换；
- 使用 EasyOCR 识别中英文文本元素；
- 使用 OpenCV 绘制文本边界框；
- 使用 PyAutoGUI 完成点击、ASCII 输入、滚动和拖拽。

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

以下命令会打开受控的 PyQt5 测试窗口。其中控制与集成验证会真实移动鼠标并输入测试文本，运行期间不要操作鼠标键盘：

```powershell
python scripts/week2_demo.py --ocr-check
python scripts/week2_demo.py --control-check
python scripts/week2_demo.py --integration-check
```

测试结果见 [第二周测试报告](docs/week2-test-report.md)。
