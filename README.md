# Neural Trace

一个用于观察本地 Qwen3.5-4B 计算过程的极简可视化工作台。

当前版本已完成：

- 默认 3D trace space：32 层节点、DeltaNet 状态路径、Full Attention 连线、残差流和可点击节点
- Qwen3.5 混合架构视图：DeltaNet / 线性注意力、Full Attention、Residual Stream、Logits
- Token 流动、激活场、attention map、输出分布和事件时间轴
- 可播放/暂停/调速/拖动的前向计算模拟器
- `D:\watch\_LLM\_think` 模型目录健康检查
- FastAPI SSE 遥测接口，已准备真实 Transformers forward/backward hooks
- Forward、Backward、Attention、Delta State 四种观测模式
- 可选图片输入：通过 POST SSE 将图像送入 Qwen3.5 Vision Encoder，并显示视觉输入/编码事件

## 启动前端

```powershell
Set-Location 'E:\neural-dialogue-visualizer'
npm install
npm run dev
```

打开 `http://localhost:5173`。

## 启动本地遥测服务

```powershell
Set-Location 'E:\neural-dialogue-visualizer'
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r server\requirements.txt
$env:MODEL_DIR = 'D:\watch\_LLM\_think'
python -m uvicorn server.main:app --reload --port 8000
```

模型文件齐全后，如需启用真实 hook runner：

```powershell
pip install -r server\requirements-model.txt
```

真实运行会尝试懒加载本地 Transformers 模型，并在层级模块上注册 forward hook 和 backward hook。当前真实 hooks 路线使用 SafeTensors checkpoint；GGUF 适合 llama.cpp 推理，但不能直接提供这里所需的 PyTorch autograd 层级事件。4B 模型的 backward 需要较大的显存；如果量化 checkpoint 不支持 autograd，服务会保留 mock 流并返回明确的 runtime error。

文本 trace 使用 GET `/api/stream?prompt=...&mode=...`；带图片时使用 POST `/api/stream`，请求体为 `{"prompt":"...","mode":"forward","image_data":"data:image/png;base64,..."}`。前端的回形针按钮会自动使用 POST 路径，图片限制为 4 MB。

如果模型仍在下载，服务会返回 `mock` 状态，前端继续显示模拟 trace；文件齐全后会显示 `CHECKPOINT READY`。只有 Transformers hook runner 成功加载后才会显示 `MODEL READY`。界面展示的是可观测的激活、状态、注意力和梯度信号，不把隐藏推理文字冒充成“模型思想”。
