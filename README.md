# ModelHub

## 项目简介

**ModelHub** 致力于打造一个"大模型聚合接口 all in one"平台，统一整合各大主流模型平台（如 OpenAI、OpenRouter、火山方舟、DMX、GPTsAPI 等），为开发者和企业提供统一、便捷的 API 调用体验。  
本项目以 OpenAI SDK 兼容为主，对于不兼容 OpenAI SDK 的平台，单独实现适配，最大程度降低多平台接入和切换的开发成本。

---

## 主要特性

- **多平台聚合**：支持主流大模型平台的统一接入与调用。
- **OpenAI SDK 兼容**：大部分平台通过 OpenAI SDK 统一调用，极大简化开发流程。
- **灵活扩展**：对不兼容 OpenAI SDK 的平台，单独实现适配，便于后续扩展。
- **上下文与对话管理**：内置对话上下文管理，支持多轮对话、图片消息等。
- **Token 统计**：内置 token 统计功能，方便费用和配额管理。

---

## 目录结构

```
modelhub/
  ├── context.py   # 上下文与消息管理，兼容OpenAI格式
  ├── dialog.py    # 各平台对话聚合与适配
  └── __init__.py
pyproject.toml     # 项目依赖与元数据
README.md          # 项目说明
.gitignore         # Git忽略规则
```

---

## 快速开始

### 1. 安装依赖

建议使用 Python 3.12 及以上版本。

```bash
pip install -r requirements.txt
# 或者使用 pyproject.toml 进行依赖管理
```

### 2. 配置环境变量

在项目根目录下创建 `.env` 文件，配置各平台的 API Key，例如：

```
OPENAI_API_KEY=你的OpenAI密钥
OPENROUTER_API_KEY=你的OpenRouter密钥
VOLC_API_KEY=你的火山方舟密钥
DMX_API_KEY=你的DMX密钥
GPTS_API_KEY=你的GPTsAPI密钥
```

### 3. 示例代码

```python
from modelhub.dialog import OpenAIDialog, OpenRouterDialog, VolcDialog, DMXDialog, GPTsAPIDialog

# 以OpenAI为例
dialog = OpenAIDialog(model_name="gpt-3.5-turbo", system_prompt="你是一个智能助手。")
response = dialog.send("你好，ModelHub！")
print(response)
```

---

## 支持的平台

- [x] OpenAI
- [x] OpenRouter
- [x] 火山方舟（Volcengine）
- [x] DMX
- [x] GPTsAPI
- [ ] 其他平台（欢迎贡献）

---

如需更多帮助或有合作意向，欢迎联系作者或提交 issue！
