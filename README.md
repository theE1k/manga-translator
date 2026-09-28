<div align="center">

# 🍓 Manga Translator

**在你正在看的网页上，把韩漫 / 日漫原地翻成中文。擦掉原文，按气泡重新嵌字，看起来就像官方汉化。**

*Read raw manhwa & manga in place: text is detected, erased and re-lettered inside the bubbles, right on the page you are reading.*

![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20·%20Apple%20Silicon-lightgrey)
![Python](https://img.shields.io/badge/python-3.12-blue)
![Chrome](https://img.shields.io/badge/Chrome-extension-orange)

<img src="docs/demo_webtoon.png" width="560" alt="左：韩文原图 · 右：翻译后">

<sub>左：韩文原图 · 右：翻译结果（测试用图为自绘，非真实作品）</sub>

</div>

---

## ✨ 为什么做这个

市面上的漫画翻译器大多面向英语读者、以日漫为主，韩文识别和下拉式条漫都是短板。这个项目专门为**用中文看韩国条漫**打磨，每个关键环节都拿真实章节做过对比实验，选出最好的方案。

## 🚀 亮点

| | 亮点 | 实测数据 |
|---|---|---|
| 🎯 | **翻译准确，懂俚语** | 真实韩漫 50 句对白（脏话、校园黑话、网络用语），准确率 **92%**；同条件下本地 4B 模型只有 42–46% |
| 🔤 | **韩文识别更准** | 整页识别后再按气泡归并。自绘测试页 6 句韩文字母全部识别正确，EasyOCR 只有 2 句完全正确 |
| 🧩 | **跨图气泡不断句** | 条漫被切成很多张图，气泡常常被切在两张之间（一话里约 20 处）。这里会把两张拼起来当成一句翻译、一次嵌字，再切回去 |
| 🧠 | **带着剧情翻译** | 每张图都附上同一话前 20 句对白和后两张图的台词，人名、语气前后一致；记录存盘，重启不丢 |
| ⚡ | **不用等** | 提前翻译往下 10 屏，同时处理 3 张，屏幕上正在看的优先。按正常阅读速度往下看，没有碰到过还没翻完的图 |
| 🎨 | **擦得干净** | 气泡内直接按底色填充；画面上的字用 big-lama 在 **Apple GPU** 上补画（每处 0.08–0.3 秒，CPU 版要 1.36 秒）；纯色、渐变背景直接按周围颜色拟合填充 |
| 💸 | **几乎不花钱** | 翻译走 OpenRouter 上的 DeepSeek V4.1 Flash，一话（约 120 张切图）约 **$0.01** |
| 🔒 | **图片不出本机** | 检测、OCR、擦字、嵌字全在本地；只有识别出的文字会发给翻译 API |

## 🛠 工作原理

```
Chrome 扩展（漫画页里）                         本地服务 (localhost:8765)
────────────────────                         ─────────────────────────────────────
按阅读顺序排队，每次最多 3 张                     ① 文字检测  comic-text-detector (+ 分割遮罩补检)
屏幕上的图优先                                   ② 文字识别  PaddleOCR PP-OCRv5（韩/中/英）· manga-ocr（日）
附带：页面地址、图片位置、后两张图  ──────────▶     ③ 跨图气泡  碰到接缝时拼接相邻两张再识别
                                               ④ 翻译      LLM + 本话前 20 句 + 后两张的台词
                                               ⑤ 擦字      底色填充 / 平面拟合 / big-lama (MPS)
用译图替换原图，长按切换原图  ◀──────────────     ⑥ 嵌字      按气泡形状排版，中文按词断行，数字不拆
```

## 📦 快速开始

> 在 macOS（Apple Silicon，M1 及以上）上开发和测试。模型加 Python 环境大约 **3 GB**，建议把项目放在空间充足的磁盘上。

```bash
git clone https://github.com/theE1k/manga-translator.git
cd manga-translator

# 1) 翻译用的 API 密钥（在 openrouter.ai 创建；只有识别出的文字会发过去）
cp .env.example .env        # 然后填入 MT_API_KEY=sk-or-...

# 2) 安装依赖、下载模型（需要 uv 或 Python 3.10+）
./setup.sh                  # 默认韩文；加日文：./setup.sh ko,ja

# 3) 启动服务（启动时会预热模型，约 6 秒）
./run.sh
```

**安装 Chrome 扩展**：打开 `chrome://extensions`，开启右上角「开发者模式」，点「加载已解压的扩展程序」，选择 `extension/` 目录。

**使用**：打开漫画页面，点右下角的 **「译」**，会从当前位置往下连续翻译；**长按**可以在原图和译文之间切换。

> 没有 API 密钥也能用：`./setup.sh` 会通过 [Ollama](https://ollama.com) 拉取本地模型 `qwen3:4b-instruct`，完全离线，但翻译准确率会明显下降（见下方实测数据）。

## ⚙️ 配置

写在 `.env` 里，或者作为环境变量：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `MT_API_KEY` | — | OpenRouter 密钥；设置了就用云端翻译 |
| `MT_API_MODEL` | `deepseek/deepseek-v4.1-flash` | 换翻译模型（任何 OpenRouter 上的模型 ID） |
| `MT_API_URL` | OpenRouter | 其他兼容 OpenAI 接口的服务 |
| `MT_API_REASONING` | `off` | 支持思考的模型默认关闭思考：快约 3 倍，输出 token 少约 30 倍 |
| `MT_OLLAMA_MODEL` | `qwen3:4b-instruct` | 本地翻译模型（没有 API 密钥时使用） |
| `MT_FONT` | Noto Sans CJK SC Bold | 嵌字字体，换成漫画字体会更像原版 |
| `MT_DET_CONF` | `0.4` | 文字检测阈值，漏检时调低 |
| `MT_WARM_LANGS` | `ko` | 启动时预热哪些语言的 OCR |

## 📊 实测数据

所有数据都来自 `tests/` 里的脚本。评测用的真实漫画对白受版权保护，没有放进仓库。

**翻译准确率**（真实韩漫一话里的 50 句代表性对白，按页面顺序逐张翻译，带前后文；判断标准为意思是否正确）

| 翻译模型 | 准确率 | 每张耗时 | 每话成本 |
|---|---|---|---|
| **deepseek/deepseek-v4.1-flash**（默认） | **92%** | ~2 秒 | ~$0.01 |
| nvidia/nemotron-3-ultra-550b（免费） | 90% | ~4.6 秒 | 免费，有频率限制 |
| qwen/qwen3-235b-a22b-2507 | 76% | ~2 秒 | ~$0.02 |
| 本地 qwen3:4b-instruct | 42–46% | ~2 秒 | 0 |

**韩文 OCR**（自绘测试页，6 句）：PaddleOCR 整页识别，6 句字母全对 · EasyOCR 只有 2 句完全正确

**速度**（M4 / 16 GB，真实条漫，模型已预热）

| 环节 | 中位数 |
|---|---|
| 检测 + OCR | 0.70 秒 |
| 翻译 API | 0.41 秒 |
| 擦字 + 嵌字 | 0.17 秒 |
| 同时处理 3 张时的吞吐 | 约每张 1 秒 |

## ⚠️ 已知限制

- 用 `<canvas>` 画图、或把一页切成很多小方块的阅读器暂时不支持（普通的 `<img>` 没问题）。
- 懒加载的网站只能提前翻译已经加载出来的图片。
- 大号拟声词（SFX）也会被翻译并重新嵌字，效果不如气泡里的对白。
- 翻译服务一次只服务一个人，适合自己用，不适合对外提供服务。
- 手机端还没做（Safari 扩展需要用 Xcode 自己编译）。

## 🗺 路线图

- [ ] 手机端（粘贴章节链接，在网页阅读器里看）
- [ ] 漫画字体和拟声词的专门处理
- [ ] 译名表，统一人名和术语
- [ ] 开机自动启动服务

## 📚 第三方模型与许可

运行时下载，不包含在本仓库里，各自遵循自己的许可：

| 用途 | 模型 | 许可 |
|---|---|---|
| 文字检测 | [comic-text-detector](https://github.com/dmMaze/comic-text-detector)（权重来自 [manga-image-translator](https://github.com/zyddnys/manga-image-translator)） | GPL-3.0 |
| 韩/中/英 OCR | [PaddleOCR PP-OCRv5](https://github.com/PaddlePaddle/PaddleOCR)，通过 [RapidOCR](https://github.com/RapidAI/RapidOCR) 调用 | Apache-2.0 |
| 日文 OCR | [manga-ocr](https://github.com/kha-white/manga-ocr) | Apache-2.0 |
| 擦字 | [LaMa](https://github.com/advimman/lama)：[ONNX 版](https://huggingface.co/Carve/LaMa-ONNX)；TorchScript big-lama 来自 [IOPaint](https://github.com/Sanster/IOPaint) 作者 | Apache-2.0 |
| 字体 | [Noto Sans CJK](https://github.com/notofonts/noto-cjk) | OFL-1.1 |

## ⚖️ 声明

本工具仅供个人阅读使用：只在你自己的浏览器里处理你正在看的页面，不存储、不分发任何漫画内容。请尊重作者和出版方的版权，不要用它制作或传播翻译版本。能买正版的作品，请支持正版。

## License

代码采用 [MIT](LICENSE) 许可。
