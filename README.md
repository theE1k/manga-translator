# Manga Translator

漫画 / 条漫原地翻译：检测文字 → OCR → 本地大模型翻译 → 擦掉原文 → 按原样式重新嵌字。
全部使用开源小模型，在你自己的 Mac 上运行，不依赖付费 API。

| 步骤 | 模型 | 说明 |
|---|---|---|
| 文字检测 | comic-text-detector (ONNX) | 输出文字框 + 像素级文字遮罩；条漫自动分块 |
| OCR | manga-ocr（日文）/ EasyOCR（韩、中、英） | 整个气泡一起识别 |
| 翻译 | Ollama + Qwen2.5-3B（可换 7B、SakuraLLM） | 整页一起翻，保持上下文 |
| 擦字 | 纯色气泡直接填色；画面上的字用 LaMa (ONNX) | 气泡里能做到像素级干净 |
| 嵌字 | Pillow | 保留竖排、匹配原字号/颜色/描边，按气泡形状排版，避头尾 |

## 1. 安装（一次）

```bash
cd ~/work/manga-translator
chmod +x setup.sh run.sh
./setup.sh            # 只装日文；要韩/中/英：./setup.sh ja,ko,zh,en
```
需要约 4GB 磁盘空间。

## 2. 启动

```bash
./run.sh
```
浏览器打开 <http://localhost:8765> 可以拖一张图进去测试效果。
命令行批量：`.venv/bin/python translate_image.py 某章节文件夹/ --src ja --tgt zh`

## 3. 浏览器扩展

**Chrome（电脑上先试）**：`chrome://extensions` → 打开开发者模式 → 加载已解压的扩展 → 选 `extension/` 文件夹。

**Safari（Mac + iPhone）**：
```bash
xcrun safari-web-extension-converter extension \
  --project-location safari --app-name "Manga Translator" \
  --bundle-identifier com.yourname.mangatranslator --swift --no-open
open "safari/Manga Translator/Manga Translator.xcodeproj"
```
Xcode 里在 Signing & Capabilities 选你的 Apple ID（免费账号即可，7 天需要重装一次），iPhone 连上电脑后选中它运行。
然后在 iPhone 上打开 设置 → App → Safari → 扩展 → Manga Translator，打开并允许所有网站。

iPhone 上的扩展设置里，服务器地址填 `./run.sh` 打印出来的 `http://192.168.x.x:8765`（手机和 Mac 要在同一个 Wi-Fi 下）。
如果连不上，可以用 `cloudflared tunnel --url http://localhost:8765` 生成一个 https 地址，再加上 `MT_TOKEN=自己设的密码 ./run.sh`。

**用法**：漫画页右下角有一个「译」按钮。点一下，就会从当前页往后连续翻译；长按可以在原图和译文之间切换。

## 调优

| 环境变量 | 作用 |
|---|---|
| `MT_OLLAMA_MODEL=qwen2.5:7b` | 更好的翻译（约 4.7GB） |
| `MT_FONT=fonts/你的漫画字体.otf` | 换嵌字字体，对「看不出替换」影响最大 |
| `MT_DET_CONF=0.3` | 漏检时调低 |

`tests/` 里是不依赖模型的擦字/嵌字测试：`python tests/make_test_page.py out && python tests/run_mock.py out`。

## 已知限制
- 用 `<canvas>` 画图、或者图片被切成很多小块的阅读器暂不支持（`<img>` 可以）。
- 有些图床会校验 Referer，扩展拿不到原图；这类站点需要单独处理。
- 画面上的大号拟声词（SFX）默认也会被翻译并重新嵌字，效果不如气泡里的对白。
