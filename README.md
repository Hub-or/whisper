#耳语
 
[[博客]](https://openai.com/blog/whisper)
[[论文]](https://arxiv.org/abs/2212.04356)
[[型号卡]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab示例]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)
 
Whisper是一种通用的语音识别模型。它是在一个包含各种音频的大型数据集上训练的，也是一个多任务模型，可以执行多语言语音识别、语音翻译和语言识别。
 
 
##方法
 
![方法](https://raw.githubusercontent.com/openai/whisper/main/approach.png)
 
Transformer序列到序列模型在各种语音处理任务上进行训练，包括多语言语音识别、语音翻译、口语识别和语音活动检测。这些任务被联合表示为解码器要预测的一系列标记，允许单个模型取代传统语音处理管道的许多阶段。多任务训练格式使用一组特殊的标记，作为任务说明符或分类目标。
 
 
##安装程序
 
我们使用Python 3.9.9和[PyTorch](https://pytorch.org/) 1.10.1来训练和测试我们的模型，但代码库预计将与Python 3.8-3.11和最近的PyTorch版本兼容。代码库还依赖于一些Python包，最值得注意的是[OpenAI的tiktoken](https://github.com/openai/tiktoken)，用于实现其快速的标记器。您可以使用以下命令下载并安装（或更新到）最新版本的Whisper：
 
    pip install -U openai-whisper
 
或者，以下命令将从该存储库中提取并安装最新的提交及其Python依赖项：
 
    使用pip安装git+https://github.com/openai/whisper.git
 
要将软件包更新为此存储库的最新版本，请运行：
 
    pip安装 --升级 --no-deps --force-reinstall git+https://github.com/openai/whisper.git
 
它还需要在您的系统上安装命令行工具[`ffmpeg`](https://ffmpeg.org/)，可以从大多数软件包管理器获得：
 
```bash
# 在 Ubuntu 或 Debian 上
使用sudo更新软件库并安装ffmpeg
 
# 在 Arch Linux 上
使用 sudo 命令安装 ffmpeg
 
# 在MacOS上使用Homebrew（https://brew.sh/）
brew安装ffmpeg
 
# 在 Windows 上使用 Chocolatey（https://chocolatey.org/）
choco 安装 ffmpeg
 
# 在 Windows 上使用 Scoop (https://scoop.sh/)
独家安装ffmpeg
```
 
你可能还需要安装[`rust`](http://rust-lang.org)，以防[tiktoken](https://github.com/openai/tiktoken)没有提供适用于你平台的预构建的 wheel。如果在上面的“pip install”命令中看到安装错误，请按照[入门页面](https://www.rust-lang.org/learn/get-started)安装Rust开发环境。此外，你可能需要配置`PATH`环境变量，例如`export PATH="$HOME/.cargo/bin:$PATH"`。如果安装失败，提示“没有名为'setuptools_rust'的模块”，则需要安装'setuptools_rust'，例如通过运行：
 
```bash
pip安装setuptools-rust
```
 
 
## 可用模型和语言
 
有五种型号，四种只有英文版本，提供速度和准确性的权衡。以下是可用模型的名称及其相对于大型模型的大致内存要求和推理速度；实际速度可能因许多因素而异，包括可用硬件。
 
| 尺寸 | 参数 | 纯英文型号 | 多语言型号 | 所需显存 | 相对速度 |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|   tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  基本  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| 小 | 244 M | `small.en` | `small` | ~2 GB | ~6x |
| 中等 |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| 大  |   1550 M   |        N/A         |      `大`       |    ~10 GB     |       1x       |
 
仅使用英语的应用程序的 `.en` 模型往往表现更好，特别是 `tiny.en` 和 `base.en` 模型。我们观察到，对于“small.en”和“medium.en”模型，差异变得不那么明显。
 
Whisper的性能因语言而异。下图显示了“large-v3”和“large-v2”模型在 Common Voice 15 和 Fleurs 数据集上使用 WER（单词错误率）或 CER（字符错误率，以斜体显示）评估的性能细分（按语言）。其他模型和数据集对应的WER/CER指标可以在[论文](https://arxiv.org/abs/2212.04356)的附录D.1、D.2和D.4中找到，以及附录D.3中翻译的BLEU（双语评估研究）分数。
 
按语言分类的WER细分（https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62）
 
 
 
## 命令行使用
 
以下命令将使用“medium”模型对音频文件中的语音进行转录：
 
    耳语音频.flac音频.mp3音频.wav--模型中等
 
默认设置（选择“小”模型）适用于转录英语。要转录包含非英语语音的音频文件，可以使用`--language`选项指定语言：
 
    耳语日语.wav --日语
 
添加“--task translate”会将语音翻译成英语：
 
    耳语日语.wav --语言日语--任务翻译
 
运行以下命令以查看所有可用选项：
 
耳语——救命
 
请参阅[tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py)以获取所有可用语言的列表。
 
 
## Python使用
 
转录也可以在Python中执行：
 
python
导入耳语
 
模型 = whisper.load_model("base")
结果 = 模型.转录("音频.mp3")
打印（结果[“文本”]）
```
 
在内部，`transcribe()`方法读取整个文件，并使用一个30秒的滑动窗口处理音频，对每个窗口执行自回归序列到序列预测。
 
以下是“whisper.detect_language()”和“whisper.decode()”的示例用法，它们提供了对模型的低级访问。
 
python
导入耳语
 
模型 = whisper.load_model("base")
 
#加载音频并填充/修剪它以适应30秒
音频 = whisper.load_audio("音频.mp3")
音频 = 耳语.pad_or_trim(音频)
 
# 制作log-Mel频谱图，并将其移动到与模型相同的设备上
mel = whisper.log_mel_spectrogram(audio).to(model.device)
 
# 检测所讲语言
_, probs = model.detect_language(mel)
print(f"检测到的语言：{max(probs, key=probs.get)}")
 
# 解码音频
选项 = 耳语。装饰选项（）
结果 = whisper.decode(模型, 音高, 选项)
 
# 打印识别的文本
打印（结果文本）
```
 
## 更多示例
 
请使用讨论中的[🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell)类别来分享更多Whisper和第三方扩展的示例用法，例如Web演示、与其他工具的集成、不同平台的端口等。
 
 
##许可证
 
Whisper的代码和模型权重在MIT许可证下发布。有关更多详细信息，请参阅[许可证](https://github.com/openai/whisper/blob/main/LICENSE)。