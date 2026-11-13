# Code-Simplification

如果把 Java 方法里的修饰符、括号和分号优先删掉，模型还能描述这段代码，或根据描述找回它吗？这个项目用同一套词法简化规则改变模型输入，分别在 **代码摘要生成** 和 **自然语言检索代码** 两个任务上比较不同的删除比例。

## 先看一个例子

在仓库根目录运行以下命令。示例只依赖 `javalang`，可以在 CPU 上运行，无需下载模型或数据集。

```bash
python -m pip install javalang==0.13.0
python ast_simplifier.py
```

输入的 Java 方法：

```java
public static int square(int value) { return value * value; }
```

删除比例设为 `0.5` 后，模型收到的文本：

```text
int square int value return value * value
```

原方法有 15 个 Java 词法 token。简化器按重要性从低到高删除 `floor(15 × 0.5) = 7` 个，同分时从左到右删除，再按原顺序拼接剩余 token。输出供模型编码，可能无法编译。比例设为 `0` 可以得到保留全部词法 token 的对照输入，空白和注释仍由分词步骤处理。

换成自己的 Java 文件，或运行示例的回归测试：

```bash
python ast_simplifier.py Example.java --ratio 0.3
python -m unittest discover -s tests
```

## 准备实验环境

项目的实验环境使用 Python 3.10。以下命令从仓库根目录执行，建议在虚拟环境中安装。GPU 用户可先安装与驱动匹配的 PyTorch 2.9.1，再安装其余依赖。[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v291)

```bash
python -m pip install -r requirements.txt
```

按要运行的任务准备对应文件：

| 任务 | 默认路径 | 输入内容 |
| --- | --- | --- |
| 代码摘要 | `java/train.jsonl`、`java/valid.jsonl`、`java/test.jsonl` | 每行一个 JSON 对象，`code` 是 Java 方法，`docstring` 是参考摘要，均为字符串 |
| 代码检索训练 | `java_research/train.txt` | 每行五个字段，以 `<CODESPLIT>` 分隔，依次为标签、URL、函数名、描述、代码。读取标签为 `1` 的配对 |
| 代码检索测试 | `java_research/java_test_0.jsonl` | 每条记录提供配对的代码与描述。检索候选集来自本次抽取的测试记录 |

摘要数据可按 CodeXGLUE 的说明从 CodeSearchNet 准备。原项目也提供了数据分享链接，提取码为 `wdg9`。检索实验接受上表两种输入格式。请记录实际使用的数据来源和划分，以便比较不同运行。[摘要数据准备](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text#download-data-and-preprocess) · [检索数据参考](https://github.com/microsoft/CodeBERT/tree/master/CodeBERT/codesearch) · [原项目数据分享](https://pan.baidu.com/s/1mChuuAIMUY9eBtS-de7ttQ?pwd=wdg9)

实验脚本优先查找根目录下的 `codet5-base-local/` 或 `codebert-base-local/`，也可以用 `--model-path` 指定本地模型目录或 Hugging Face 模型 ID。首次在线加载需要下载权重。数据集和模型按各自来源的条款使用。[CodeT5](https://huggingface.co/Salesforce/codet5-base) · [原项目 CodeT5 镜像](https://hf-mirror.com/Salesforce/codet5-base/tree/main) · [CodeBERT](https://huggingface.co/microsoft/codebert-base) · [CodeSearchNet 语料许可](https://github.com/github/CodeSearchNet#licenses)

## 实验一 · 生成 Java 方法摘要

```bash
python train_summarization.py --ratio 0.3 --train-fraction 0.1 --seed 42
python inference.py --ratio 0.3
```

第一条命令在 10% 的训练集上微调 CodeT5，按验证集的语料级 SacreBLEU 选择检查点，最后在测试集上评估一次。模型和分词器写入 `outputs/summarization/ratio-0.3/model/`，测试指标写入同一运行目录下的 `checkpoints/test_results.json`。第二条命令用该模型为内置 Java 方法生成摘要，也可传入文件，例如 `python inference.py Example.java --ratio 0.3`。

## 实验二 · 用自然语言检索代码

```bash
python run_search_experiment.py --model codet5 --ratios 0 0.3 0.5 --sample-fraction 0.1 --epochs 3 --seed 42
```

每个删除比例都使用相同的训练样本、测试候选集和随机种子。模型训练三轮后，用测试描述的首行检索候选集中的全部代码。MRR 是各查询对应代码排名倒数的平均值，数值越高，配对代码平均越靠前。评估只将同一行的代码计为相关项，其他实现即使符合相同描述也不计分。改变候选集大小或数据来源会改变这个指标，比较时应固定这些条件。

每完成一个比例，脚本都会把测试 MRR、样本数和运行参数写入 `outputs/search/results.csv`。全部比例完成后生成 `search_benchmark_result.png`。重复实验可用 `--output-dir` 指向新的目录。切换骨干模型可用 `--model codebert`。

## 代码入口

| 文件 | 用途 |
| --- | --- |
| [ast_simplifier.py](ast_simplifier.py) | 用 `javalang` 分词，按优先级裁剪 token；也可独立运行示例 |
| [prepare_data.py](prepare_data.py) | 预览摘要数据的处理效果，例如 `python prepare_data.py java/train.jsonl --limit 5` |
| [train_summarization.py](train_summarization.py) / [inference.py](inference.py) | 训练摘要模型，加载训练产物生成摘要 |
| [model_search.py](model_search.py) / [run_search_experiment.py](run_search_experiment.py) | 编码描述和代码，训练检索模型并报告 MRR |
