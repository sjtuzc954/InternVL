# InternVL SFT

文档整合自官方tutorial：https://internvl.readthedocs.io/en/latest/internvl2.0/finetune.html, 如有细节问题可查看该网页。

## 环境准备

环境准备：

```bash
git clone https://github.com/sjtuzc954/InternVL.git
conda create -n internvl python=3.9 -y
conda activate internvl
pip install -r requirements.txt
pip install flash-attn==2.3.6 --no-build-isolation
```

## 训练数据与模型

以下工作目录均为 `internvl_chat` 。

自定义SFT数据共包含三个文件，分别是元数据 `ui_dataset.json`/`ui_dataset_eval.json`，训练数据 `ui-dataset.zip`，训练脚本 `internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui.sh`.

训练前，将 `ui-dataset.zip` 中的内容解压到 `data/ui-dataset` 目录下

```bash
cd data/
wget https://github.com/sjtuzc954/InternVL/releases/download/v0.2/ui-dataset.zip
unzip ui-dataset.zip
```

元数据和训练脚本已准备好，分别位于 `shell/data` 和 `shell/internvl2.0/2nd_finetune`，可根据需要修改（例如对于其他规模的模型，可仿照1B的脚本修改官方脚本）。

预训练模型下载（以1B为例）：

```bash
mkdir -p pretrained && cd pretrained/
huggingface-cli download --resume-download --local-dir-use-symlinks False OpenGVLab/InternVL2-1B --local-dir InternVL2-1B
```

## 开始训练

可以指定训练脚本的`GPUS`，`PER_DEVICE_BATCH_SIZE`，来适应训练设备的显存大小。例如

```bash
GPUS=1 PER_DEVICE_BATCH_SIZE=4 bash shell/internvl2.0/2nd_finetune/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui.sh
```

训练完成后，可以选择合并LoRA权重，并打包为huggingface格式，之后通过 `AutoModel` 使用模型。

```bash
python -m tools.merge_lora work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui_merge/
cp pretrained/InternVL2-1B/config.json work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui_merge
cp pretrained/InternVL2-1B/*.py work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui_merge
```

## 评估

可以使用 `tensorboard` 工具查看训练过程中的 train/eval loss 曲线。

```bash
tensorboard --logdir ./work_dirs/internvl_chat_v2_0/internvl2_1b_qwen2_0_5b_dynamic_res_2nd_finetune_lora_ui --port 10097 --host 127.0.0.1
```

之后通过浏览器访问 `localhost:10097`。也可以运行 `demo.py` 进行模型推理。