# 在自己的数据上快速开始使用UniTS。

## 使用自己的数据进行分类。

我们以分类任务作为示例。与其他任务的主要区别在于数据格式。您可以按照提供的数据集指南来调整自己的数据。

### 1. 准备数据

我们支持时间序列数据集的常见数据格式。

您可以按照[数据集格式指南](https://www.aeon-toolkit.org/en/latest/examples/datasets/data_loading.html)将您的数据集转换为`.ts`格式的数据集。

数据集应包含`newdata_TRAIN.ts`和`newdata_TEST.ts`文件。

### 2. 定义数据集配置文件

为了支持多个数据集，我们的代码库使用`data_set.yaml`来保存数据集信息。
示例可以在`data_provider`文件夹中找到。

这是一个分类数据集的示例。如果希望使UniTS支持多个数据集，可以在一个配置文件中添加多个数据集配置。
```yaml
task_dataset:
  CLS_ECG5000: # 数据集和任务名称
    task_name: classification # 任务类型
    dataset: ECG5000 # 数据集的名称
    data: UEA # 数据集的数据类型，如果使用'.ts'文件，则使用UEA
    embed: timeF # 使用的嵌入方法
    root_path: ../dataset/UCR/ECG5000 # 数据集的根路径
    seq_len: 140 # 输入序列的长度
    label_len: 0 # 标签序列的长度，用于分类的情况下为0
    pred_len: 0 # 预测序列的长度，用于分类的情况下为0
    enc_in: 1 # 变量数量
    num_class: 5 # 类别数量
    c_out: None # 输出变量数量，用于分类的情况下为0
```
### 3. 调整您的UniTS模型

#### 加载预训练权重（可选）

您可以加载预训练的SSL/有监督的UniTS模型。
运行SSL预训练或有监督训练脚本以获取预训练检查点。
通常，SSL预训练模型具有更好的迁移学习能力。

#### 设置微调脚本

注意：在使用以下脚本之前，请删除标题！

微调/有监督训练

```bash
model_name=UniTS # 模型名称，UniTS
exp_name=UniTS_supervised_x64 # 实验名称
wandb_mode=online # 使用wandb记录训练日志，如果不想使用，请更改为disabled
project_name=supervised_learning # 在wandb中的项目名称

random_port=$((RANDOM % 9000 + 1000))

# 有监督学习
torchrun --nnodes 1 --nproc-per-node=1  --master_port $random_port  run.py \
  --is_training 1 \ # 1表示训练，0表示测试
  --model_id $exp_name \
  --model $model_name \
  --lradj supervised \ # 您可以在utils/tools.py的adjust_learning_rate函数中定义自己的学习率衰减方案
  --prompt_num 10 \ # 提示标记的数量
  --patch_len 16 \ # UniTS中每个标记的路径大小
  --stride 16 \ # 步幅等于路径大小
  --e_layers 3 \
  --d_model 64 \
  --des 'Exp' \
  --learning_rate 1e-4 \ # 根据数据集调整以下超参数。由于时间序列数据的高度多样性，您可能需要为新数据调整超参数。
  --weight_decay 5e-6 \
  --train_epochs 5 \
  --batch_size 32 \ # 真实批量大小= batch_size * acc_it
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $project_name \
  --clip_grad 100 \ # 梯度裁剪以避免NaN
  --pretrained_weight ckpt_path.pth \ # 如果要微调模型，请提供预训练ckpt的路径，否则只需删除它
  --task_data_config_path data_provider/multi_task.yaml # 重要：更改为your_own_data_config.yaml
```
- 提示学习

对于提示学习，仅微调标记，固定模型。
您必须加载预训练模型权重。

```bash
# 提示调优
torchrun --nnodes 1 --master_port $random_port run.py \
  --is_training 1 \
  --model_id $exp_name \
  --model $model_name \
  --lradj prompt_tuning \
  --prompt_num 10 \
  --patch_len 16 \
  --stride 16 \
  --e_layers 3 \
  --d_model $d_model \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 3e-3 \
  --weight_decay 0 \
  --prompt_tune_epoch 2 \ # 提示调优的时期数
  --train_epochs 0 \
  --acc_it 32 \
  --debug $wandb_mode \
  --project_name $ptune_name \
  --clip_grad 100 \
  --pretrained_weight auto \ # 预训练ckpt的路径，您必须为提示学习添加它
  --task_data_config_path  data_provider/multi_task.yaml # 重要：更改为your_own_data_config.yaml
```

###
如果您在使用我们的代码中遇到任何问题，请随时提出问题。

本文档将进行更新。