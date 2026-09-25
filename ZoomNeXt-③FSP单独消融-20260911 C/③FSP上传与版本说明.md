# 原版ZoomNeXt＋③FSP单独消融

本次根据用户要求参考旧FSP实现，而非实施之前提出的跨层差异引导新设计。
FeatureShrinkageDecoderBlock类逐字提取自本地ZoomNeXt-main/ZoomNeXt-main/methods/zoomnext/layers.py；来源校验记录见FSP历史来源.json。
保留旧版五层FSP位置：MHSIU → FSP → RGPU。第5层无上一层输入，第4至1层使用相邻层门控聚合；FSP包含膨胀率1、2、3的卷积分支。第5层保留历史模块中未用到的prev_proj和gate等参数以贴近旧结构。
本版不是完整FSPNet论文复现，不保证与旧服务器中产生提升的代码相同，更不保证复现旧数值。论文记录应称FSP启发的解码融合，而非直接称完整FSPNet。

模型EffB1_ZoomNeXt_FSP_Only继承原版EffB1_ZoomNeXt，只改解码融合。①关闭、②关闭、MSCA关闭，不创建相关分支。
使用原版forward与BCE＋UAL损失。继承相同训练配置，包括骨干预训练、随机种子、学习率和轮数，不加载①或②的最终权重。继承配置中的edge_loss_weight字段在本模型中不产生边缘损失。
原版入口zoomnext.py保持原样。训练器继续使用上次的main_for_image.py，必须支持传递cfg.model_cfg。

## 上传到已有服务器项目（六个文件）

| 本目录文件 | 服务器位置 |
| --- | --- |
| methods/zoomnext/fsp_only.py | /home/gyz/ZoomNeXt/methods/zoomnext/fsp_only.py |
| methods/zoomnext/fsp_legacy_block.py | /home/gyz/ZoomNeXt/methods/zoomnext/fsp_legacy_block.py |
| methods/__init__.py | /home/gyz/ZoomNeXt/methods/__init__.py |
| configs/icod_train_fsp_only.py | /home/gyz/ZoomNeXt/configs/icod_train_fsp_only.py |
| check_fsp_only.py | /home/gyz/ZoomNeXt/check_fsp_only.py |
| run_fsp_only.py | /home/gyz/ZoomNeXt/run_fsp_only.py |

此映射假设服务器仍有①②实验使用的完整工程。也可把本目录完整上传为独立项目，提供具有正确绝对数据路径的dataset.yaml。
注册文件保留①②版本注册。若服务器此文件有额外自行修改，请保留并增加from .zoomnext.fsp_only import EffB1_ZoomNeXt_FSP_Only。

## 在GPU1上启动（先检查可用性，只执行一次）

```bash
cd /home/gyz/ZoomNeXt && {
    LOG="fsp_only_$(date +%Y%m%d_%H%M%S).log"
    nohup /home/gyz/anaconda3/envs/zoomnext/bin/python -u run_fsp_only.py --gpu 1 --data-cfg dataset.yaml > "$LOG" 2>&1 < /dev/null &
    echo "PID=$! 日志=$LOG"
    echo "$LOG" > last_fsp_only_log.txt
}
```

应看到MODEL: EffB1_ZoomNeXt_FSP_Only和FSP=True; guidance=False; frequency=False; edge=False; MSCA=False。
启动器先检查数据与模型，然后将源码、配置、数据路径、环境和权重结果独立保存到fsp_only_runs；从日志SNAPSHOT/RESULTS定位本次快照。正式服务器训练尚未执行，暂无性能结果。
