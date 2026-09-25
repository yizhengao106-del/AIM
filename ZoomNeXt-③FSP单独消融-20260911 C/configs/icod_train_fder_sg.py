_base_ = ["icod_train_fder_restored.py"]

model_name = "EffB1_ZoomNeXt_FDER_SG"
model_cfg = dict(use_sg=True)
output_dir = "outputs_fder_sg_20260907"
info = "fder_restored_plus_sg"
