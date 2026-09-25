_base_ = ['icod_train_fder_restored.py']
model_name = 'EffB1_ZoomNeXt_FSP_Only'
model_cfg = dict(use_fsp=True)
load_from = None
output_dir = 'outputs_fsp_only'
info = 'fsp_only_legacy'
