_base_ = ['icod_train_fder_restored.py']
model_name = 'EffB1_ZoomNeXt_FSPResidual'
model_cfg = dict(use_fsp_residual=True, fusion_stages=(3, 2), use_difference=True)
load_from = None
output_dir = 'outputs_fsp_residual'
info = 'fsp_residual_difference'
