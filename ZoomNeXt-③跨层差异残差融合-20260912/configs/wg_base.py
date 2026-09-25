_base_ = ['icod_train_fder_restored.py']
model_name = 'EffB1_ZoomNeXt_WeightGuided'
model_cfg = dict(use_weight_guidance=False, use_frequency=False, use_edge=False,
                 guided_stages=(3, 2))
load_from = None
output_dir = 'outputs_weight_guided'
info = 'wg_base'
