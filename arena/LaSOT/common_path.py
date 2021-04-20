import os
######################################################################
dataset_name_ = 'UAV123'
video_name_ = ''  # 'airplane-9'

######################################################################
dataset_root_ = './UAV123/'
save_dir = './analysis'

######################### Refine Module ################################
model_dir = './'
model_code = 'a'
#refine_path = os.path.join(model_dir, "SEx_beta/SEcm_r34/SEcmnet_ep0040-{}.pth.tar".format(model_code))
# refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet-alex_ep0040-a.pth.tar'
# refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet_ep0040-c.pth.tar'
refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet-efb0_ep0040-a.pth.tar'
# refine_path = '/home/ymz2/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet-mbv2_ep0040-c.pth.tar'

#print(refine_path)
RF_type = 'AR_CrsM_R18SR20_pixCorr_woPr_woNL_corner_{}'.format(model_code)
selector_path = 0
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
