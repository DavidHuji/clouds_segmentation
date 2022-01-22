IMG_MAX_VAL = 255
MSK_MAX_VAL = 113
DOWN_SCALE_SIZE = 128

epochs =0
augmentations = False
train_all = True

cross_entropy_loss = True
focal_loss = False
weighted_loss = True

using_unet = True
using_michals_unet = False

overfit_data = False
one_ch_in = True
norm_with_average_sub = True

unify_classes_first_and_third = True
use_only_single_class = False

use_gradient_accumulation = 1  # 1 or -1 for False
btch_size = 8

# use initialisation weights
use_initialisation_weights = True
initialisation_weights = '/home/gamir/DER-Roei/davidn/michal/weights/ir_3classes/weights.pt'

print(f'augmentations={augmentations}')
print(f'train_all={train_all}')
print(f'cross_entropy_loss={cross_entropy_loss}')
print(f'focal_loss={focal_loss}')
print(f'using_unet={using_unet}')
print(f'using_michals_unet={using_michals_unet}')
print(f'overfit_data={overfit_data}')


print(f'weighted_loss={weighted_loss}')
print(f'one_ch_in={one_ch_in}')
print(f'norm_with_average_sub={norm_with_average_sub}')
print(f'unify_classes_first_and_third={unify_classes_first_and_third}')
print(f'use_only_single_class={use_only_single_class}')
print(f'use_gradient_accumulation={use_gradient_accumulation}')
print(f'use_initialisation_weights={use_initialisation_weights}')
