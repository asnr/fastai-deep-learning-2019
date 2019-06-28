# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Notes
#
# Training on top of Resnet34 gave rubbish results, roughly the same as flipping a coin. Fine tuning Resnet34 gave decent results though, error rate of 10% on validation set. Training on top of Resnet50 gave crazy good results, no errors on the validation set (I mean, the validation set is really small, so say <5% error?).
#
# My guess is that Resnet34 doesn't have features that can effectively distinguish between soccer and footy. However, a little bit of fine tuning is sufficient to build those features. Resnet50 on the other hand already has features that can distinguish between the two.

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *
from fastai.metrics import error_rate

bs = 64

images_path = 'soccer_vs_footy_images'
image_filenames = get_image_files(images_path)
image_filenames[:4]

np.random.seed(2)
pat = r'/([^/]+)_\d+.(jpg|jpeg|JPG|JPEG|png)$'

data = ImageDataBunch.from_name_re(
    images_path, image_filenames, pat, ds_tfms=get_transforms(), size=224, bs=bs
).normalize()

data.show_batch(rows=3, figsize=(8,7))

print(data.classes)

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')

# +
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
# -

interp.plot_top_losses(9, figsize=(15,11), heatmap=False)

interp.plot_confusion_matrix(figsize=(12,12), dpi=60)

# ## Customising

learn.load('stage-1')

learn.unfreeze()

learn.lr_find()

learn.recorder.plot()

# Validation loss is still going down in the fourth epoch.
# Accuracy isn't going up, but that might be just because the validation set is tiny.
learn.fit_one_cycle(4, max_lr=slice(1e-4,1e-3))

# +
interp_fine_tuned = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)
# -

interp_fine_tuned.plot_confusion_matrix(figsize=(12,12), dpi=60)

interp_fine_tuned.plot_top_losses(2, figsize=(11,11), heatmap=False)

# ## Resnet 50

data = ImageDataBunch.from_name_re(
    images_path, image_filenames, pat, ds_tfms=get_transforms(), size=299, bs=bs//2
).normalize()

learn = cnn_learner(data, models.resnet50, metrics=error_rate)

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8)
