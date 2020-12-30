# Generative Adversarial Networks with Weight Normalization and ResNet

A GAN implementation based on the paper [*On the Effects of Batch and Weight Normalization in Generative Adversarial Networks*](https://arxiv.org/abs/1704.03971)

One major change was adding some tweaks so that the method could work with a residual network. For this I've changed the architecture substantially, completely switched to pytorch and removed any batchnorm code, thus I decided that I should open a new repository and keep the old one for reference (which also means I will not fix the numerous bugs there).

I will be testing on direct training on CelebA-HQ, the result of which will determine whether I'll even bother documenting this code. 

If you want to try it now:

```
python split_data.py --dataset folder --dataroot /path/to/img_align_celeba --test_num 200

python main.py --dataset folder --dataroot /path/to/img_align_celeba --image_size 160 --crop_size 160 --dis_feature 64 128 256 384 512 --dis_block 1 1 1 1 1 --gen_feature 64 128 256 384 512 --gen_block 1 1 1 1 1 --save_path /some/path
```

*The code is for Python 3*
