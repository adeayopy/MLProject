# Optimization as a Model for Few-shot Learning
Pytorch implementation of [Optimization as a Model for Few-shot Learning](https://openreview.net/forum?id=rJY0-Kcll) in ICLR 2017 (Oral)


## Prerequisites
- python 3+
- pytorch 0.4+ (developed on 1.0.1 with cuda 9.0)
- [pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [tqdm](https://tqdm.github.io/) (a nice progress bar)

## Data
  - You can download it from [here](https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR/view?usp=sharing) (~2.7GB, google drive link)

## Preparation
- Make sure Mini-Imagenet is split properly. For example:
  ```
  - data/
    - miniImagenet/
      - train/
        - n01532829/
          - n0153282900000005.jpg
          - ...
        - n01558993/
        - ...
      - val/
        - n01855672/
        - ...
      - test/
        - ...
  - main.py
  - ...
  ```
  - It'd be set if you download and extract Mini-Imagenet from the link above
- Check out ``, make sure `--data-root` is properly set in main.py

## Run
For 5-shot, 5-class training, run
```bash
python main.py
```
Hyper-parameters are referred to the [author's repo](https://github.com/twitter/meta-learning-lstm).

For 5-shot, 5-class evaluation, run *(remember to change `--resume` and `--seed` arguments)*
Also change the mode to test intead of train
```bash
python main.py
```


## References
- [markdtw](https://github.com/markdtw/meta-learning-lstm-pytorch) (Implementation guide)
- [CloserLookFewShot](https://github.com/wyharveychen/CloserLookFewShot) (Data loader)
- [pytorch-meta-optimizer](https://github.com/ikostrikov/pytorch-meta-optimizer) (Casting `nn.Parameters` to `torch.Tensor` inspired from here)
- [meta-learning-lstm](https://github.com/twitter/meta-learning-lstm) (Author's repo in Lua Torch)

