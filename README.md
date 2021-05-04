# Mutli-agent task allocation

This code uses generative adversarial networks to generate diverse task allocation plans for a toy MRTA example with hand-crafted rewards. 

To train the allocation generator and discriminator, run
```sh
python train.py
```
To change hyperparameters, check out `params.py`.

To test the allocation generator, put trained weights in `./test_weights`, and run
```sh
python test_alloc.py
```