# Mutli-agent task allocation

This code uses generative adversarial networks to generate diverse task allocation plans for Multi-agent teams. 

To change hyperparameters, check out `params.py`.
Specifically, `params['sim_env']` controls whether we are using the toy environment (with hand-crafted rewards) or the ergodic search environment.


To train the allocation generator and discriminator with the pre-trained reward network weight  (as a surrogate approximation to speed up training), run
```sh
python train.py
```

To test the allocation generator, relocate trained weights as `logs/test_weights/generator_weight`, and run
```sh
python test_alloc.py
```

(Optional) To retrain the reward network weight, run:
```sh
python train_simulation_reward.py
```
Put the trained weight in `logs/reward_logs/reward_weight` for training.

The training data for the reward network is stored in 
`logs/training_data/*.npy`