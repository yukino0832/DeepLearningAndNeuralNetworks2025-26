This repository is for the practice of course "Deep Learning and Neural Networks".

## Minst Detect

### Install
anaconda is required.
```bash
git clone https://github.com/your-username/DeepLearningAndNeuralNetworks2025-26.git && cd DeepLearningAndNeuralNetworks2025-26
source env.sh ./your_env_name
```

### Train

```bash
bash scripts/train.sh
```

### Predict
put images you want to predict in the `image_test` directory, then run
```bash
bash scripts/inference.sh
```
