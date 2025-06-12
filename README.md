# PARM 

Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, and Ying-Cong Chen. PARM: Multi-Objective Test-Time Alignment via Preference-Aware Autoregressive Reward Model. In *ICML*, 2025.

## Installation
Our code is based on [TRL](https://github.com/huggingface/trl) and [PEFT](https://github.com/huggingface/peft) for training and [Model_Arithmetic](https://github.com/eth-sri/language-model-arithmetic) for inference. 
```
conda create -n genarm python=3.10
conda activate parm

cd language-model-arithmetic/
pip install -e .

cd ../peft/
pip install -e .

cd ..
pip install -r requirements.txt
```

## Training
```
cd code/training
bash run.sh
```

## Evaluation
```
cd code/evaluation
python generate_outputs.py
python compute_reward.py
```
