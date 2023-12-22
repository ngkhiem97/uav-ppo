
### Installation
```
virtualenv --no-site-packages UAV_PPO --python=python3.6
source ~/UAV_PPO/bin/activate
pip install -r requirements.txt
cd ./
git clone --branch release_18 https://github.com/Unity-Technologies/ml-agents.git
cd ./ml-agents
pip install -e ./ml-agents-envs
pip install gym-unity==0.27.0
```
### Training
 ```
python3 train.py --save-model-interval 5 --env-name navigation --eval-batch-size 0 --min-batch-size 2048 --num-threads 1 --hist-length 5
```
### Testing

[//]: # (Remember to modify the threshold value to 0.5m and 0.25m:)
```
python3 test.py --env-name navigation --eval-batch-size 2000 --hist-length 5
```
