# ee449-homework-3---reinforcement-learning-solved
**TO GET THIS SOLUTION VISIT:** [EE449 Homework 3 – Reinforcement Learning Solved](https://www.ankitcodinghub.com/product/ee449-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;112972&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EE449 Homework 3 - Reinforcement Learning  Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
You should prepare your homework by yourself alone and you should not share it with other students, otherwise you will be penalized.

Introduction

This homework consists of some more advanced algorithms (PPO, DQN that are not provided in EE449 course material. You do not need to worry about the content of these algorithms as these algorithms are directly given to you in Stable Baselines library. However, you are encouraged to study these algorithms as they could be beneficial for you in the future. Also, some helper codes are provided to you under HW3 folder in ODTUClass course page.

Homework Task and Deliverables

In the scope of this homework, you will use reinforcement learning algorithms to train an agent to play an Atari game. Your task will be to provide the agent with a few consecutive frames of the game as input, and train it to output a single action that maximizes its long-term cumulative reward. You will use OpenAI Gym to access the Atari game environment and the PyTorch library to implement the algorithms. By the end of the assignment, you will gain a better understanding of how reinforcement learning can be used to solve complex problems such as playing Atari games. Also, you will learn how to track your training using TensorBoard.

The homework is composed of 3 parts. In the first part you will answer some basic questions about RL. In the second part you will use Proximal Policy Optimization (PPO) [4] and Deep Q-Network (DQN) [5] to train your agent. In this part, you will also need to tune the hyperparameters of the algorithms and evaluate the performance of your agents using metrics such as average episode reward and win rate. Finally, you will compare the two algorithm visually and quantitatively and interpret the results by your own conclusions.

You should submit a single report in which your answers to the questions, the required experimental results (performance curve plots, visualizations etc.) and your deductions are presented for each part of the homework. Moreover, you should append your Python codes to the end of the report for each part to generate the results and the visualizations of the experiments. Namely, all the required tasks for a part can be performed by running the related code file. The codes should be well structured and well commented. The non-text submissions (e.g. image) or the submissions lacking comments will not be evaluated. Similarly answers/results/conclusions written in code as a comment will not be graded.

The report should be in portable document format (pdf) and named as hw3 name surname eXXXXXX where name, surname and Xs are to be replaced by your name, surname and digits of your user ID, respectively. You do not need to send any code files to your course assistant(s), since everything should be in your single pdf file.

Do not include the codes in utils.py to the end of your pdf file.

1 Basic Questions

Compare the following terms in reinforcement learning with their equivalent terms in supervised learning (if any) and provide a definition for each, in your own wording:

• Agent

• Environment

• Reward

• Policy

• Exploration

• Exploitation

2 Experimental Work

# Import environment libraries

import gym super mario bros from nespy.wrappers import JoypadSpace from gym super mario bros.actions import SIMPLE MOVEMENT

# Start the environment

env = gym super mario bros.make(’SuperMarioBros-v0’) # Generates the environment env = JoypadSpace(env, SIMPLE MOVEMENT) # Limits the joypads moves with important moves startGameRand(env)

# Import preprocessing wrappers

from gym.wrappers import GrayScaleObservation from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecMonitor from matplotlib import pyplot as plt

# Apply the preprocessing

env = GrayScaleObservation(env, keep_dim=True) # Convert to grayscale to reduce dimensionality env = DummyVecEnv([lambda: env])

env = VecFrameStack(env, 4, channels_order=’last’) # Stack frames env = VecMonitor(env, “./train/TestMonitor”) # Monitor your progress Don’t forget to create “./train/’ directory (or some alternative name) as your CHECKPOINT DIR and “./logs/’ directory as LOG DIR.

2.1 PPO

from utils import SaveOnBestTrainingRewardCallback from stable_baselines3 import PPO

callback = SaveOnBestTrainingRewardCallback(save freq=10000, check freq=1000, chk dir=CHECKPOINT_DIR)

model = PPO(’CnnPolicy’, env, verbose=1, tensorboard log=LOG DIR, learning rate=0.000001, n steps=512) model.learn(total_timesteps=4000000, callback=callback)

2.2 DQN

Repeat 2.1 with DQN algorithm. Make sure Tensorboard is logging ep rew mean and loss properly. from stable_baselines3 import DQN

model = DQN(’CnnPolicy’,

env, batch size=192, verbose=1, learning starts=10000, learning rate=5e-3, exploration fraction=0.1, exploration initial eps=1.0, exploration final eps=0.1, train freq=8, buffer size=10000, tensorboard log=LOG DIR )

model.learn(total timesteps=4000000, log interval=1, callback=callback)

3 Benchmarking and Discussions

3.1 Benchmarking

Play with the hyperparameters of the two abovementioned RL algorithms in order to get the best result. Also play with the preprocessing methods given in Gym and Stable-Baselines3 libraries. Make sure you are comparing PPO and DQN with the same preprocessing methods at least once (try to coincide their best results). Train each PPO and DQN scenario with 3 different hyperparameter, preprocessing methods for at least 1 million timesteps. Using Tensorboard module,

• Plot 3 different PPO scenario for ep rew mean value in one figure,

• Plot 3 different PPO scenario for entropy loss value in one figure,

• Plot 3 different DQN scenario for ep rew mean value in one figure,

• Plot 3 different DQN scenario for loss value in one figure,

• Plot PPO vs DQN comparison for ep rew mean where they are using same preprocessing methods in one figure,

• Plot PPO vs DQN comparison for loss where they are using same preprocessing methods in one figure,

Then, decide on your best algorithm, hyperparameter and preprocessing triplet and train your model for 5 million timesteps in total. Plot figures for ep rew mean and loss values of the final model. Additionally, using the saveGameModel function provided in utils.py, create a video of your best game and upload it to YouTube as unlisted. Share the YouTube link in your homework PDF file.

Finally, change your environment using to gym super mario bros.make(’SuperMarioBrosRandomStages-v0’). This will generate a random stage of the game in each time. Use your best models in each PPO and DQN to play in this environment for couple of times. Observe the generalizability of your models.

3.2 Discussions

Answer the following questions:

1. Watch your agent’s performance on the environment using saved model files at timesteps 0 (random), 10000, 100000, 500000 and 1 million. Could you visually track the learning progress of PPO and DQN algorithms? When did Mario be able to jump over the longer pipes? What are the highest scores your agent could get (top left of the screen) during those timesteps?

2. Compare the learning curves of the PPO and DQN algorithms in terms of the average episode reward over time. Which algorithm learns faster or more efficiently?

3. How do the policies learned by the PPO and DQN algorithms differ in terms of exploration vs exploitation? Which algorithm is better at balancing these two aspects of the learning process?

4. Compare the performances of the PPO and DQN algorithms in terms of their ability to generalize to new environments or unseen levels of the game. Which algorithm is more robust or adaptable?

5. How do the hyperparameters of the PPO and DQN algorithms affect their performance and learning speed? Which hyperparameters are critical for each algorithm, and how do they compare?

6. Compare the computational complexity of the PPO and DQN algorithms. Which algorithm requires more or less computational resources, and how does this affect their practicality or scalability?

7. Considering what you know about Neural Networks by now, if ’MlpPolicy’ was used instead of ’CnnPolicy’, how it would affect the performance of the algorithms in terms of handling highdimensional input states, such as images? Which policy is better suited for such tasks, and why?

4 Remarks

1. On a newer generation GPU, a training process with 1 million steps takes around 2.5 hours. On an average GPU (including the ones in Google Colab), it takes around 4-5 hours. Hence you are strongly advised not to do your homework on the last day of submission.

References

[1] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein,
