# Python Environment

1. python version: Python 3.9.23 (main, Jun  5 2025, 13:25:08) [MSC v.1929 64 bit (AMD64)] on win32
2. cuda version: Cuda compilation tools, release 12.6, V12.6.68
   Build cuda_12.6.r12.6/compiler.34714021_0
3. PyTorch version: torch    2.7.1+cu126
4. Nvidia drivers: NVIDIA-SMI 576.52  Driver Version: 576.52
5. GPU: laptop GTX 1660Ti 6GB 	| 	PC RAM: 24GB

### See more about dependencies in requirements.txt

    `pip install -r requirements.txt`

* Help you figure out dependency versions.

---

### Preparing virtual environment (with Conda)

Follow these steps to install all python packages:

```
# Go to repo directory in conda prompt: cd: <....>

conda create -n racecar_env python=3.9

activate racecar_env

pip install numpy matplotlib pyyaml ipython opencv-python gymnasium

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

pip install "gymnasium[box2d]"
```

### Using GPU for PyTorch

You might run into build problems if you run:

```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

In that case, you can just install CPU version with:

```
pip3 install torch torchvision
```

If you really want to use gpu for faster compute, you need to install Nvidia GPU drivers (maybe some other steps if you are unlucky). You might have to install the VScode build tools, and install MVSC etc...

You can find tutorials online...

### Running code from debugger

We will run the train and run program with the **vscode debugger tool**. Here is how to set it up

1. Select python interpreter. In VScode, click on top search bar:` > Python: Select Interpreter` and select **racecar_env**
2. In root folder, create a folder called ".vscode"
3. Copy 'launch.json' from './docs/' to '.vscode/'
4. Click on the debug icon on the left panel
5. Select the launch configurations from the drop down list next to the play button

   ![1753945875396](image/README/1753945875396.png)

   You can first try `Run RaceCar_1`, see if everything is working.

# Code files overview

### config.yml

    You can make changes to hyperparameters in config.yml

### main.py

Deals with the reinforcement learning high level control. Including the training and run loop. The DQN Agent class is defined here. 

In the training loop:

* The environment is observed
* Action is selected by agent through model inference
* Environment transition is recorded in memory
* The policy model is updated
* Epsilon exploration rate is adjusted in real time
* The best model weights are stored

In the run loop:

- Steps through the environment with agent decisions. Epsilon is set to a minimum

In    `if__name__ == "__main__":`

* Program arguments defined in launch.json will be parsed when main.py is ran with arguments
* Decisides whether or not to load policy network from `./runs/best_model_RaceCar_1.pt`. If it loads the policy, the model will continue training from the previous stage, not starting over from random weights

## Demo Notebook (GitHub Viewer)

[View the demo notebook](./Env_Wrapper_Demo.ipynb)
