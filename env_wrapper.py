import gymnasium as gym
import numpy as np
import cv2
import yaml
from collections import deque
from IPython.display import display
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode

class MyRaceCarEnv():
    def __init__(self, hyperparameter_set, render_mode='none'):
        """
        Args:
            hyperparameter_set: Name of the hyperparameter set in config.yml
            render_mode: 'none', 'jupyter', or 'opencv'
        """
        yaml_path = "./config.yml"
        with open(yaml_path, 'r') as file:
            yaml_all = yaml.safe_load(file)
            hyperparameter = yaml_all[hyperparameter_set]
            
        self.start_frame = 55  # skip the frames when zooming in
        self.skip_frames = hyperparameter['skip_frames']
        self.render_mode = render_mode.lower()
        
        # Validate render mode
        if self.render_mode not in ['none', 'jupyter', 'opencv']:
            raise ValueError("render_mode must be 'none', 'jupyter', or 'opencv'")

        self.env = gym.make("CarRacing-v3", continuous=False)
        self.action_dim = self.env.action_space.shape
        self.state_dim = [
            hyperparameter['stack_frames'], 
            hyperparameter['resize_height'], 
            hyperparameter['resize_width']
        ]
        
        self.stack_frames_count = self.state_dim[0]
        self.state_preprocessor = StatePreprocessor(
            self.stack_frames_count, 
            self.state_dim[1], 
            self.state_dim[2]
        )
        
        # Rendering setup
        self.fig = None
        self.img_window = None
        self.ax = None
        self.window_name = "RaceCar Environment"
        
        if self.render_mode == 'opencv':
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 500, 500)

    def initialize_env(self):
        '''similar to reset, but skip starting frames when zooming in'''
        state, _ = self.env.reset()
        reward = 0
        
        # Skip initial frames
        for _ in range(self.start_frame):
            next_state, reward, terminated, truncated, _ = self.env.step(
                self.env.action_space.sample()
            )
            if terminated or truncated:
                break
        
        # Initialize stacked state
        stacked_state = self.state_preprocessor.reset(next_state)
        
        # Initialize rendering if needed
        if self.render_mode == 'jupyter':
            self._init_jupyter_render()
        elif self.render_mode == 'opencv':
            self._render_opencv(next_state)
            
        return stacked_state
        
    def step(self, action):
        '''same as env.step(), but integrated with frame skipping and rendering'''
        total_reward = 0
        terminated = truncated = False
        last_state = None

        for _ in range(self.skip_frames):
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            self.state_preprocessor.add_frame(next_state)
            total_reward += reward
            last_state = next_state
            
            # Render if enabled
            if self.render_mode != 'none':
                self._render_frame(self.state_preprocessor.preprocess_state(next_state))
            
            if terminated or truncated:
                break

        next_stacked_state = self.state_preprocessor.get_stacked_frames()
        return next_stacked_state, total_reward, terminated, truncated, {}

    def _init_jupyter_render(self):
        """Initialize matplotlib figure for Jupyter rendering"""
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(5, 5))
        self.img_window = self.ax.imshow(
            np.zeros((self.state_dim[1], self.state_dim[2])), 
            cmap='gray', 
            vmin=0, 
            vmax=1
        )
        self.ax.set_title("Environment Viewer")
        self.ax.axis('off')
        plt.tight_layout()
        plt.show(block=False)
        # display(self.fig)
        plt.ioff()

    def _render_frame(self, frame):
        """Render the current frame based on the selected mode"""
        if self.render_mode == 'jupyter':
            self._render_jupyter(frame)
        elif self.render_mode == 'opencv':
            self._render_opencv(frame)

    def _render_jupyter(self, frame):
        """Update Jupyter display with new frame"""
        self.img_window.set_data(frame)
        self.img_window.set_clim(0, 1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _render_opencv(self, frame):
        """Render frame using OpenCV"""
        # Resize for better display
        display_frame = cv2.resize(frame, (500, 500))
        # Convert BGR to RGB for OpenCV
        # display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, display_frame)
        cv2.waitKey(1)

    def close(self):
        """Clean up resources"""
        if self.render_mode == 'opencv':
            cv2.destroyWindow(self.window_name)
        elif self.render_mode == 'jupyter' and self.fig is not None:
            plt.close(self.fig)
        self.env.close()
        del self


class StatePreprocessor():
    def __init__(self, stack_frames_count, resize_width=94, resize_height=94):
        self.stack_height = stack_frames_count
        self.frames = deque(maxlen=stack_frames_count)
        self.resize_width = resize_width
        self.resize_height = resize_height

    def preprocess_state(self, state):
        # 1. Crop FIRST to avoid resizing artifacts
        crop_up, crop_down = 0, 11
        crop_left, crop_right = 0, 0
        cropped = state[crop_up:-(crop_down+1), crop_left:-(crop_right+1)]
        
        # 2. Convert to grayscale using luminance-preserving weights
        # (Standard ITU-R BT.601 weights)
        grayscale = 0.299 * cropped[...,0] + 0.587 * cropped[...,1] + 0.114 * cropped[...,2]
        
        # 3. Resize after cropping
        resized = cv2.resize(grayscale, (self.resize_width, self.resize_height), 
                            interpolation=cv2.INTER_AREA)
        
        # 4. Fixed normalization range (0-255 assumed)
        normalized = resized / 255.0
        
        # Optional: Contrast enhancement
        # normalized = np.clip((normalized - 0.5) * 1.5 + 0.5, 0, 1)
        
        return normalized

    def add_frame(self, frame):
        processed_frame = self.preprocess_state(frame)
        self.frames.append(processed_frame)
    
    def get_single_frame(self, index):
        assert index < len(self.frames), "Index out of range"
        return self.frames[index]

    def get_stacked_frames(self):
        return np.stack(self.frames, axis=0)

    def reset(self, frame):
        self.frames.clear()
        for i in range(self.stack_height):
            self.add_frame(frame)
        return self.get_stacked_frames()

def initialize_image_window(img_arr, color=False):
    # Set up the figure and axis
    if color == False:
        fig, ax = plt.subplots(figsize=(5, 5))
        img = ax.imshow(img_arr, cmap='gray', vmin=0, vmax=1)
        plt.title("Environment Viewer")
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking display
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))  # Two subplots in a row
        
        # Display first image (grey input)
        img1 = ax1.imshow(img_arr, cmap='gray', vmin=0, vmax=1)
        ax1.set_title("Image 1")
        ax1.axis('off')
        # Display second colored image
        img2 = ax2.imshow(img_arr)
        ax2.set_title("Image 2")
        ax2.axis('off')
        
        plt.suptitle("Environment Viewer")
        plt.tight_layout()
        plt.show(block=False)  # Non-blocking display
        img = [img1, img2]
        ax = [ax1, ax2]

    return fig, img, ax

    
def plot_stacked_images(image_list):
    """
    Plot stacked images in rows with index information.
    Parameters:
    image_list (list of numpy arrays): List of stacked images with shape (4, X, X)

    """
    plt.ioff()
    num_images = len(image_list)
    # Create a figure with rows equal to number of images and 4 columns
    fig, axes = plt.subplots(num_images, 4, figsize=(12, 3*num_images))
    # If there's only one image, axes will be 1D
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i, stacked_img in enumerate(image_list):
        if stacked_img.shape[0] != 4:
            raise ValueError(f"Image at index {i} has shape {stacked_img.shape}, expected first dimension to be 4")
        for j in range(4):
            ax = axes[i, j]
            ax.imshow(stacked_img[j], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Channel {j+1}')
            ax.axis('off')
            if j == 0:  # Only add to first image in row
                ax.text(0.05, 0.95, f'Index {i}', 
                        transform=ax.transAxes, 
                        color='white', 
                        fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    plt.tight_layout()
    # plt.show() # do not plot in tk window
    display(fig)  # This shows the plot in Jupyter
    plt.ion()
