# BlechRNN

Neural network pipeline to estimate neural firing rates from spike train data using continuous-time RNNs.

## Architecture
The model uses an autoencoder architecture with three main components:

- **Encoder**: Projects input spike data to latent space
  - Reduces dimensionality of spike train data
  - Learns efficient compressed representations
  - Implemented as a feedforward neural network

- **RNN**: Learns temporal dynamics in latent space 
  - Continuous-time RNN (CTRNN) architecture
  - Captures temporal dependencies in neural activity
  - Supports variable time steps via dt parameter
  - Optional bidirectional processing

- **Decoder**: Projects latent representations back to firing rate space
  - Reconstructs firing rate estimates from latent space
  - Matches input dimensionality
  - Includes rectification to ensure non-negative rates

## Key Features
- Continuous-time RNN implementation
- Optional dropout for regularization
- PCA preprocessing of spike data
- MSE and Poisson loss functions
- Cross-validation during training
- GPU acceleration support

## Additions by Vincent Calia-Bogan as of 7/11/2025: 

  ## Inferred Firing Rate Extraction and Visualization:
  - Generates predicted firing rates from trained RNN models  
  - Automatically reverses PCA and z-scoring (if enabled) to restore predictions to neuron space when possible  
  - Produces per-taste and per-neuron performance visualizations:
    - Spike raster
    - Convolved firing rate
    - RNN-predicted firing rate  
  - Saves predicted firing rates and latent states to:
    - HDF5 files (under `/rnn_output`)
    - `.parquet` files (for both firing and latent activity)  
  
  ## Latent Vector Storage:
  - Saves latent trajectories (`latent_out`) from the RNN per trial and taste  
  - Latents saved as `.parquet` using Polars, with columns:
    - `taste`
    - `trial`
    - `time`
    - `latent_dim_*`  
  - Saved in both:
    - The dataset’s `artifacts/` folder
    - The global `pred_latent/` output directory  
  
  ## New Plot Outputs:
  - Train/test loss curves (`loss.png`, `run_loss.png`)  
  - Heatmaps of predicted vs. true firing rates (both raw and z-scored)  
  - Per-taste mean neuron activity plots  
  - Individual neuron diagnostic plots:
    - Spike raster
    - Convolved firing
    - RNN output  
  - Trial-wise latent factor time series plots  
  - Taste-wise mean activity overlays across neurons  
  
  ## Directory Structure Assumption:
  The updated script assumes each `.h5` file is stored in its own directory:
   `` h5_dir/
    ├── DatasetA/
    │ └── DatasetA.h5
    ├── DatasetB/
    │ └── DatasetB.h5
    ``
  Only directories containing exactly one `.h5` file are processed. Others are skipped with a warning.
  Nested dirs were enforced to transparently enable serial processing of multiple datasets automatically without making modifications to `ephys_data.py`. 
  
  If you have multiple .h5 files in a single tld, run `redist_h5.py`, which will take those files and copy them into nested sub-dirs that are appropriate for automated processing. 
  
  ## Modules
  - `get_data.py`: Loads and preprocesses spike train data from .h5 files. This has been deprecated and does not exist in this branch of my fork of Abuzar Mahmood's main branch 
  - `model.py`: Implements CTRNN and autoencoder network architectures
  - `train.py`: Handles model training with cross-validation
  - `run_model.py`: End-to-end pipeline for data loading, training and evaluation, extracting predicted firing rates and latents, as well as plotting all initial outputs
  - `redist_h5.py`: Helper function to create nested directory structure required for serial processing of multiple datasets.
  - `ephys_data.py`: Comprehensive helper function written by Abuzar Mahmood that seemlessly extracts relevant data from bulky .h5 files, among other functions.
      - Single modification made to ephys_data: modified get_hdf5_path to seemlessly handle multiple .h5 files in a single dir. Original class available from AbuzarMahmood/BlechClust. 
        See:
      ```python
          @staticmethod
              def get_hdf5_path(data_dir):
                  """
                  # Look for the hdf5 file in the directory
                  """
                  hdf5_path = glob.glob(
                          os.path.join(data_dir, '**.h5'))
                  if not len(hdf5_path) > 0:
                      raise Exception('No HDF5 file detected')
                  elif len(hdf5_path) > 2:
                      selection_list = ['{}) {} \n'.format(num,os.path.basename(file)) \
                              for num,file in enumerate(hdf5_path)]
                      selection_string = \
                              'Multiple HDF5 files detected, please select a number:\n{}'.\
                                      format("".join(selection_list))
                      file_selection = input(selection_string)
                      return hdf5_path[int(file_selection)]
                  else:
                      return hdf5_path[0]
      ```
  - `visualize.py`: Comprehensive plotting function written by Abuzar Mahmood that is extensively used in `run_model.py`
  - `blechrnn_config`: JSON-based configuration for model settings and file pathing. Useful for SSH-based modification of runtime settings when managing remotely.
  
  ## Outputs Per Dataset
  
  Each dataset produces, for each taste:
  - `/plots/`  
    - Inputs
    - Predicted vs. true firing rates
    - Latents
    - Loss curves
    - Per-neuron diagnostic plots
  
  - `/artifacts/`  
    - Trained model (`.pt`)
    - Loss/cross-val history (`.json`)
    - Raw latent vectors (`.parquet`)
    - Raw predicted firing (`.parquet`)  
  
  - `/rnn_output/` (inside original HDF5 file)  
    - `pred_firing`
    - `latent_out`
    - `pred_x`  
  
  ## Configuration Highlights (Specific to `run_model.py`)
  
  - `hidden_size`: Hidden state size of RNN  
  - `train_steps`: Number of training iterations  
  - `loss_name`: Loss type (`mse`, `smooth`, etc.)  
  - `bin_size`: Binning window in ms  
  - `use_pca`: Whether to apply PCA preprocessing  
  - `retrain`: Whether to overwrite and retrain even if saved model exists. If model exists and `retrain = False`, will skip re-training the model. 
  
  ## Caveats of Vincent's updates (end of section of Vincent's updates):
  - Reverse PCA only applied if model output dimension matches PCA input dimension  
  - Neuron-space reconstruction is approximate when PCA is applied  
  - Stimulus timing is hardcoded relative to `time_lims`; check if offset aligns with experiment  
  - Some plots assume a `bin_size` of 25 ms; update config to change  

## Usage

### Quick Start
The main pipeline follows these steps:
1. Operates on a per-taste basis (for a single dataset, re-trains per taste to avoid muddling infrence from different stimuli)
2. Loads spike train data and bins it
3. Preprocesses using PCA and scaling
4. Splits data into train/test sets
5. Trains the model with cross-validation
6. Saves trained model and generates performance plots
7. Extracts latent vectors
8. Regenerates firing rate data and extracts it
9. Saves both firing and latent vectors to file

### Detailed Example

```python
from get_data import load_spike_data
from model import CTRNN_plus_output
from train import train_model
import torch

# Load and preprocess data
spike_data = load_spike_data('path/to/data.h5')
input_size = spike_data.shape[-1]

# Create model
model = CTRNN_plus_output(
    input_size=input_size,
    hidden_size=64,
    output_size=input_size,
    dt=0.1
)

# Train model
trained_model = train_model(
    net=model,
    inputs=spike_data,
    labels=spike_data,  # For autoencoder
    output_size=input_size,
    train_steps=1000,
    lr=0.01
)

# Save model
torch.save(trained_model.state_dict(), 'trained_model.pth')
```

### Configuration Options

Key parameters that can be tuned:
- `hidden_size`: Number of hidden neurons in CTRNN
- `dt`: Time step for continuous-time dynamics
- `train_steps`: Number of training iterations
- `lr`: Learning rate for optimization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/blechRNN.git
cd blechRNN
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
All dependencies are listed in requirements.txt:
- PyTorch (2.3.1) - Deep learning framework
- NumPy (2.0.0) - Numerical computing
- Scikit-learn (1.4.0) - Data preprocessing
- Matplotlib (3.8.2) - Visualization
- SciPy (1.14.0) - Scientific computing
- tqdm (4.66.1) - Progress bars
## Contributing

We welcome contributions to BlechRNN! Here's how you can help:

### Reporting Issues
- Use the GitHub issue tracker
- Include detailed description and steps to reproduce
- Attach relevant data samples if possible

### Development Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add docstrings for new functions/classes
- Comment complex algorithms
- Keep functions focused and modular

### Running Tests
```bash
python -m pytest tests/
```
