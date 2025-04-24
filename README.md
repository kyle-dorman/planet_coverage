# Download Planet Data

Repo for downloading PlanetScope data based on a set of predefined grids. 

## Initial Setup

Some one time repo initialization work. 

### Install Conda
You can install miniconda if on Mac or Linux. On Windows install Anaconda.

#### Miniconda
Follow directions [HERE](https://docs.anaconda.com/miniconda/install/)

For Mac, perfer to install miniconda using brew. 
```bash
brew doctor
brew update
brew upgrade
brew upgrade --cask --greedy
brew install --cask miniconda
```

#### Anaconda
Follow directions [HERE](https://docs.anaconda.com/anaconda/install/)

### Open Terminal
On Windows, open an Andaconda Terminal, on Linux/Mac open a regular terminal. 

### Install Git
Check if git is installed. It will return something e.g. `/usr/bin/git` if git is installed. 
```bash
# Linux/Mac
which git
# Windows
where git
```

If git is not installed, install it. 
```bash
# Windows
conda install git
# Mac
brew install git
```

### Clone repo
This command will create a new folder `planet_download` in your terminal's current directory. If you want it installed somewhere specific, move to that folder first (`cd SOMEWHERE/ELSE`)
```bash
git clone git@github.com:kyledorman/planet_download.git
```

After cloning the repo, enter the folder
```bash
cd planet_download
```

### Create conda environment
```bash
conda env create -f environment.yml
```

### Activate Jupyter Widgets
```bash
conda activate planet_download
jupyter nbextension enable --py widgetsnbextension
```

## Download data

### Activate conda environment
```bash
conda activate planet_download
```

### Edit the config file
See the file `src/config.py` to see all configuration options. You ***MUST*** set the `grid_dir` and `save_dir` variables. You should also add a `processing_dir` if you wish the data to be copied for further Neural Network processing. Feel free to edit the `config.yaml` directly. 

`grid_dir` - The path to a folder of geojson grid files in the wgs84 CRS. 
`save_dir` - The path to a folder where you want to save the data. During the download process, data will be saved to folders following the convention: `save_dir/YEAR/MONTH/GRID_ID`
`processing_dir` - The path to a folder where you process the surface reflectance and UDM data with a Neural Network. e.g. `Y:\planet\stateMap\processing`

### Planet API Key
Look up your API key from the [Planet Account Settings](https://www.planet.com/account/#/user-settings). 

### Option 1: Run download script via Jupyter
1. Launch jupyter notebook
```bash
jupyter notebook --notebook-dir=notebooks --port=8892
```
2. Open the `run.ipynb`.
3. Set the `MONTH`, `YEAR`, `CONFIG_FILE`, and `PL_API_KEY` variables. 
4. Run the remaining cells

### Option 2: Run download script via CLI
Inpsect `run.py` script to see how it can be used.
```bash
python src/scripts/run.py --help
```

You will be prompted to enter your Planet API Key the first time you run the script.

#### Example 1: 
Download files for a single month/year combination
```bash
python src/scripts/run.py --config-file config.yaml --year 2022 --month 11
```
This will download data for the year 2022 and the month of November (11).

#### Example 2: 
Download files for multiple month/year combinations
```bash
python src/scripts/run.py --config-file config.yaml --year 2020 --year 2021 --year 2022 --month 09 --month 10 --month 11
```
This will download data for 3 years (2020, 2021, 2022) and 3 months (Set, Oct, Nov) for a total of 9 year/month combindations. You must provide at least 1 year and 1 month. 

## Inspect Results
You can inspect the results of a download using an included jupyter notebook. 

Launch jupyter notebook
```bash
jupyter notebook --notebook-dir=notebooks --port=8892
```
Run the notebook `inspect.ipynb` to visualize the downloaded results. 

## Format code
```bash
conda activate planet_download
./lint.sh
```

## Update dependencies
After changing the environment.yml file, run
```bash
conda activate planet_download
conda env update --file environment.yml --prune
conda activate planet_download
```

## TODOs
- Am I handling grids with overlapping zones correctly?
- Discuss cloud filtering logic changes w/Kate
