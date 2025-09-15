### Prereq set up:
1. You'll need to install Python: https://www.python.org/downloads/ **IMPORTANT** make sure you check the box that asks if you want python.exe added to PATH
2. Click the <> Code button above on this repo
3. click Download ZIP
4. Unzip it in your location of choice
5. Enter the baja-suspension folder in your file browser
6. Right click anywhere inside the folder and click 'Open in Terminal'
 * This might not work the same way on windows 10, but the goal is to be in the baja-suspension folder in a terminal. You can also just copy its path and cd to the path, or open the folder in VScode and open a terminal in there
7. Once you're in the terminal, do the following set up steps

### To set up:
1. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate # or 'source venv/bin/activate' on Mac
```
2. Install the dependencies and setup package
```
pip install -e .
```

You will only need to do the set up steps once. After that, to run it, you can:
1. open the baja-suspension folder
2. right click and 'Open in Terminal'
3. Type `venv\Scripts\activate` to activate the venv
4. Then do the following from the 'To Run' section (you can mix and match different parameters as needed, or try the examples below)

### To run:
1. Set the simulation configuration in `sim_config.yml`

- For example, to run a simulation of the **front suspension** with **2026 hardpoints**, set the following in `sim_config.yml`
    ```yml
    HARDPOINTS: '2026'
    TYPE:       'front'

    PLOTS:
    CAMBER:   True
    TOE:      True
    CASTER:   True
    3D:       True
    ```

2. Run the main script
    ```python
    python main.py
    ``` 