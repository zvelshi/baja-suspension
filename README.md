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
1. Set the simulation configuration in `kin_config.yml`

- For example, to run a simulation of the **front suspension** with **2026 hardpoints**, set the following in `kin_config.yml`
    ```yml
    HARDPOINTS: '2026'
    SIM_STEPS:  330 
    SIMULATION: 'steer_travel' # 'steer', 'steer_travel', 'travel', 'ackermann'

    HALF: 'front' # 'front' or 'rear'
    SIDE: 'right' # 'left' or 'right'

    TRAVEL:
    MIN: -90  # [mm] Min range to sweep TRAVEL through during sim 
    MAX:  240 # [mm] Max for range to sweep TRAVEL through during sim

    STEER: 
    MIN: -40 # [mm] Min for range to sweep STEER through during sim
    MAX:  40 # [mm] Max for range to sweep STEER through during sim

    PLOTS:
    3D:           True
    CAMBER:       True
    TOE:          True
    CASTER:       True
    AXLE_PLUNGE:  True
    AXLE_ANGLES:  True
    ```

2. 
- Run the single simulation script via
    ```python
    py main.py sim
    ```
- Run the global optimizer script via
    ```python
    py main.py opt
    ```