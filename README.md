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
```
python main.py --hardpoints <year> --sim_type <type>
```

**Parameters:**
- `--hardpoints`: Vehicle hardpoints yaml located under `/hardpoints` -> [`2021`, `2024`, or `2026`]
- `--sim_type`: Analysis type [`front`, `rear`, or `vehicle`]

**Examples:**

Analyze front double A-arm suspension with 2026 hardpoints file
```
python main.py --hardpoints 2026 --sim_type front
```

Analyze rear semi-trailing link suspension with 2024 hardpoints file
```
python main.py --hardpoints 2024 --sim_type rear
```
