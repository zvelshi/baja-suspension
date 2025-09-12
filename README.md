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

### To run:
```
python main.py --hardpoints <year> --sim_type <type>
```

**Parameters:**
- `--hardpoints`: Vehicle configuration year (`2021`, `2024`, or `2026`)
- `--sim_type`: Suspension analysis type (`front`, `rear`, or `vehicle`)

**Examples:**

Analyze front double A-arm suspension with 2026 geometry
```
python main.py --hardpoints 2026 --sim_type front
```

Analyze rear semi-trailing link suspension with 2024 geometry
```
python main.py --hardpoints 2024 --sim_type rear
```
