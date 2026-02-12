# default
import sys
import os
import datetime
import shutil

class DualLogger(object):
    """
    Writes to both the terminal and a log file simultaneously.
    Buffers partial writes to ensure [INFO] tags only appear on full lines.
    """
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8')
        self.buffer = ""

    def write(self, message):
        """Standard print() calls go here: Terminal + File"""
        self.terminal.write(message)
        if message:
            self.buffer += message
            if "\n" in self.buffer:
                lines = self.buffer.split("\n")
                for line in lines[:-1]:
                    timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
                    self.log.write(f"{timestamp} [INFO]  {line}\n")
                self.buffer = lines[-1]
                self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def debug(self, message):
        """Writes ONLY to the log file, skipping the terminal."""
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        msg_str = str(message).strip() # Clean up input
        self.log.write(f"{timestamp} [DEBUG] {msg_str}\n")
        self.log.flush()

def setup_logging(mode: str):
    """
    Creates a dedicated timestamped folder for the run and redirects logs there.
    Returns: The path to the new run directory.
    """
    base_dir = "out"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, mode, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "run.log")
    if not isinstance(sys.stdout, DualLogger):
        sys.stdout = DualLogger(log_path)
        sys.stderr = sys.stdout
    print(f"--- Run Directory Created: {run_dir} ---")
    print(f"--- Logging Started: {log_path} ---")
    return run_dir

def log_to_file(message):
    """
    Helper function to print debug info ONLY to the log file.
    Usage: log_to_file("This is a secret debug message")
    """
    if hasattr(sys.stdout, 'debug'):
        sys.stdout.debug(message)

def save_configs(run_dir, config_files, hardpoints_name):
    """
    Copies relevant config files to the run directory for reproducibility.
    """
    log_to_file("-> Backing up configuration files...")
    
    for cfg in config_files:
        if os.path.exists(cfg):
            shutil.copy(cfg, run_dir)
            log_to_file(f"Backed up config: {cfg}")

    hp_path = f"config/hardpoints/{hardpoints_name}.yml"
    if os.path.exists(hp_path):
        shutil.copy(hp_path, run_dir)
        log_to_file(f"Backed up hardpoints: {hp_path}")
    else:
        log_to_file(f"WARNING: Could not find hardpoints file: {hp_path}")