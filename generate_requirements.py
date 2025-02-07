import os
import re
import subprocess
import sys
import json
import pkgutil
import importlib.util

def find_imports_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    imports = re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', content, re.MULTILINE)
    return imports

def find_imports_in_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = json.load(file)
    imports = set()
    for cell in content['cells']:
        if cell['cell_type'] == 'code':
            cell_content = ''.join(cell['source'])
            imports.update(re.findall(r'^\s*(?:import|from)\s+([a-zA-Z0-9_\.]+)', cell_content, re.MULTILINE))
    return imports

def install_and_freeze_packages(packages, requirements_file):
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    with open(requirements_file, 'w', encoding='utf-8') as file:
        subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=file, check=True)

def is_standard_library(module_name):
    if module_name in sys.builtin_module_names:
        return True
    if importlib.util.find_spec(module_name) is None:
        return True
    return False

if __name__ == "__main__":
    files_to_check = [
        "./app/app_local.py",
        "./app/main.py",
        "./app/test.py",
        "./tests/test_api.py",
        "./tests/verify_tokenizer.py",
        "./Creuse_Tom_2_scripts_notebook_modélisation_122024.ipynb"
    ]
    
    imports = set()
    for file_path in files_to_check:
        if file_path.endswith('.ipynb'):
            imports.update(find_imports_in_notebook(file_path))
        else:
            imports.update(find_imports_in_file(file_path))
    
    # Filter out standard library modules and submodules
    filtered_imports = {imp.split('.')[0] for imp in imports if not is_standard_library(imp.split('.')[0])}
    
    # Check if the virtual environment already exists
    if not os.path.exists("myenv"):
        # Create a new virtual environment
        subprocess.run([sys.executable, "-m", "venv", "myenv"], check=True)
    
    # Activate the virtual environment
    activate_script = os.path.join("myenv", "Scripts", "activate.bat") if os.name == 'nt' else os.path.join("myenv", "bin", "activate")
    
    # Install packages in the virtual environment
    if os.name == 'nt':
        activate_command = f"myenv\\Scripts\\activate && python -m pip install --upgrade pip"
    else:
        activate_command = f"source myenv/bin/activate && python -m pip install --upgrade pip"
    
    for package in filtered_imports:
        activate_command += f" && python -m pip install {package}"
    
    subprocess.run(activate_command, shell=True, check=True)
    
    # Generate requirements.txt
    requirements_file = "requirements.txt"
    pip_path = os.path.join("myenv", "Scripts", "pip.exe") if os.name == 'nt' else os.path.join("myenv", "bin", "pip")
    with open(requirements_file, 'w', encoding='utf-8') as file:
        subprocess.run([pip_path, "freeze"], stdout=file, check=True)
    
    print(f"Requirements have been written to {requirements_file}")