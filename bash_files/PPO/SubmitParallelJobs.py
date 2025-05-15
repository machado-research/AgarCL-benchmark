import sys
import os
import uuid
import itertools

def modify_and_run_script(original_file, var_options, base_command):
    # Generate all combinations of parameter values
    keys = var_options.keys()
    values = (var_options[key] for key in keys)
    
    for combination in itertools.product(*values):
        var_changes = dict(zip(keys, combination))
        new_file = f"modified_script_{uuid.uuid4().hex[:8]}.sh"
        
        # Read the original script
        with open(original_file, 'r') as file:
            lines = file.readlines()
        
        # Remove last line and prepare new command
        modified_lines = lines[:-1]
        new_command = base_command + " " + " ".join(f"{key} {value}" for key, value in var_changes.items())
        
        # Write new script
        with open(new_file, 'w') as file:
            file.writelines(modified_lines)
            file.write(new_command + "\n")
        
        # Execute the modified script using sbatch
        os.system(f'sbatch {new_file}')

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python script.py <original_file> <base_command> key1=val1,val2 key2=val3,val4 ...")
        sys.exit(1)
    
    original_file = sys.argv[1]
    base_command = "python " + sys.argv[2]
    var_options = {arg.split('=')[0]: arg.split('=')[1].split(',') for arg in sys.argv[3:]}
    modify_and_run_script(original_file, var_options, base_command)
