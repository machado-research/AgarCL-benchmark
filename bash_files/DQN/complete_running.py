import sys
import os
import uuid
import json

def find_checkpoints(base_folder):
    folder_paths = []
    checkpoints = []
    steps = []
    for root, dirs, files in os.walk(base_folder):
        for dir in dirs:
            if any(sub_dir.endswith("_checkpoint") or sub_dir.endswith("_except") for sub_dir in os.listdir(os.path.join(root, dir))):
                max_checkpoint = None
                max_number = -1
                for sub_dir in os.listdir(os.path.join(root, dir)):
                    if sub_dir.endswith("_checkpoint") or sub_dir.endswith("_except"):
                        try:
                            number = int(sub_dir.split("_")[0])
                            if number > max_number:
                                max_number = number
                                max_checkpoint = os.path.join(root, dir, sub_dir)
                        except ValueError:
                            continue
                # If both "_checkpoint" and "_except" exist, take the one with the higher number
                checkpoint_candidates = [
                    os.path.join(root, dir, sub_dir)
                    for sub_dir in os.listdir(os.path.join(root, dir))
                    if sub_dir.endswith("_checkpoint") or sub_dir.endswith("_except")
                ]
                max_checkpoint = max(
                    checkpoint_candidates,
                    key=lambda x: int(os.path.basename(x).split("_")[0]),
                    default=None
                )
                if max_checkpoint:
                    csv_path = os.path.join(root, dir, "episodic_rewards.csv")
                    checkpoints.append(max_checkpoint)
                    if os.path.exists(csv_path):
                        with open(csv_path, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                checkpoint_number = int(os.path.basename(max_checkpoint).split("_")[0])
                                filtered_lines = [
                                    line for line in lines
                                    if line.strip().split(",")[1].isdigit() and int(line.strip().split(",")[1]) <= checkpoint_number + 5000
                                ]
                                if filtered_lines:
                                    # Overwrite the episodic_rewards.csv file with the filtered lines
                                    with open(csv_path, 'w') as f:
                                        f.writelines(filtered_lines)
                                    
                                    # Find the maximum step value from the filtered lines
                                    last_step = max(int(line.strip().split(",")[1]) for line in filtered_lines) + 1
                                    steps.append(last_step)
                                else:
                                    steps.append(checkpoint_number)
                                folder_paths.append(os.path.join(root, dir))
                    else: 
                        steps.append(0)
                        folder_paths.append(os.path.join(root, dir))

    return checkpoints, steps, folder_paths


def get_args(checkpoint_path):

        args_file = os.path.join(os.path.dirname(checkpoint_path), "args.txt")
        if not os.path.exists(args_file):
            raise FileNotFoundError(f"args.txt not found in {os.path.dirname(checkpoint_path)}")

        with open(args_file, "r") as f:
            args = json.load(f)

        return args

def modify_and_run_script(base_folder_sh,base_command, checkpoints_path):
    checkpoints, steps, folder_paths = find_checkpoints(checkpoints_path)
    
    for i, checkpoint in enumerate(checkpoints):
        new_file = f"modified_script_{uuid.uuid4().hex[:8]}.sh"
        
        # Prepare the command with the checkpoint path
        new_command = f"{base_command} --load {checkpoint} --step-offset {steps[i]} --outdir {folder_paths[i]}"
        args = get_args(checkpoint)

        args_string = " ".join([f"--{key} {value}" for key, value in args.items() if key != "step_offset"])
        new_command += f" {args_string}"
        with open(base_folder_sh, 'r') as file:
            lines = file.readlines()
        
        modified_lines = lines[:-1]
        with open(new_file, 'w') as file:
            file.writelines(modified_lines)
            file.write(new_command + "\n")
        
        # # Execute the modified script using sbatch
        # os.system(f'sbatch {new_file}')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <base_folder> <base_command> <checkpoint_path>")
        sys.exit(1)
    
    base_folder_sh = sys.argv[1]
    base_command = "python " + sys.argv[2]
    checkpoint_path = sys.argv[3] if len(sys.argv) > 3 else None

    if checkpoint_path and not os.path.isdir(checkpoint_path):
        print(f"Error: The checkpoint path '{checkpoint_path}' does not exist or is not a directory.")
        sys.exit(1)

    modify_and_run_script(base_folder_sh,base_command, checkpoint_path)
