import os
import json

def create_json(sweep_folder, save_path):
    file_names = os.listdir(os.path.join(sweep_folder, 'LIDAR_TOP'))
    file_dict = {}
    for i, file_name in enumerate(file_names):
        file_dict[f"sweep_{i}"] = file_name
    print(f'Dicted {len(file_dict)} sweep bin files.')
    with open(save_path, 'w') as f:
        json.dump(file_dict, f, indent = 4)
    print('Saved the required json file')

def load_json(json_path, sweep_folder):
    with open(json_path, 'r') as f:
        file_dict = json.load(f)
    
    print(f'Got json with {len(file_dict)} files')
    # Check if the files exist in the sweep folder
    sweep_folder = os.path.join(sweep_folder, 'LIDAR_TOP')
    for key in file_dict:
        file_name = file_dict[key]
        file_path = os.path.join(sweep_folder, file_name)
        if not os.path.exists(file_path):
            print(f'file: {file_name} doesnt exist.')
            break
    
    print('Checking existense of file complete.\n')
    return file_dict


if __name__ == '__main__':
    sweep_folder = '/home/saksham/samsad/mtech-project/datasets/nuscenes/sweeps'
    save_path = '/home/saksham/samsad/mtech-project/datasets/nuscenes/sweeps.json'
    if not os.path.exists(save_path):
        create_json(sweep_folder, save_path)
    
    file_dict = load_json(save_path, sweep_folder= sweep_folder)
