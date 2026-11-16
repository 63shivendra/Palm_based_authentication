import os

def list_structure(path, depth=3):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        if level <= depth:
            indent = ' ' * 4 * level
            print(f'{indent}{os.path.basename(root)}/ ({len(files)} files, {len(dirs)} subdirs)')
            if files and level == depth:
                print(f'{indent}    sample: {files[:5]}')

print("Structure of real_palm_dataset:")
list_structure("data/real_palm_dataset", depth=4)

