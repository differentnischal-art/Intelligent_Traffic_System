import os

labels_dir = "dataset/Ambulance labelling.v2i.yolov8/train/labels"
fixed = 0

for f in os.listdir(labels_dir):
    if not f.endswith('.txt'):
        continue
    path = os.path.join(labels_dir, f)
    with open(path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    changed = False
    for line in lines:
        parts = line.strip().split()
        # valid seg line needs at least 7 numbers (class + 3 coordinate pairs)
        if len(parts) >= 7:
            new_lines.append(line)
        elif len(parts) == 5:
            # this is a box-only line with no polygon — remove it
            changed = True
            print(f"Removed box-only line in: {f}")
        else:
            new_lines.append(line)

    if changed:
        with open(path, 'w') as file:
            file.writelines(new_lines)
        fixed += 1

print(f"\nFixed {fixed} label files")