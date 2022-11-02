import os
from pathlib import Path

# Your root directory path
rootdir = Path(r"C:\Users\CharlesFettinger\Saved Games\MechWarrior Online\UI\MechIcons")

#Your batch size
batch_size = 32

def walk_dirs(directory, batch_size):
    walk_dirs_generator = os.walk(directory)
    for dirname, subdirectories, filenames in walk_dirs_generator:
        for i in range(0, len(filenames), batch_size):
            # slice the filenames list 0-31, 32-64 and so on
            yield [os.path.join(dirname, filename) for filename in filenames[i:i+batch_size]]

# Finally iterate over the walk_dirs function which itself returns a generator
for file_name_batch in walk_dirs(rootdir, batch_size):
    for file_name in file_name_batch:
        # Do some processing on the batch now
        print (file_name)
        pass