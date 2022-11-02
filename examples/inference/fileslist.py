import os
from pathlib import Path
dirpath = r"C:\Users\CharlesFettinger\Saved Games\MechWarrior Online\UI\MechIcons"
paths = sorted(Path(dirpath).iterdir(), key=os.path.getmtime, reverse=False)
for path in paths[:693]:
    print(path.name)