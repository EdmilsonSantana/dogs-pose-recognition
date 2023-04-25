import os
import sys

origin = sys.argv[1]
destination = sys.argv[2]

for i, filename in enumerate(os.listdir(origin)):
    _, file_extension = os.path.splitext(filename)
    os.rename(f'{origin}/{filename}', f'{destination}/{str(i)}{file_extension}')
