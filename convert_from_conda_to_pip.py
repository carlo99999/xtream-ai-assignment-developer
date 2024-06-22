# read_conda_export.py
with open('conda_packages.txt', 'r') as conda_file:
    lines = conda_file.readlines()

with open('requirements.txt', 'w') as req_file:
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            continue
        package = line.split('=')[0]
        version = line.split('=')[1]
        req_file.write(f'{package}=={version}\n')
