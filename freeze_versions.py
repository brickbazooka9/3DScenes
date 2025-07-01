# freeze_versions.py
import pkg_resources

with open("requirements.txt", "w") as f:
    for dist in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower()):
        line = f"{dist.project_name}=={dist.version}"
        print(line)
        f.write(line + "\n")