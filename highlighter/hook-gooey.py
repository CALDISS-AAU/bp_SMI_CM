from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect all data files and submodules associated with Gooey
datas = collect_data_files('gooey')
hiddenimports = collect_submodules('gooey')