from PyInstaller.utils.hooks import collect_submodules, collect_data_files
datas = collect_data_files('setfit')
hiddenimports = collect_submodules('setfit')
