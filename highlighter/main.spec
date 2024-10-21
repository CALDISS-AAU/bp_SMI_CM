# -*- mode: python ; coding: utf-8 -*-

block_cipher=None

from os.path import join

PATH_TO_SPIRE_LIB='' # path to spire/doc/lib dir
PATH_TO_REPO = '' # path to this repository

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=[(join(PATH_TO_SPIRE_LIB, 'Spire.Doc.Base.dll'), './spire/doc/lib'), (join(PATH_TO_SPIRE_LIB, 'libSkiaSharp.dll'), './spire/doc/lib')],
    datas=[(join(PATH_TO_REPO, 'highlighter', 'pipeline_functions.py'), '.'), (join(PATH_TO_REPO, 'modelling', 'models', 'rep_speech_model'), 'rep_speech_model'), (join(PATH_TO_REPO, 'highlighter', 'img', 'program_icon.png'), '.\\gooey\\images')],
    hiddenimports=[],
    hookspath=['join(PATH_TO_REPO, 'highlighter', 'hook-gooey.py:.'), 'join(PATH_TO_REPO, 'highlighter', 'hook-setfit.py:.')],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure,a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)