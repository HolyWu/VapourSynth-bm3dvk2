project('bm3dvk2', 'cpp',
    default_options: ['buildtype=release', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++20'],
    license: 'GPL-3.0-or-later',
    license_files: 'LICENSE',
    meson_version: '>=1.6.0',
    version: '1.0',
)

py = import('python').find_installation(pure: false)

incdir = include_directories(
    run_command(py, '-c', 'import vapoursynth as vs; print(vs.get_include())', check: true).stdout().strip(),
)

add_project_arguments('-D_CRT_SECURE_NO_WARNINGS', '-DVS_USE_API_43', language: 'cpp')

shared_module('bm3dvk2',
    files('src/bm3dvk.cpp'),
    dependencies: dependency('vulkan-headers'),
    gnu_symbol_visibility: 'hidden',
    include_directories: incdir,
    install: true,
    install_dir: py.get_install_dir() / 'vapoursynth/plugins',
    name_prefix: '',
)
