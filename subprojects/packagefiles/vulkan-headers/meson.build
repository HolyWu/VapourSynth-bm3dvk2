project('vulkan-headers')

vulkan_headers_dep = declare_dependency(include_directories: include_directories('include', is_system: true))

meson.override_dependency('vulkan-headers', vulkan_headers_dep)
