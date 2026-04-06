import site, os

search_paths = site.getsitepackages() + [site.getusersitepackages()]

for p in search_paths:
    f = os.path.join(p, 'lightweight_gan', 'lightweight_gan.py')
    if os.path.exists(f):
        with open(f, 'r') as file:
            content = file.read()
        patched = content.replace(
            "assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'",
            "# GPU assertion removed for CPU inference"
        ).replace(
            'assert torch.cuda.is_available(), "You need to have an Nvidia GPU with CUDA installed."',
            '# GPU assertion removed for CPU inference'
        )
        with open(f, 'w') as file:
            file.write(patched)
        print('Patched:', f)
        break
else:
    print('Could not find lightweight_gan.py - searching manually...')
    import lightweight_gan
    f = lightweight_gan.__file__.replace('__init__.py', 'lightweight_gan.py')
    print('Found at:', f)
    with open(f, 'r') as file:
        content = file.read()
    patched = content.replace(
        "assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'",
        "# GPU assertion removed for CPU inference"
    )
    with open(f, 'w') as file:
        file.write(patched)
    print('Patched:', f)