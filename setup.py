import setuptools

setuptools.setup(
    name="extension_dia",
    packages=setuptools.find_namespace_packages(),
    version="0.0.1",
    author="rsxdalv",
    description="DIA: A text-to-dialogue model",
    url="https://github.com/rsxdalv/extension_dia",
    project_urls={},
    scripts=[],
    install_requires=[
        "nari-tts @ git+https://github.com/rsxdalv/dia@stable",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
