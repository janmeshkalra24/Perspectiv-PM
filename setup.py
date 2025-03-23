from setuptools import find_packages, setup

setup(
    name="screen_understanding",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "httpx>=0.24.0",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "redis>=5.0.0",
        "tenacity>=8.0.0",
        "rich>=13.0.0",
        "pyautogui>=0.9.54",
    ],
    python_requires=">=3.7",
) 