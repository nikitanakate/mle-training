import subprocess

def check_and_install(package):
    try:
        __import__(package)
        print(f"{package} already installed.")
    except ImportError:
        install = input(f"{package} not found. Do you want to install it? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call(['pip', 'install', package])
                print(f"{package} installed successfully!")
            except Exception as e:
                print(f"An error occurred while installing {package}: {e}")
        else:
            print(f"{package} not installed.")

if __name__ == "__main__":
    packages_to_check = ['numpy', 'pandas', 'scipy', 'scikit-learn', 'argparse', 'logging', 'configparser']

    for package in packages_to_check:
        check_and_install(package)


