import subprocess


def install_phem():
    script_path = "SearchingOptimalEnsembles/posthoc/phem/install_phem.sh"
    subprocess.run([script_path], check=True)


def main():
    install_phem()


if __name__ == "__main__":
    main()
