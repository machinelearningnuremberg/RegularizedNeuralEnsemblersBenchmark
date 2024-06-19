import subprocess


def install_tabrepo():
    script_path = "SearchingOptimalEnsembles/metadatasets/tabrepo/install_tabrepo.sh"
    subprocess.run([script_path], check=True)


def main():
    install_tabrepo()


if __name__ == "__main__":
    main()
