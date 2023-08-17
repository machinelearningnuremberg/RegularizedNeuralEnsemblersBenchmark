# type: ignore
# pylint: skip-file

import os

import numpy as np
import ujson

if __name__ == "__main__":
    path = "../AutoFinetune/experiments/output/model_predictions/"
    save_path = "../AutoFinetune/aft_data/predictions/"
    files = os.listdir(path)

    # full_dataset_name = "mini_set1_TEX_DTD_v1"
    # selected_files = [f for f in files if full_dataset_name in f]

    # file = '71610484905224209_mini_set1_TEX_DTD_v1'
    # file = '8306091001675516_micro_set2_FNG_v1'

    aggregated_info = {}
    # for file in files[15000:16000]:
    for file in files:
        try:
            if "micro" not in file:
                continue

            temp_path = os.path.join(path, file)
            dataset_name = "_".join(file.split("_")[1:])

            if dataset_name not in aggregated_info.keys():
                aggregated_info[dataset_name] = {"args": [], "time_info": []}

            test_predictions = np.load(os.path.join(temp_path, "test_predictions.npy"))
            test_targets = np.load(os.path.join(temp_path, "test_targets.npy"))
            val_predictions = np.load(os.path.join(temp_path, "val_predictions.npy"))
            val_targets = np.load(os.path.join(temp_path, "val_targets.npy"))

            # append columns of only ones
            test_indicator = np.ones(test_predictions.shape[0])[:, np.newaxis]
            val_indicator = np.zeros(val_predictions.shape[0])[:, np.newaxis]
            split_indicator = np.concatenate((test_indicator, val_indicator), axis=0)
            predictions = np.concatenate((test_predictions, val_predictions), axis=0)
            predictions = np.expand_dims(predictions, axis=0)
            targets = np.concatenate((test_targets, val_targets), axis=0)

            with open(os.path.join(temp_path, "args.json")) as f:
                args = ujson.load(f)

            train_time = args.pop("train_time")
            test_time = args.pop("test_time")
            val_time = args.pop("val_time")
            time_info = {
                "train_time": train_time,
                "test_time": test_time,
                "val_time": val_time,
            }

            if "predictions" not in aggregated_info[dataset_name].keys():
                aggregated_info[dataset_name]["predictions"] = predictions
                aggregated_info[dataset_name]["targets"] = targets
                aggregated_info[dataset_name]["split_indicator"] = split_indicator
            else:
                aggregated_info[dataset_name]["predictions"] = np.concatenate(
                    (aggregated_info[dataset_name]["predictions"], predictions), axis=0
                )

            aggregated_info[dataset_name]["args"].append(args)
            aggregated_info[dataset_name]["time_info"].append(time_info)
        except Exception as e:
            print(e)
            print("Error in file: ", file)
            continue
        # delete constant columns in dataframe

    aggregated_args = []
    for dataset_name in aggregated_info.keys():
        os.makedirs(os.path.join(save_path, "per_dataset", dataset_name), exist_ok=True)
        aggregated_info[dataset_name]["predictions"] = aggregated_info[dataset_name][
            "predictions"
        ].astype(np.float16)
        np.save(
            os.path.join(save_path, "per_dataset", dataset_name, "predictions.npy"),
            aggregated_info[dataset_name]["predictions"],
        )
        aggregated_info[dataset_name]["targets"] = aggregated_info[dataset_name][
            "targets"
        ].astype(np.int)
        np.save(
            os.path.join(save_path, "per_dataset", dataset_name, "targets.npy"),
            aggregated_info[dataset_name]["targets"],
        )
        aggregated_info[dataset_name]["split_indicator"] = (
            aggregated_info[dataset_name]["split_indicator"].reshape(-1).astype(int)
        )
        np.save(
            os.path.join(save_path, "per_dataset", dataset_name, "split_indicator.npy"),
            aggregated_info[dataset_name]["split_indicator"],
        )
        # aggregated_info[dataset_name]["targets"] = aggregated_info[dataset_name]["targets"].astype(np.float16).tolist()
        # aggregated_info[dataset_name]["split_indicator"] = aggregated_info[dataset_name]["split_indicator"].reshape(-1).astype(int).tolist()
        print(dataset_name, len(aggregated_info[dataset_name]["args"]))

        with open(
            os.path.join(save_path, "per_dataset", dataset_name, "args.json"), "w"
        ) as f:
            ujson.dump(aggregated_info[dataset_name]["args"], f)
        with open(
            os.path.join(save_path, "per_dataset", dataset_name, "time_info.json"), "w"
        ) as f:
            ujson.dump(aggregated_info[dataset_name]["time_info"], f)

        aggregated_args.extend(aggregated_info[dataset_name]["args"])
    # save json
    with open(os.path.join(save_path, "aggregated_args.json"), "w") as f:
        ujson.dump(aggregated_args, f)

    print("Done")
