import matplotlib.pyplot as plt
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

pd.options.display.float_format = '{:,.2f}'.format

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

class Reporter:

    def __init__(self, 
                    report_id: int,
                    report_structure: dict,
                    experiments_results_path: str | None = None,
                    project_configs_path: str | None = None,
                    report_output_path: str | None = None,
                    config_filter: dict | None = None,
                    dataset_filter: list | None = None,
                    baseline: str | None = None,
                    split: str = "test"
            ):
        """
        Args:
        - report_structure: dictionary containing as first level keys the project names, 
                            as second level keys the experiment names.

                            Example:
                            { project_name1:
                                {
                                    table_name1_1: experiment_name1_1,
                                    table_name1_2: experiment_name1_2,
                                    ...
                                },
                              project_name2:
                                {
                                    ...
                                }
                            }}
        """
        
        self.report_id = report_id
        self.report_structure = report_structure
        self.config_filter = config_filter
        self.baseline = baseline
        self.split = split
        self.dataset_filter = [] if dataset_filter is None else dataset_filter

        current_path = Path(__file__).parent.absolute()

        if experiments_results_path is None:
            experiments_results_path = current_path / "experiments_results"

        if project_configs_path is None: 
            project_configs_path = current_path / "cluster_scripts" / "slurm_pineda" / "experiments_confs" 
        
        if report_output_path is None:
            report_output_path = current_path / "reports" 

        self.experiments_results_path = Path(experiments_results_path)
        self.project_configs_path = Path(project_configs_path)
        self.report_output_path = Path(report_output_path) / report_id

        self.report_output_path.mkdir(parents=True, exist_ok=True)
        self.experiment_names = self.report_structure["experiment_names"].keys()
        self.project_names = self.report_structure["project_names"].keys()

    def load_project_config(self, project_name):
        
        with open(self.project_configs_path / (project_name + ".yml"), 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def check_config(self, config):
        passed = True
        if self.config_filter is not None:
            for key, value in self.config_filter.items():
                if config.get(key, value) != value:
                    passed = False
        return passed

    def gather_results(self):
        projects_configs = []
        experiments_configs = []
        experiments_results = []
        missing_experiments_list = defaultdict(lambda: defaultdict(list))

        for project_name in self.report_structure["project_names"].keys():
            project_config = self.load_project_config(project_name)
            projects_configs.append(project_config)
            num_datasets = project_config["num_datasets"]

            for experiment_name in self.experiment_names:
                for dataset_id in range(num_datasets):
                    
                    if dataset_id in self.dataset_filter: continue

                    temp_path = self.experiments_results_path / project_name / experiment_name / str(dataset_id)

                    #read experiment config
                    try:
                        with open(temp_path / "args.json") as json_file:
                            experiment_config = json.load(json_file)
                        passed = self.check_config(experiment_config)
                    except:
                        missing_experiments_list[project_name][experiment_name].append(dataset_id)
                        passed = False
                        print("Args file not found for "+str(temp_path))

                    if passed: 
                        try:
                            with open(temp_path / "results.json") as json_file:
                                experiment_results = json.load(json_file)
                            experiment_results["dataset_id"] = dataset_id
                            experiment_results["experiment_name"] = experiment_name
                            experiment_results["project_name"] = project_name
                            experiments_configs.append(experiment_config)
                            experiments_results.append(experiment_results)
                        except:
                            print("Results file not found for "+ str(temp_path))
                            missing_experiments_list[project_name][experiment_name].append(dataset_id)

        return projects_configs, experiments_configs, experiments_results, missing_experiments_list
    
    def report_missing_experiments(self, missing_experiments_list):
        missing_experiments_list = default_to_regular(missing_experiments_list)
        missing_experiments_counts  = defaultdict(lambda: defaultdict(int))

        for project_name, experiments in missing_experiments_list.items():
            for experiment_name, missing_experiments in experiments.items():
                missing_experiments_counts[project_name][experiment_name] = len(missing_experiments)

        missing_experiments_counts = default_to_regular(missing_experiments_counts)

        with open(self.report_output_path / "missing_experiments_list.yml", 'w') as yaml_file:
            yaml.dump(missing_experiments_list, yaml_file)

        with open(self.report_output_path / "missing_experiments_counts.yml", 'w') as yaml_file:
            yaml.dump(missing_experiments_counts, yaml_file)        

 

    def process_table(self, table):
        table = pd.pivot_table(table, values=["val_metric", "test_metric"], index="experiment_name", columns=["project_name", "dataset_id"])
        if self.baseline is not None:
            for column in table.columns:
                table[column][np.isnan(table[column])] = table[column][self.baseline] 
                if table[column][self.baseline] != 0:
                    table[column] /= max(table[column][self.baseline], 1e-8)
        return table
    
    def report_ranked_metric(self, table):

        if self.split == "test":
            table = table[["test_metric"]].rank().groupby("project_name", axis=1).mean()
            table.columns = pd.MultiIndex.from_product([["Test"], table.columns])
        else:
            table = table["val_metric"].rank().groupby("project_name", axis=1).mean()
            table.columns = pd.MultiIndex.from_product([["Validation"], table.columns])    
        
        #return pd.concat([test_table, val_table], axis=1)
        return table
    
    def report_aggregated_metric(self, table):
        if self.split == "test":
            table = table[["test_metric"]].groupby("project_name", axis=1).mean()
            table.columns = pd.MultiIndex.from_product([["Test"], table.columns])
        else:
            table = table["val_metric"].groupby("project_name", axis=1).mean()
            table.columns = pd.MultiIndex.from_product([["Validation"], table.columns])    
        
        #return pd.concat([test_table, val_table], axis=1)
        return table
    
    def report_time(self, table):
        return table
    
    def report_dataset_size(self, table):
        return table

    def rename_projects(self, table):
        new_columns=[]
        for i, column in enumerate(table.columns.levels[1]):
            new_columns.append(self.report_structure["project_names"][column])
        table.columns = table.columns.set_levels(new_columns, level=1)
        return table

    def rename_experiments(self, table):
        new_index=[]
        for i, index in enumerate(table.index):
            new_index.append(self.report_structure["experiment_names"][index])    
        table.index = new_index
        return table

    def write_tables(self, tables):

    
        def highlight_smallest(s, smallest, second_smallest):
            # Find the smallest and second smallest values
            #smallest = s.min()
            #second_smallest = s.nsmallest(2).iloc[-1]

            # Apply formatting
            #return ['\\textbf{' + str(v) + '}' if v == smallest else ('\\textit{' + str(v) + '}' if v == second_smallest else str(v)) for v in s]
            return '\\textbf{' + str(s) + '}' if s == smallest else ('\\underline{\\textit{' + str(s) + '}}' if s == second_smallest else str(s))
                                                                     
        for table_name, table in tables.items():
            table = table.loc[self.experiment_names]
            table = self.rename_projects(table)
            table = self.rename_experiments(table)
            table.columns.names=["",""]
            smallest_ones = table.apply(lambda x: x.min()).apply(lambda x: '{:,.4f}'.format(x))
            second_smallest_ones = table.apply(lambda x: x.nsmallest(2).iloc[-1]).apply(lambda x: '{:,.4f}'.format(x))

            for column in table.columns:
                table[column] = table[column].apply(lambda x: '{:,.4f}'.format(x))
                smallest = smallest_ones[column]
                second_smallest = second_smallest_ones[column]
                table[column] = table[column].apply(lambda x: highlight_smallest(x, smallest, second_smallest))

            table.to_csv(self.report_output_path / (table_name+".csv"))
            #styled_df = table.apply(highlight_smallest, axis=0)
            #s = styled_df.style.highlight_max(
            #   # props='cellcolor:[HTML]{FFFF00}; color:{red}; itshape:; bfseries:;'
            #)
            #styled_df = styled_df.style.format("{:.2f}".format)
            table = table.apply(lambda x: x.replace("{r}{Test}", "{c}{\textbf{Test}}"))
            table = table.apply(lambda x: x.replace("{r}{Test}", "{c}{\textbf{Test}}"))

            # styler = table.style.set_table_styles([
            #     {'selector': 'toprule', 'props': ':hline;'},
            #     {'selector': 'midrule', 'props': ':hline;'},
            #     {'selector': 'bottomrule', 'props': ':hline;'},
            #     ], overwrite=False)

            styler = table.style
            styler.applymap_index(lambda v: "font-weight: bold;", axis="index")
            styler.applymap_index(lambda v: "font-weight: bold;", axis="columns")
            latex_table =styler.to_latex(convert_css=True, hrules=True)
            #styled_df = styled_df.style.format(precision=2, subset=[1])
            #latex_table = table.to_latex(
            #                                 column_format="ccccc",
            #                                # hrules=True
            #                                 )
            with open(self.report_output_path / (table_name+".tex"), 'w') as f:
                f.write(latex_table)

    def report(self):

        projects_configs,\
            experiments_configs,\
            experiments_results,\
            missing_experiments_list =  self.gather_results()

        table = pd.DataFrame(experiments_results)
        table.to_csv(self.report_output_path / "complete_results.csv")

        experiment_configs_table = pd.DataFrame(experiments_configs)

        self.report_missing_experiments(missing_experiments_list)
        processed_table = self.process_table(table)
        aggregated_table = self.report_aggregated_metric(processed_table)
        ranked_table = self.report_ranked_metric(processed_table)
        self.write_tables({"aggregated": aggregated_table, 
                            "ranked": ranked_table})

        self.report_time(table)
        self.report_dataset_size(table)

    @classmethod
    def load_reporter(cls, reporter_config_file):
        with open(reporter_config_file) as file:
            config = yaml.safe_load(file)
        
        return Reporter(**config)

       

if __name__ == "__main__":

    reporter_config_file = Path(__file__).parent.absolute() / "reporter_configs/report5.yml"
    reporter =  Reporter.load_reporter(reporter_config_file)
    reporter.report()