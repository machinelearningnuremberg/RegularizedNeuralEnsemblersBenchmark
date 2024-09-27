import matplotlib.pyplot as plt
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('axes', axisbelow=True)

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
                    per_dataset: bool = False,
                    plot_by_groups: bool = False,
                    plot_as_curves: bool = False,
                    include_legend: bool = True,
                    title: str = "",
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
        self.per_dataset = per_dataset
        self.dataset_filter = [] if dataset_filter is None else dataset_filter
        self.plot_by_groups = plot_by_groups
        self.plot_as_curves = plot_as_curves
        self.title = title
        self.include_legend = include_legend

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
        self.project_names = self.report_structure["project_names"].keys()

        if self.plot_by_groups:
            self.experiment_names = []
            self.experiment_table_names = []
            self.groups = []
            for group in self.report_structure["experiment_groups"].keys():
                self.experiment_names.extend(list(self.report_structure["experiment_groups"][group].keys()))
                self.experiment_table_names.extend(list(self.report_structure["experiment_groups"][group].values()))
                self.groups.extend([group]*len(self.report_structure["experiment_groups"][group].keys()))
            self.translator_to_table_names = dict(zip(self.experiment_names, self.experiment_table_names))
        else:
            self.experiment_names = self.report_structure["experiment_names"].keys()

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
            num_seeds = project_config["num_seeds"]

            for experiment_name in self.experiment_names:
                for dataset_id in range(num_datasets):
                    
                    if dataset_id in self.dataset_filter: continue

                    for seed in range(num_seeds):
                        temp_path = self.experiments_results_path / project_name / experiment_name / str(dataset_id) / str(seed)

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
                                experiment_results["seed"] = seed
                                experiments_configs.append(experiment_config)
                                experiments_results.append(experiment_results)
                            except:
                                print("Results file not found for "+ str(temp_path))
                                missing_experiments_list[project_name][experiment_name].append(dataset_id)

        return projects_configs, experiments_configs, experiments_results, missing_experiments_list
    
    def write_missing_experiments(self, missing_experiments_list):
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
        table = pd.pivot_table(table, values=["val_metric", "test_metric"], index="experiment_name", columns=["project_name", "dataset_id", "seed"])
        
        if self.split == "test": 
            table = table["test_metric"]
        else:
            table = table["val_metric"]

        #average across runs
        table = table.groupby(["dataset_id","project_name"],axis=1).mean()

        if self.baseline is not None:
            for column in table.columns:
                table[column][np.isnan(table[column])] = table[column][self.baseline] 
                #if (table[column][self.baseline] != 0) and (not self.per_dataset):
                if not self.per_dataset:
                    table[column][table[column]==0] = 1e-08
                    table[column] /= table[column][self.baseline]
                    #table[column] = max(table[column], 1e-05)/ max(table[column][self.baseline], 1e-5)
                    table[column][table[column]>10] = 10 #imputing the very large errors
        table = table.loc[self.experiment_names]
         
        return table
    
    def report_ranked_metric(self, table):
        if self.per_dataset:
            mean = table.rank().groupby(["project_name", "dataset_id"], axis=1).mean()
            std = table.rank().groupby(["project_name", "dataset_id"], axis=1).std()       
            #mean.columns = mean.columns.droplevel("project_name")
            #std.columns = std.columns.droplevel("project_name")
        else:
            mean = table.rank().groupby(["project_name"], axis=1).mean()
            std = table.rank().groupby(["project_name"], axis=1).std()
        return mean, std
    
    def report_aggregated_metric(self, table):
        if self.per_dataset:
            mean = table.groupby(["project_name", "dataset_id"], axis=1).mean()
            std = table.groupby(["project_name", "dataset_id"], axis=1).std()
            #mean.columns = mean.columns.droplevel("project_name")
            #std.columns = std.columns.droplevel("project_name")
        else:
            mean = table.groupby("project_name", axis=1).mean()
            std = table.groupby("project_name", axis=1).std()            
        return mean, std
    
    def report_time(self, table):
        return table
    
    def report_dataset_size(self, table):
        return table

    def rename_projects(self, table, suffix=""):
        new_columns=[]
        for i, column in enumerate(table.columns):
            new_columns.append(self.report_structure["project_names"][column]+suffix)
        table.columns = new_columns
        return table

    def rename_experiments(self, table):
        new_index=[]
        for i, index in enumerate(table.index):
            new_index.append(self.report_structure["experiment_names"][index])    
        table.index = new_index
        return table

    def write_tables(self, tables):
        def highlight_smallest(m, s, smallest, second_smallest):
            return '\\textbf{' + str(m) + '}$_{\pm'+ str(s)+'}$' if m == smallest \
                        else ('\\underline{\\textit{' + str(m) + '}}$_{\pm'+ str(s)+'}$' if m == second_smallest else str(m)+'$_{\pm'+ str(s)+'}$')
                                                                     
        for table_name, table_group in tables.items():
            table_mean = table_group["mean"]
            table_std = table_group["std"]

            if not self.per_dataset:
                table_mean = self.rename_projects(table_mean)
                table_std = self.rename_projects(table_std, suffix="_std")
            else:
                new_column_names = [str(y)+"-"+x for (x,y) in table_mean.columns]
                table_mean.columns.droplevel(0)
                table_mean.columns.droplevel(0)
                table_mean.columns = new_column_names
                table_std.columns =  new_column_names
                table_mean = table_mean[np.sort(new_column_names).tolist()]
                table_std = table_std[np.sort(new_column_names).tolist()]
                table_std.columns = [str(x)+"_std" for x in table_std.columns]
            table_mean = self.rename_experiments(table_mean)
            table_std = self.rename_experiments(table_std)

            #table.columns.names=["",""]
            smallest_ones = table_mean.apply(lambda x: x.min()).apply(lambda x: '{:,.4f}'.format(x))
            second_smallest_ones = table_mean.apply(lambda x: x.nsmallest(2).iloc[-1]).apply(lambda x: '{:,.4f}'.format(x))

            table = pd.concat([table_mean, table_std], axis=1)

            for column in table_mean.columns:
                table[column] = table[column].apply(lambda x: '{:,.4f}'.format(x))
                table[column+"_std"] = table[column+"_std"].apply(lambda x: '{:,.4f}'.format(x))

                smallest = smallest_ones[column]
                second_smallest = second_smallest_ones[column]
                #table[column] = table[column].apply(lambda x: highlight_smallest(x, smallest, second_smallest))
                #table[column+"_std"] = table[column+"_std"].apply(lambda x: highlight_smallest(x, smallest, second_smallest))
                #table[column] = table[[column, column+"_std"]].apply(lambda x: x[column]+" $\pm$ "+x[column+"_std"], axis=1)
                if not self.per_dataset:
                    table[column] = table[[column, column+"_std"]].apply(lambda x: highlight_smallest(x[column],x[column+"_std"], smallest, second_smallest), axis=1)
            
            table = table[table_mean.columns]
            table.to_csv(self.report_output_path / (table_name+".csv"))

            #table = table.apply(lambda x: x.replace("{r}{Test}", "{c}{\textbf{Test}}"))
            #table = table.apply(lambda x: x.replace("{r}{Test}", "{c}{\textbf{Test}}"))

            styler = table.style
            styler.applymap_index(lambda v: "font-weight: bold;", axis="index")
            styler.applymap_index(lambda v: "font-weight: bold;", axis="columns")
            latex_table =styler.to_latex(convert_css=True, hrules=True)

            with open(self.report_output_path / (table_name+".tex"), 'w') as f:
                f.write(latex_table)

    def make_plot_by_groups(self, df, fontsize=30):
        for project in self.report_structure["project_names"]:

            # Use a colormap (e.g., viridis) to generate unique colors for each method class
            cmap = plt.get_cmap('viridis')
            groups = np.unique(self.groups)
            colors = cmap(np.linspace(0, 1, len(groups)))

            # Create a color map dictionary
            color_map = {group: color for group, color in zip(groups, colors)}

            # Assign colors to the DataFrame based on the method class
            df['color'] = df['group'].map(color_map)
            df = df.iloc[df[project].argsort()]
            df["name"] = [self.translator_to_table_names[x] for x in df.index]
            # Plot the bar chart
            plt.figure(figsize=(10, 10))
            bars = plt.bar(df["name"], df[project], color=df['color'], width=0.9, zorder=1)


            for bar in bars:
                yval = bar.get_height()  # Get the height of each bar (i.e., metric value)
                plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 3),
                          ha='center', va='bottom', fontsize=fontsize*0.5)

            # Add labels and title
            plt.xlabel('Method', fontsize=fontsize*1.5)
            plt.ylabel('Normalized NLL', fontsize=fontsize*1.5)
            # plt.title('Metric by Method with Different Colors for Method Classes')

            # # Rotate x-axis labels for better readability
            plt.xticks(rotation=45, ha='right', fontsize=fontsize)
            plt.yticks(fontsize=fontsize)  # Adjust font size of y-ticks
            # # Show the plot
            
            plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, zorder=0)  # Grid behind bars with zorder=0
            
            legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[group]) for group in groups]
            
            plt.legend(legend_handles, groups, title="Method Type", fontsize=fontsize, title_fontsize=fontsize)

            plt.tight_layout()
            # plt.show()
            plt.savefig(self.report_output_path / (f"{project}_plot.png"))
            plt.savefig(self.report_output_path / (f"{project}_plot.pdf"))

    def make_plot_as_curves(self, table, fontsize=30, linewidth=5):
        
        plt.figure(figsize=(10, 10))
        plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, zorder=-1)  # Grid behind bars with zorder=0
        plt.grid(True, which='both', axis='x', linestyle='--', linewidth=0.7, zorder=-1)  # Grid behind bars with zorder=0

        for project_name in table.columns:
            plt.plot(table[project_name], label=project_name, linewidth=linewidth)

        plt.axhline(y=1., color='black', linestyle='--', linewidth=linewidth*0.5)  
        plt.xlabel('DropOut Rate', fontsize=fontsize*1.5)
        plt.ylabel('Normalized NLL', fontsize=fontsize*1.5)
        # plt.title('Metric by Method with Different Colors for Method Classes')

        # # Rotate x-axis labels for better readability
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)  # Adjust font size of y-ticks
        #
        plt.title(self.title, fontsize=fontsize*1.5)
        
        if self.include_legend:
            #plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=fontsize, columnspacing=1)
        #plt.legend(fontsize=fontsize)
        plt.tight_layout()
        
        table.to_csv(self.report_output_path / (f"curves_table.csv"))
        plt.savefig(self.report_output_path / (f"curves_plot.png"))
        plt.savefig(self.report_output_path / (f"curves_plot.pdf"))


    def report(self):
        projects_configs,\
            experiments_configs,\
            experiments_results,\
            missing_experiments_list =  self.gather_results()
        table = pd.DataFrame(experiments_results)
        table.to_csv(self.report_output_path / "complete_results.csv")

        experiment_configs_table = pd.DataFrame(experiments_configs)

        processed_table = self.process_table(table)
        aggregated_mean, aggregated_std = self.report_aggregated_metric(processed_table)
        ranked_mean, ranked_std = self.report_ranked_metric(processed_table)
        self.write_missing_experiments(missing_experiments_list)
 
        if self.plot_by_groups:
            aggregated_mean["group"] = self.groups
            self.make_plot_by_groups(aggregated_mean)

        else:
            self.write_tables({"aggregated":
                                    { "mean": aggregated_mean,
                                    "std": aggregated_std 
                                    },
                                "ranked":
                                {
                                    "mean": ranked_mean,
                                    "std": ranked_std,
                                    }
                                }                   
                            )

            self.report_time(table)
            self.report_dataset_size(table)

            if self.plot_as_curves:
                self.make_plot_as_curves(aggregated_mean)

    def make_double_plot_as_curves(self, table_paths, titles, fontsize=24, linewidth=5):
        fig, axs = plt.subplots(1, 2, figsize=(24, 8),gridspec_kw={'wspace': 0.3})
        for i, table_path in enumerate(table_paths):
            table = pd.read_csv(table_path)
            axs[i].grid(True, which='both', axis='y', linestyle='--', linewidth=0.7, zorder=-1)  # Grid behind bars with zorder=0
            axs[i].grid(True, which='both', axis='x', linestyle='--', linewidth=0.7, zorder=-1)  # Grid behind bars with zorder=0
            values = table.iloc[:,0].values

            for project_name in table.columns[1:]:
                axs[i].plot(values, table[project_name], label=project_name, linewidth=linewidth)

            axs[i].axhline(y=1., color='black', linestyle='--', linewidth=linewidth*0.5)  
            axs[i].set_xlabel('DropOut Rate', fontsize=fontsize*1.5)

            if i == 0:
                axs[i].set_ylabel('Normalized NLL', fontsize=fontsize*1.5)
            # plt.title('Metric by Method with Different Colors for Method Classes')

            # # Rotate x-axis labels for better readability
            axs[i].set_xticks(values)
            axs[i].set_xticklabels(values, fontsize=fontsize)
            
            axs[i].set_yticks(axs[i].get_yticks())
            axs[i].set_yticklabels(axs[i].get_yticklabels(), fontsize=fontsize)
            
            axs[i].set_title(titles[i], fontsize=fontsize*1.5)
            
        fig.legend(*axs[0].get_legend_handles_labels(), loc='upper center', bbox_to_anchor=(0.5, 0.0),ncol=7, fontsize=fontsize, columnspacing=1)
        #plt.tight_layout()
        
        fig.savefig(self.report_output_path / (f"double_curves_plot.png"), bbox_inches="tight")
        fig.savefig(self.report_output_path / (f"double_curves_plot.pdf"), bbox_inches="tight")


    @classmethod
    def load_reporter(cls, reporter_config_file):
        with open(reporter_config_file) as file:
            config = yaml.safe_load(file)
        
        return Reporter(**config)

    
if __name__ == "__main__":

    report_id = "report18"
    reporter_configs_path = Path(__file__).parent.absolute() / "reporter_configs"
    reporter_output_path =  Path(__file__).parent.absolute() / "reports"
    reporter_config_file = reporter_configs_path / (report_id+".yml")
    reporter =  Reporter.load_reporter(reporter_config_file)
    reporter.report()
    reporter.make_double_plot_as_curves(table_paths=[reporter_output_path / "report18"/ "curves_table.csv",
                                                     reporter_output_path / "report19"/ "curves_table.csv"],
                                        titles=["Stacking", "Model Average"] )