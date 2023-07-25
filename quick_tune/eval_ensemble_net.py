import os
import torch
from ensemble_net import EnsembleNet
from metadataset import EnsembleMetaDataset
from trainer import train, test
import itertools
import numpy as np
from typing import List, Optional, Tuple
from scipy.stats import binom
import warnings
import time
import json
import argparse
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class TransferBO4E:

    def __init__(self, model, metadataset, model_path,
                                            acqf_name : str = "UCB",
                                            n_suggestions: int = 1,
                                            reload_model: bool = True,
                                            bo_iters: int = 200,
                                            candidates_limit: int = None,
                                            exploration_factor: float = 0.1,
                                            retrain: bool = False,
                                            dataset_split: str = "val",
                                            individuals_in_elite: int = 5,
                                            exploration_factor_for_elite: float = 0.5,
                                            max_ensemble_size: int = 10,
                                            num_epochs_retrain: int = 10,
                                            min_observed_pipelines_to_retrain: int = 10,
                                            support_size: int = 20, #size when conditioning
                                            subset_size: int = 10, # size when sampling
                                            max_num_observed_pipelines: int = 100,
                                            permute: bool = True,
                                            generator_iter = 2,
                                            verbose = False,
                                            device = "cuda"
                                            ):
        self.model = model.to(device)
        self.metadataset = metadataset
        self.acqf_name = acqf_name
        self.bo_iters = bo_iters
        self.model_path = model_path
        self.reload_model = reload_model
        self.n_suggestions = n_suggestions
        self.candidates_limit = candidates_limit
        self.neighrboorhood_ratio = 0.5
        self.num_epochs_retrain = num_epochs_retrain
        self.device = device
        self.min_observed_pipelines_to_retrain = min_observed_pipelines_to_retrain
        self.max_num_observed_pipelines = max_num_observed_pipelines
        self.subset_size = subset_size
        self.permute = permute
        self.generator_iter = generator_iter

        self.observed_pipeline_ids = []
        self.observed_ensembles = []
        self.observed_response_per_ensemble = []
        self.ensemble_elite = []
        self.ensemble_elite_response = []
        self.observed_time = []
        self.val_performance_curve = []
        self.test_performance_curve = []
        self.time_curve = []


        self.exploration_factor = exploration_factor
        self.exploration_factor_for_elite = exploration_factor_for_elite
        self.retrain = retrain
        self.dataset_split = dataset_split
        self.best_perf = -np.inf
        self.individuals_in_elite = individuals_in_elite
        self.max_ensemble_size = max_ensemble_size
        self.support_size = support_size
        self.state = "explore"
        self.start_time = None
        self.candidates_generator = self.candidates_generator_v1
        self.verbose = verbose



    def train (self, *args, **kwargs):
        return train(*args, **kwargs)

    def test (self, *args, **kwargs):
        return test(*args, **kwargs)

    def jaccard_distance(self,  set1: List[int],
                                set2: List[int]) -> float:
        intersection = set(set1).intersection(set(set2))
        union = set(set1).union(set(set2))
        return 1 - len(intersection)/len(union)


    def candidates_generator_v1(self, iter=2, subset_size=5, permute = True) -> List[List[int]]:

        if len(self.observed_pipeline_ids) > 2:
            for i in range(self.generator_iter):
                if len(self.observed_pipeline_ids) > self.subset_size:
                    subset_of_pipeline_ids = np.random.choice(self.observed_pipeline_ids, self.subset_size, replace=False)
                else:
                    subset_of_pipeline_ids = self.observed_pipeline_ids

                n = np.random.randint(1, min(len(subset_of_pipeline_ids), self.max_ensemble_size))

                if self.permute:
                    precandidates = list(itertools.permutations(subset_of_pipeline_ids, n))
                else:
                    precandidates = list(itertools.combinations(subset_of_pipeline_ids, n))

                #ix = np.arange(len(precandidates))
                np.random.shuffle(precandidates)

                if self.state == "greedy":
                    print("candidates generated: ", len(precandidates))
                    yield precandidates[:10000]
                else:
                    candidates = []
                    for precandidate in precandidates[:1000]:
                        #precandidate = list(precandidates[k])
                        precandidate = list(precandidate)
                        subset_of_single_candidates = np.arange(len(self.single_candidates))
                        subset_of_single_candidates = np.random.choice(subset_of_single_candidates, self.subset_size,
                                                                       replace=False)

                        for j in subset_of_single_candidates:
                            new_precandidate = precandidate + self.single_candidates[j]
                            #if new_precandidate not in self.observed_ensembles:
                            candidates.append(new_precandidate)
                    #if self.verbose:
                    print("candidates generated: ", len(candidates))
                    if len(candidates) > 0 :
                        yield candidates

        else:
            yield self.single_candidates



    def candidates_generator_v2(self, iter= 2 ,subset_size = 15, permute = True) -> List[List[int]]:
        #for now only one new candidate to the pipeline is considered
        #TODO: implement a more beatiful way to generate candidates
        for _ in range(iter):
            if len(self.observed_pipeline_ids) > 2:

                if len(self.observed_pipeline_ids) > subset_size:
                    subset_of_pipeline_ids = np.random.choice(self.observed_pipeline_ids, subset_size, replace=False)
                else:
                    subset_of_pipeline_ids = self.observed_pipeline_ids

                n = np.random.randint(1, min(self.max_ensemble_size, len(subset_of_pipeline_ids)))
                if permute:
                    precandidates = list(itertools.permutations(subset_of_pipeline_ids, n))
                else:
                    precandidates = list(itertools.combinations(subset_of_pipeline_ids, n))

                if not self.state:
                    return precandidates

                else:
                    candidates = []
                    for k in range(len(precandidates)):
                        precandidate = list(precandidates[k])
                        subset_of_single_candidates = np.arange(len(self.single_candidates))
                        subset_of_single_candidates = np.random.choice(subset_of_single_candidates, subset_size, replace=False)

                        for j in subset_of_single_candidates:
                            new_precandidate = precandidate + self.single_candidates[j]
                            candidates.append(new_precandidate)

                    if len(candidates)>0 and not self.state:
                        yield candidates

            for elite_member in self.ensemble_elite + [[]]:
                if len(elite_member)> 1:
                    candidates1 = []
                    candidates2 = []
                    n = np.random.randint(1, min(self.max_ensemble_size, len(elite_member)))

                    if permute:
                        precandidates = list(itertools.permutations(elite_member, n))
                    else:
                        precandidates = list(itertools.combinations(elite_member, n))

                    for k in range(len(precandidates)):
                        precandidate = list(precandidates[k])
                        if precandidate not in self.observed_ensembles:
                            candidates1.append(precandidate)

                        #assuming that single candidates are not in the list of precandidates
                        for single_candidate in self.single_candidates:
                            new_precandidate = precandidate + single_candidate
                            #if new_precandidate not in self.observed_ensembles:
                            candidates2.append(new_precandidate)

                    if len(candidates1) > 0 and self.state:# and len(candidates1[0]) > 1:
                        #dont yield single candidates that are know
                        print(len(candidates1))
                        yield candidates1

                    if len(candidates2) > 0 and not self.state:# and len(candidates2[0]) > 1
                        print(len(candidates2))
                        yield candidates2
                else:
                    candidates = []
                    for single_candidate in self.single_candidates:
                        precandidate = elite_member + single_candidate
                        #if precandidate not in self.observed_ensembles:
                        candidates.append(precandidate)
                    #   candidates += self.single_candidates
                    if len(candidates) > 0:
                        yield candidates


    def acqf(self, candidates: List[List[int]]) -> torch.FloatTensor:

        candidates = torch.LongTensor(candidates).to(self.device)
        x_query = self.hp_candidates[candidates]
        yp_query = self.observed_response_per_pipeline[candidates]

        x_support = []
        yp_support = []

        for i in range(1,self.model.num_encoders):

            if len(self.observed_pipeline_ids) == 0:
                x_support.append(x_query)
                yp_support.append(yp_query)
            else:
                idxs = np.random.choice(self.observed_pipeline_ids, self.support_size)
                #idxs = self.observed_pipeline_ids
                x_support.append(self.hp_candidates[idxs].unsqueeze(0).repeat(len(candidates),1,1))
                yp_support.append(self.observed_response_per_pipeline[idxs].unsqueeze(0).repeat(len(candidates),1))

        x = [x_query] + x_support
        yp = [yp_query] + yp_support
        mean, std = self.model.predict(x,yp)

        if self.acqf_name == "mean":
            acqf_values = -mean
        elif self.acqf_name == "UCB":
            acqf_values = -mean + self.exploration_factor*std
        else:
            raise NotImplementedError

        return acqf_values

    def get_unobserved_pipelines(self, ensemble: List[int]) -> Tuple[List[int], List[int], List[int]]:

        unobserved_pipelines_in_ensemble = []
        observed_pipelines_in_ensemble = []
        index_in_ensemble = []

        for i, pipeline_id in enumerate(ensemble):
            if pipeline_id not in self.observed_pipeline_ids:
                unobserved_pipelines_in_ensemble.append(pipeline_id)
                index_in_ensemble.append(i)
            else:
                observed_pipelines_in_ensemble.append(pipeline_id)

        return observed_pipelines_in_ensemble, unobserved_pipelines_in_ensemble,  index_in_ensemble


    def update_ensemble_elite(self, ensemble: List[int], y_ensemble: int):

        #so far only one ensemble is considered in the elite
        #TODO: consider more than one ensemble in the elite
        if self.individuals_in_elite > 0:
            if len(self.ensemble_elite) < self.individuals_in_elite:
                self.ensemble_elite.append(ensemble)
                self.ensemble_elite_response.append(y_ensemble)
            else:
                if y_ensemble > min(self.ensemble_elite_response):
                    index = np.argmin(self.ensemble_elite_response)
                    self.ensemble_elite[index] = ensemble
                    self.ensemble_elite_response[index] = y_ensemble

                for i in range(len(self.ensemble_elite)):
                    change = binom(1, self.exploration_factor_for_elite).rvs()
                    if change:
                        j = np.random.randint(len(self.observed_ensembles))
                        self.ensemble_elite[i] = self.observed_ensembles[j]

    def observe(self, ensemble: List[int]):

        total_time = 0
        ensemble = torch.LongTensor(ensemble)
        _, y_ensemble, y_per_pipeline = self.metadataset.get_y_ensemble(ensemble)

        observed_pipelines_in_ensemble, unobserved_pipelines_id, index_in_ensemble = self.get_unobserved_pipelines(ensemble.tolist())

        for i, pipeline_id in enumerate(unobserved_pipelines_id):
            #This is only useful if we propose many pipelines at the same time

            j = index_in_ensemble[i]
            self.observed_response_per_pipeline[pipeline_id] = y_per_pipeline[0][j].item()
            self.observed_pipeline_ids.append(pipeline_id)
            total_time += self.metadataset.time_info[["train_time", self.dataset_split+"_time"]].iloc[pipeline_id].sum()
            self.single_candidates.remove([pipeline_id])

        self.observed_ensembles.append(ensemble.tolist())
        self.observed_response_per_ensemble.append(y_ensemble.item())
        self.update_ensemble_elite(ensemble.tolist(), y_ensemble.item())

        if self.best_perf< y_ensemble.item():
            self.best_perf = y_ensemble.item()
            self.best_ensemble = ensemble.tolist()

        if len(unobserved_pipelines_id) > 0:
            self.val_performance_curve.append(self.get_normalized_regret())
            self.time_curve.append(time.time()-self.start_time)
            self.test_performance_curve.append(self.test_best_ensemble_on_dataset())

        if self.reload_model:
            self.model.load_checkpoint(self.model_path)

        if self.retrain and (len(self.observed_pipeline_ids) > self.min_observed_pipelines_to_retrain)\
                and len(unobserved_pipelines_id) > 0:
            self.train(self.model, self.metadataset, observed_pipeline_ids = self.observed_pipeline_ids,
                                                     num_epochs = self.num_epochs_retrain,
                                                     model_path = self.model_path.split(".")[0] + "_retrain.pt")

    def suggest(self)-> List[int]:
        """
        Suggest a new ensemble to be evaluated
        returns: ensemble
        """
        max_acq_value = -np.inf
        chosen_ensemble = [0]
        for candidates in self.candidates_generator():

            if self.state == "explore":
                acq_values = self.acqf(candidates)
                if torch.max(acq_values) > max_acq_value:
                    max_acq_value = torch.max(acq_values)
                    chosen_ensemble = candidates[torch.argmax(acq_values)]
            else:
                _, y_values, _ = self.metadataset.get_y_ensemble(candidates, batch_size=len(candidates))
                argmax_set = torch.where(y_values == torch.max(y_values))[0]
                ix = np.random.randint(argmax_set.shape).item()
                chosen_ensemble = candidates[ix]

        if len(self.observed_pipeline_ids) > 2 and self.state == "explore":
            self.state = "greedy"
        else:
            self.state = "explore"
        return chosen_ensemble

    def get_normalized_regret(self)-> float:

        best_perf = self.metadataset.get_best_performance()
        worst_perf = self.metadataset.get_worst_performance()
        normalized_regret = (best_perf - np.array(self.observed_response_per_ensemble).max())/(best_perf - worst_perf)
        return normalized_regret.item()

    def init_for_dataset(self, dataset_name, split = "val"):
        self.dataset_name = dataset_name
        self.metadataset.set_dataset(dataset_name, split = split)
        self.hp_candidates = torch.FloatTensor(self.metadataset.get_hp_candidates().values).to(self.device)
        self.num_candidates = len(self.hp_candidates)
        self.single_candidates = torch.arange(self.num_candidates).reshape(-1,1).tolist()


        #reset variables
        self.observed_response_per_pipeline = torch.zeros(self.num_candidates).to(self.device)
        self.best_ensemble = None
        self.best_perf = -np.inf
        self.observed_pipeline_ids = []
        self.observed_ensembles = []
        self.observed_response_per_ensemble = []
        self.ensemble_elite = []
        self.ensemble_elite_response = []
        self.observed_time = []
        self.val_performance_curve = []
        self.test_performance_curve = []
        self.time_curve = []

    def test_best_ensemble_on_dataset(self):
        self.metadataset.set_dataset(self.dataset_name, split = "test")
        _, y_ensemble, _ = self.metadataset.get_y_ensemble(self.best_ensemble)
        best_perf = self.metadataset.get_best_performance()
        worst_perf = self.metadataset.get_worst_performance()
        normalized_regret = (best_perf - y_ensemble.item())/(best_perf - worst_perf)
        self.metadataset.set_dataset(self.dataset_name, split = "val")

        return normalized_regret.item()

    def optimize(self):

        """
        Bayesian Optimization for ensemble selection

        """
        self.start_time = time.time()

        while (len(self.observed_pipeline_ids)<self.max_num_observed_pipelines):
            iteration_start_time = time.time()

            chosen_ensemble = self.suggest()
            self.observe(chosen_ensemble)
            normalized_regret = self.get_normalized_regret()

            if self.verbose:
                print("N. Regret: ", normalized_regret," Time: ", time.time() - iteration_start_time)
                print("N. pipelines: ", len(self.observed_pipeline_ids))
                print("N. ensembles: ", len(self.observed_ensembles))
        return self.val_performance_curve, self.test_performance_curve, self.time_curve, self.best_ensemble



if __name__ == "__main__":

    #read args
    parser = argparse.ArgumentParser(description='TBOE')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_dim', type=int, default=65)
    parser.add_argument('--max_num_pipelines', type=int, default=50, help="max number of pipelines in the ensemble during learning")
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--hidden_dim_ff', type=int, default=32)
    parser.add_argument('--num_encoders', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--val_freq', type=int, default=50)
    parser.add_argument('--train_model', type=int, default=1)
    parser.add_argument('--exploration_factor', type=float, default=1)
    parser.add_argument('--retrain', type=int, default=1)
    parser.add_argument('--experiment_id', type=str, default="TBOEtest")
    parser.add_argument('--max_ensemble_size', type=int, default=5)
    parser.add_argument('--max_num_observed_pipelines', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--criterion_type', type=str, default="weighted_listwise")
    parser.add_argument('--generator_iter', type=int, default=2)
    parser.add_argument('--num_epochs_retrain', type=int, default=10)
    parser.add_argument('--support_size', type=int, default=20)
    parser.add_argument('--subset_size', type=int, default=10)
    parser.add_argument('--data_version', type=str, default="micro")
    parser.add_argument('--reload_model', type=int, default=1)

    #parse arguments
    args = parser.parse_args()

    batch_size = args.batch_size
    input_dim = args.input_dim
    max_num_pipelines = args.max_num_pipelines
    hidden_dim = args.hidden_dim
    hidden_dim_ff = args.hidden_dim_ff
    num_encoders = args.num_encoders
    num_epochs = args.num_epochs
    val_freq = args.val_freq
    train_model = args.train_model
    exploration_factor = args.exploration_factor
    retrain = args.retrain
    experiment_id = args.experiment_id
    max_ensemble_size = args.max_ensemble_size
    max_num_observed_pipelines = args.max_num_observed_pipelines
    seed = args.seed
    verbose = args.verbose
    criterion_type = args.criterion_type
    generator_iter = args.generator_iter
    num_epochs_retrain = args.num_epochs_retrain
    support_size = args.support_size
    subset_size = args.subset_size
    data_version = args.data_version
    reload_model = args.reload_model

    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, "models", experiment_id + ".pt")

    print(args)

    split_ids = list(range(5))

    torch.manual_seed(seed)
    np.random.seed(seed)

    for split_id in split_ids:

        #check if save_path exists
        metadataset = EnsembleMetaDataset(batch_size = batch_size,
                                          data_version = data_version,
                                          max_num_pipelines = max_num_pipelines,
                                          split_id = split_id,
                                          data_path = os.path.join(current_dir, "..", "..", "AutoFinetune",
                                                                  "aft_data" , "predictions"),
                                          seed = seed)
        #x = loader.get_batch()

        model = EnsembleNet(dim_in = input_dim, hidden_dim = hidden_dim,
                                                     hidden_dim_ff = hidden_dim_ff,
                                                     num_encoders = num_encoders)

        if train_model:
            losses = train(model, metadataset, val_freq = val_freq,
                                            num_epochs = num_epochs,
                                            criterion_type = criterion_type,
                                            model_path = model_path)

        else:
            model.save_checkpoint(model_path)


        model.eval()
        tboe = TransferBO4E(model, metadataset, model_path, retrain=retrain,
                            max_ensemble_size=max_ensemble_size,
                            exploration_factor=exploration_factor,
                            max_num_observed_pipelines=max_num_observed_pipelines,
                            generator_iter=generator_iter,
                            num_epochs_retrain=num_epochs_retrain,
                            reload_model=reload_model,
                            verbose=verbose)
        #Possible important hps:
        # Sampling strategy
        # Exploration factor
        # subset_size, iter, permute
        # retrain, number of iterations to retrain
        for dataset in metadataset.split_datasets["test"]:
        #test BO
            #if load_model and os.path.exists(model_path):
            #    model.load_checkpoint(model_path)

            tboe.init_for_dataset(dataset)
            val_perf, test_perf, runtime, best_ensemble = tboe.optimize()

            results_dict = {"val_perf": val_perf,
                            "test_perf": test_perf,
                            "runtime": runtime,
                            "best_ensemble": best_ensemble,
                            }
            save_path = os.path.join(current_dir,"..", "results", experiment_id, dataset)
            os.makedirs(save_path, exist_ok=True)

            with open(os.path.join(save_path, "results.json"), "w") as f:
                json.dump(str(results_dict), f)


