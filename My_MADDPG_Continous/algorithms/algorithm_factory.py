# algorithms/algorithm_factory.py
class AlgorithmFactory:
    @staticmethod
    def create_algorithm(algo_type, *args, **kwargs):
        if algo_type == "maddpg":
            from agents.maddpg.MADDPG_agent import MADDPG
            return MADDPG(*args, **kwargs)
        elif algo_type == "independent":
            from agents.independent.IndependentRL import IndependentRL
            return IndependentRL(*args, **kwargs)
        elif algo_type == "centralized":
            from agents.centralized.CentralizedRL import CentralizedRL
            return CentralizedRL(*args, **kwargs)

# main_train.py
algo_type = "maddpg"  # 或 "independent" 或 "centralized"
agent = AlgorithmFactory.create_algorithm(algo_type, *args)