# from typing import overload
# from hydra import compose, initialize
# from omegaconf import OmegaConf
# from Src.run import run_thread
# from threading import Thread
# import os


# def main():
#     # config = load_config()
#     run_experiment(None)

# def load_config(overrides=[]):
#     config = None #TODO: Can do overrrides
#     with initialize(config_path=".", job_name="job_name"):
#         config = compose(config_name="config", overrides=overrides)
#     return config

# def set_path(path):
#     os.chdir(path)
    

# def run_experiment(config):
#     seed = 0
#     threads = []
#     for i in range(2):
#         with initialize(config_path=".", job_name="job_name"):
#             config = compose(config_name="config")
#             t = Thread(target=run_thread, args=(config, seed + i * 2000))
#             path = f"outputs/test{i}"
#             set_path(path)
#             t.start()
#             threads.append(t)
#         # run_thread(config, seed + i * 2000)
#     # 1/0
#     print("Waiting")
#     for t in threads:
#         t.join()
#         print("OHHHHHHH")
#     print("Finished Wait")

# if __name__ == "__main__":
#     main()