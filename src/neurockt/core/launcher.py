import yaml
from importlib import import_module
from logging import getLogger


logger = getLogger(__name__)


class Run:
    def __init__(self, config, wandb, resume, run_id=None, replacer=None):
        self.wandb = wandb
        self.run_id = run_id
        with open(f"{config}.yaml", "r") as file:
            def return_if_exists(key):
                return configs[key] if key in configs else {}
            configs = yaml.safe_load(file)
            self.action = configs["action"]
            self.args = configs["args"]
            self.units = return_if_exists("units")
            self.modules = return_if_exists("modules")
            self.project = configs["project"]
            self.artifacts = configs["artifacts"]

        # specify the name of saved model
        self.args["resume"] = self.artifacts["model"] if resume else None

        # add default tags
        self.project["tags"].extend(config.split("/"))

        # replace ?
        if type(replacer) == str:
            for _, unit_dict in self.units.items():
                for k, v in unit_dict.items():
                    unit_dict[k] = v.replace("?", replacer)

        import copy
        self.configs = copy.deepcopy(configs)

    def callback(self):
        if self.args["resume"] is not None:
            self.project["id"] = self.run_id
            self.project["resume"] = "must"
        if self.wandb:
            import wandb
            wandb.init(**self.project)
            if self.args["resume"] is None:
                wandb.config.update({"complement": self.configs})
            from fastai.callback.wandb import WandbCallback
            return WandbCallback(log_model=True,
                                 model_name=self.artifacts["model"])
        return None

    def import_units(self):
        units = {}
        from units.units import import_unit
        for unit_type, unit_dict in self.units.items():
            for k, v in unit_dict.items():
                units[k] = import_unit(unit_type, v)

        self.units = units

    def prepare_action(self):
        self.import_modules()
        self.import_units()
        self.import_dataset()
        logger.info(f"Action -> actions.{self.action}")
        self.action = import_module(f"actions.{self.action}")

    def run_action(self):
        self.prepare_action()
        return self.action.main(cb_func=self.callback, units=self.units, **self.args)

    def init_action(self):
        self.prepare_action()
        return self.action.initialize(cb_func=self.callback, units=self.units, **self.args)

    def import_dataset(self):
        if "dataset" in self.args:
            dataset = self.args["dataset"]
            self.args["dataset"] = import_module(f"datasets.{dataset}").load()

    def import_modules(self):
        for k, v in self.modules.items():
            self.modules[k] = import_module(v)

        for k, v in self.args.items():
            if type(v) is not str:
                continue
            v = v.split('.', 1)
            if len(v) == 2:
                module, name = v
                self.args[k] = self.modules[module].__dict__[name]


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    import argparse
    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('config', type=str, help="configsディレクトリの設定ファイルを指定する")
    parser.add_argument("--id", type=str, help="ユニット名称に?が存在した場合に置換を行う")
    parser.add_argument('--wandb', action='store_true', help="W&Bにログを記録する")
    parser.add_argument('--resume', action='store_true', help="アクションを再開する")
    parser.add_argument('--run', type=str, help="実験IDを指定する")
    parser.add_argument('-i', action='store_true', help="アクションの初期化のみを行う")
    args = parser.parse_args()

    config = f"./configs/{args.config}"
    run = Run(config=config,
              wandb=args.wandb,
              resume=args.resume,
              run_id=args.run,
              replacer=args.id)
    if not args.i:
        run.run_action()
    else:
        run.init_action()
