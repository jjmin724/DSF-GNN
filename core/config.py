from core.utils import load_json, set_seed

cfg = load_json("./config/config.json")
set_seed(cfg["random_seed"])
