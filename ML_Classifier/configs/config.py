from pathlib import Path

base_path = Path(__file__).parent
file_path = (base_path / "../data/data.csv").resolve()

feature_pickle=(base_path/'..//models//features.pickle').resolve()
enc_pickle=(base_path/'..//models//encoder.pickle').resolve()
mod_pickle=(base_path/'..//models//model.pickle').resolve()