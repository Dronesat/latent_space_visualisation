import json
from bs4 import BeautifulSoup

# read HTML file
with open("plt_latent_embed_epoch_30_tsne.html", "r", encoding="utf-8") as file:
    html = file.read()

# parse HTML and extract the script containing json
soup = BeautifulSoup(html, "html.parser")
scripts = soup.find_all("script")

# loop script tags to find "var spec ="
spec_script = None
for script in scripts:
    if script.string and "var spec =" in script.string:
        spec_script = script.string
        break

# extract JSON string part of "var spec = {...};"
start_index = spec_script.find("var spec =") + len("var spec =")
end_index = spec_script.find("var embedOpt")
spec_json_str = spec_script[start_index:end_index].strip().rstrip(";")

# parse into dictionary
spec_dict = json.loads(spec_json_str)

# get the first dataset
datasets = spec_dict["datasets"]
first_dataset_key = next(iter(datasets))
dataset = datasets[first_dataset_key]

for entry in dataset:
    entry_str = entry["id"], entry["emb-x"], entry["emb-y"]
    print(entry['emb-x'])

with open('file.txt', 'w') as f:
    json.dump(dataset, f)