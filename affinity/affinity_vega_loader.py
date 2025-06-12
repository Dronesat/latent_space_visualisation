import json
from bs4 import BeautifulSoup

class AffinityVegaLoadHTML:
    def __init__(self, html_file):
        self.html_file = html_file
        self.spec_dict = None
        self._load_and_parse()

    def _load_and_parse(self):
        # Read HTML file
        with open(self.html_file, "r", encoding="utf-8") as file:
            html = file.read()

        # Parse HTML and extract the script containing the JSON
        soup = BeautifulSoup(html, "html.parser")
        scripts = soup.find_all("script")

        # Find the script with "var spec ="
        spec_script = None
        for script in scripts:
            if script.string and "var spec =" in script.string:
                spec_script = script.string
                break

        # Extract JSON string from the script
        start_index = spec_script.find("var spec =") + len("var spec =")
        end_index = spec_script.find("var embedOpt")
        spec_json_str = spec_script[start_index:end_index].strip().rstrip(";")

        # Parse JSON string to dictionary
        self.spec_dict = json.loads(spec_json_str)

    def get_dataset(self, dataset_name):
        if not self.spec_dict:
            raise ValueError("spec_dict is not loaded")
        datasets = self.spec_dict.get("datasets", {})
        if dataset_name not in datasets:
            raise KeyError(f"Dataset '{dataset_name}' not found in spec")
        return datasets[dataset_name]
