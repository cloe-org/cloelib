import yaml

class Reader:

    def __init__(self, filepath: str):

        self.filepath = filepath
        self.data = self._load_yaml()

    def _load_yaml(self) -> dict:

        try:
            with open(self.filepath, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file {self.filepath} not found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error while parsing input file: {e}")

    def get_section(self, section: str) -> dict:

        keys = section.split(".")
        current_level = self.data
        for key in keys:
            if key not in current_level:
                raise KeyError(f"Section '{section}' not found in input file.")
            current_level = current_level[key]
        return current_level

    def get_full_config(self) -> dict:

        return self.data
