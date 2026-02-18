class BaseExtractor:
    def compute(self, rr_sequence):
        raise NotImplementedError("Subclasses must implement the compute method.")