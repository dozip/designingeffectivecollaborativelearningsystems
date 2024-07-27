class Supply_Chain():
    """Class contains all important parameters of the supply chain on a global level
    """

    def __init__(self, adjacency_matrix, lead_time_matrix) -> None:
        self.ajaceny_matrix = adjacency_matrix
        self.lead_time_matrix = lead_time_matrix

    # SETTER
    def set_adjaceny_matrix(self, matrix):

        self.adjaceny_matrix = matrix

    def set_lead_time_matrix(self, matrix):

        self.lead_time_matrix = matrix

    # GETTER
    def get_adjaceny_matrix(self):

        return self.adjaceny_matrix

    def get_lead_time_matrix(self, matrix):

        return self.lead_time_matrix
