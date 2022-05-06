# version         Beta
# date            11/25/2021
# author          Alexander-Maurice Illig
# affilation      Institute of Biotechnology, RWTH Aachen
# email           a.illig@biotec.rwth-aachen.de

class InvalidVariantError(Exception):
    """
    Description
    -----------
    Exception raised when entered variant does not follow the required scheme
    (integer enclosed by two one letter code representations of amino acids).

    Attributes
    ----------
    variant: str
        Variant that causes the error
    message: str
        Explanation of the error
    """
    def __init__(self,variant: str):
        self.variant = variant
        message="The entered variant '%s' does not follow the required scheme (integer enclosed by two one letter code representations of amino acids). Check separator or variant."%(self.variant)
        self.message = message
        super().__init__(self.message)


class ActiveSiteError(Exception):
    """
    Description
    -----------
    Exception raised when requested position is not implemented in the DCA model.

    Attributes
    ----------
    position: int
        Position that causes the error
    variant: str
        Variant including that position
    message: str
        Explanation of the error
    """
    def __init__(self,position: int,variant: str):
        self.position = position
        self.variant = variant
        message="The position '%d' of variant '%s' is not an active site in the DCA model."%(self.position, self.variant)
        self.message = message
        super().__init__(self.message)