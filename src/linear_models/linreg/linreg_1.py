class MyLineReg():

    REPR_STR = '{class_name} class: n_iter={n_iter}, learning_rate={learning_rate}'

    def __init__(self, n_iter=100, learning_rate=0.1) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    
    def __repr__(self) -> str:
        return self.REPR_STR.format(
            class_name=self.__class__.__name__,
            n_iter=self.n_iter,
            learning_rate=self.learning_rate
        )
