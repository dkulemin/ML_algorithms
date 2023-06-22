class MyLineReg():
    def __init__(self, n_iter, learning_rate) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate

    
    def __repr__(self) -> str:
        representation_string = '{class_name} class: n_iter={n_iter}, learning_rate={learning_rate}'
        return representation_string.format(
            class_name=self.__class__.__name__,
            n_iter=self.n_iter,
            learning_rate=self.learning_rate
        )
