class Pipeline:
    """
    A minimal pipeline for chaining transformers and a final estimator.

    Each intermediate step must implement:
    - fit(X, y=None)
    - transform(X)

    The final step may implement either:
    - fit(X, y=None) and transform(X)
    or
    - fit(X, y=None) and predict(X)
    """

    def __init__(self, steps):
        if not isinstance(steps, list) or len(steps) == 0:
            raise ValueError("steps must be a non-empty list of (name, step) tuples")

        self.steps = steps
        self._validate_steps()

    def _validate_steps(self):
        for item in self.steps:
            if not isinstance(item, tuple) or len(item) != 2:
                raise ValueError("each pipeline step must be a (name, step) tuple")
            name, step = item
            if not isinstance(name, str) or not name:
                raise ValueError("step name must be a non-empty string")
            if step is None:
                raise ValueError("step object cannot be None")

    @property
    def named_steps(self):
        return dict(self.steps)

    def fit(self, X, y=None):
        Xt = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"Intermediate step '{name}' must implement fit and transform"
                )
            if y is not None:
                try:
                    step.fit(Xt, y)
                except TypeError:
                    step.fit(Xt)
            else:
                step.fit(Xt)
            Xt = step.transform(Xt)

        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "fit"):
            raise TypeError(f"Final step '{final_name}' must implement fit")

        if y is not None:
            try:
                final_step.fit(Xt, y)
            except TypeError:
                final_step.fit(Xt)
        else:
            final_step.fit(Xt)
        return self

    def transform(self, X):
        Xt = X
        for name, step in self.steps:
            if not hasattr(step, "transform"):
                raise TypeError(f"Step '{name}' does not implement transform")
            Xt = step.transform(Xt)
        return Xt

    def fit_transform(self, X, y=None):
        Xt = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"Intermediate step '{name}' must implement fit and transform"
                )
            if y is not None:
                try:
                    step.fit(Xt, y)
                except TypeError:
                    step.fit(Xt)
            else:
                step.fit(Xt)
            Xt = step.transform(Xt)

        final_name, final_step = self.steps[-1]

        if hasattr(final_step, "fit_transform"):
            if y is not None:
                try:
                    return final_step.fit_transform(Xt, y)
                except TypeError:
                    return final_step.fit_transform(Xt)
            return final_step.fit_transform(Xt)

        if not hasattr(final_step, "fit") or not hasattr(final_step, "transform"):
            raise TypeError(
                f"Final step '{final_name}' must implement fit_transform or fit and transform"
            )

        if y is not None:
            try:
                final_step.fit(Xt, y)
            except TypeError:
                final_step.fit(Xt)
        else:
            final_step.fit(Xt)
        return final_step.transform(Xt)

    def predict(self, X):
        Xt = X

        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise TypeError(f"Intermediate step '{name}' must implement transform")
            Xt = step.transform(Xt)

        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "predict"):
            raise TypeError(f"Final step '{final_name}' does not implement predict")
        return final_step.predict(Xt)
