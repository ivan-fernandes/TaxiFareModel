from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

class ParamTrainer:

    def __init__(self, X, y, pipeline):
        self.pipeline = pipeline
        self.X_train = X
        self.y_train = y

    def train(self, params):
        self.params = params

        grid_search = GridSearchCV(
        self.pipeline, 
        param_grid={
            'features__distance__standardscaler__copy': [True],
            'model__min_samples_leaf': [3],
            'model__oob_score': [True],
            'model__min_weight_fraction_leaf': [0.0, 0.1]
            },
        cv=5)

        grid_search.fit(self.X_train, self.y_train)
        grid_search.score(self.X_test, self.y_test)


        params = dict(
            random_forest = dict(
                line_count = 1_000,
                hyper_params = dict(
                    features__distance__distancetransformer__distance_type = ["euclidian", "manhattan"],
                    features__distance__standardscaler__with_mean = [True, False],
                    model__max_depth = [1, 2, 3]
                )
            ),
            linear_regression = dict(
                line_count = 1_000,
                hyper_params = dict(
                    features__distance__distancetransformer__distance_type = ["euclidian", "manhattan"],
                    features__distance__standardscaler__with_mean = [True, False],
                    model__normalize = [True, False]
                )
            )
        )