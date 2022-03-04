def outer_objective(clf,param_function):
    def inner_objective(trial):
        params = param_function(trial)
        model = clf.set_params(**params).fit(x_train,y_train)

        pred = model.predict(x_test)

        f1 = f1_score(y_test,pred)

        return f1
    return inner_objective

def rf_param_function(trial):
    params = {
        "n_estimators":trial.suggest_int("n_estiamtors",100,500,step=100),
        "max_depth":trial.suggest_int("max_depth", 2, 32, log=True)
    }
    return params

def hgbc_param_function(trial):
    params = {
        "max_iter":trial.suggest_int("max_iter",100,500,step=100),
        "max_depth":trial.suggest_int("max_depth", 2, 32, log=True)
    }
    return params

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(outer_objective(RandomForestClassifier(),rf_param_function),n_trials=10)

hgbc_study = optuna.create_study(direction="maximize")
hgbc_study.optimize(outer_objective(HistGradientBoostingClassifier(),hgbc_param_function),n_trials=10)