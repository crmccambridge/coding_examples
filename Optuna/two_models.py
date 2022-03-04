def rf_objective(trial):
    
    params = {
        "n_estimators":trial.suggest_int("n_estiamtors",100,500,step=100),
        "max_depth":trial.suggest_int("max_depth", 2, 32, log=True)
    }

    model = RandomForestClassifier().set_params(**params).fit(x_train,y_train)

    pred = model.predict(x_test)

    f1 = f1_score(y_test,pred)

    return f1

def hgbc_objective(trial):

    params = {
        "max_iter":trial.suggest_int("max_iter",100,500,step=100),
        "max_depth":trial.suggest_int("max_depth", 2, 32, log=True)
    }

    model = HistGradientBoostingClassifier().set_params(**params).fit(x_train,y_train)

    pred = model.predict(x_test)

    f1 = f1_score(y_test,pred)

    return f1

rf_study = optuna.create_study(direction="maximize")
rf_study.optimize(rf_objective,n_trials=10)

hgbc_study = optuna.create_study(direction="maximize")
hgbc_study.optimize(hgbc_objective,n_trials=10)