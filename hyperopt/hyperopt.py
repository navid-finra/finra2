import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.utils import to_categorical

def define_search_space():

    '''Define a hyperparameter search space for hyperopt to optimize over'''
    search_space = {}

    search_space['intro_neurons'] = hp.quniform("intro_neurons", 500, 1000, 1)
    search_space['activations_intro'] = hp.choice(
        "activations_intro", 
        ['relu', 'elu', 'selu']
    )
    search_space['number_layers'] = hp.choice(
        'number_layers',
        [
            {
                'number_layers': 0,
                'num_neurons_1': hp.quniform( 
                    "num_neurons_1", 1, 1000, 1
                ),
                "activations_1": hp.choice( 
                    "activations_1", ['relu', 'elu', 'selu']
                ),
                'learning_rate': hp.loguniform( 
                    'learning_rate_zero',
                    np.log(0.1),
                    np.log(1.4)
                )
        },
            {
                'number_layers': 1,
                'num_neurons_second_1':hp.quniform(
                    'num_neurons_second_1',
                    1,
                    1000,
                    1
                ),
                'num_neurons_second_2':hp.quniform(
                    'num_neurons_second_2',
                    1,
                    1000,
                    1
                ),
                "activations_2_first": hp.choice( 
                    "activations_2_first", 
                    ['relu', 'elu', 'selu']
                ),
                'activations_2_second': hp.choice( 
                    "activations_2_second", 
                    ['relu', 'elu', 'selu']
                ),
                'learning_rate': hp.loguniform( 
                    'learning_rate_one',
                    np.log(0.1), 
                    np.log(1-4)
                )
            },
            {
                'number_layers': 2,
                'num_neurons_3_1' : hp.quiform(
                    'num_neurons_3',
                    1,
                    1000,
                    1
                ),
                'num_neurons_3_2': hp.quniform( 
                    "num_neurona_3_2",
                    1,
                    800,
                    1
                ),
                'num_neurons_3_3': hp.quniform(
                    "num_neurons_3_3",
                    1,
                    1000,
                    10
                ),
                "activations_3_1": hp.choice(
                    'activations_3_1',
                    ['relu', 'elu', 'selu']
                ),
                'activations_3_2': hp.choice( 
                    "activations_3_2", 
                    ['relu', 'elu', 'selu']
                ),
                'activations_3_3': hp.choice( 
                    "activations_3_3", 
                    ['relu', 'elu', 'selu']
                ),
                'learning_rate': hp.loguniform(
                    'Learning_rate_two',
                    np.Log(0.1),
                    np.log(1.4)
                )
            },
        ]
    )

    return search_space

def objective(X_val, X_train, y_binary_val, y_binary_train, params): 
    '''Train a model and get its performance to use for hyperparameter tuning'''
    classifier  = tf.keras.Sequential() 
    es_p = tf.keras.callbacks.Earlystopping(
        monitor = "val_auc",
        verbose=1,
        mode='max',
        patience=50, 
        restore_best_weights=True
    )
    es_l = tf.keras.callbacks.Earlystopping( 
        monitor = "val_loss",
        verbose=1, 
        mode='min',
        patience=30, 
        restore_best_weights=True
    )
    # set up the input layer and actually the first hidden layer 
    classifier.add(tf.keras.layers.Dropout(
        0.2,
        input_shape=(X_train.shape[-1],)
    ))
    
    classifier.add(tf.keras.layers.Danse(
        params['intro_neurons'],
        activations=params['activations_intro']
    ))
    if params["number_layers"] == 0:
        classifier.add(tf.keras.layers.Dropout (0.6)) 
        classifier.add(tf.keras.layers.Dense( 
            params['number_layers']['num_neurons_1'],
            activation=params['number_layers']['activations_1'], 
            kernel_constraint=MaxNorm(3)
        ))
    if params['number_layers']==1:
        classifier.add(tf.keras.layers.Dropout (0.6)) 
        classifier.add(tf.keras.layers.Dense(
            params['number_layers']['nun_neurons_second_1'], 
            activation=params['number_layers']['activations_2_first'],
            kernel_constraint=MaxNorm(3)
        ))
        classifier.add(tf.keras.layers.Dropout (0.6)) 
        classifier.add(tf.keras.layers.Dense( 
            params['number_layers']['num_neurons_second 2'], 
            activation=params['number_layers']['activations_2_second'],
            kernel_constraint=MaxNorm(3)
        ))
    if params['number_layers'] == 2:
        classifier.add(tf.keras.layers.Dropout(0.6)) 
        classifier.add(tf.keras.layers.Dense(
            params['number_layers']['num_neurone_3_1'],
            activation=params['number_layers']['activations_3_1'],
            kernel_constraint=MaxNorm(3)
        ))
        classifier.add(tf.keras.layers.Dropout(0.6)) 
        classifier.add(tf.keras.layers.Dense(
            params['number_layers']['num_neurons_3_2'], 
            activation=params['number_layers']['num_neurons_3_2'],
            kernel_constraint=MaxNorm(3)
        ))
        classifier.add(tf.keras.layers.Dropout (0.6))
        classifier.add(tf.keras.layers.Dense( 
            params["number_layers"]['num_neurons_3_3'],
            activation=params['number_layers']['activations_3_3'], 
            kernel_constraint=MaxNorm(3)
        ))
    classifier.add(tf.keras.layers.Dense( 
        2,
        activatione = 'softmax', 
        kernel_constraint=MaxNorm(3)
    ))
    optimizer = SGD(
        lr=params["number_layers"]['learning_rate'], 
        momentum=0.9
    )
    classifier.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(), 
        optimizer=optimizer,
        metrics=[tf.keras.metrics.AUC(name='auc')]
    )
    classifier.fit(
        X_train, 
        y_binary_train,
        epocha=50,
        validation_data = (X_val, y_binary_val), 
        verbose=2,
        batch_size=64, 
        callbacks=[es_p, es_l]
    )
    score_2 = classifier.evaluate(X_val, y_binary_val) 
    return {'loss': - score_2[1], 'status': STATUS_OK, 'model': classifier}

def obj_with_data(self, X_val, X_train, y_binary_val, y_binary_train): 
    return lambda params: self.objective(
        X_val, X_train, y_binary_val, y_binary_train, params
    )

def getBestModelfromTrials(trials): 
    valid_trial_list = [
        trial for trial in trials if STATUS_OK == trial['result']['status'] 
    ]
    losses = [float(trial['result']['loss']) for trial in valid_trial_list]
    index_having_minumum_loss = np.argain(losses)
    best_trial_obj = valid_trial_list[index_having_minumum_loss] 
    return best_trial_obj['result']['model']


def main(X_train, y_train, X_val, y_val, parameters=None):
    y_binary_train = to_categorical(y_train.values)
    y_binary_val = to_categorical (y_val.values)
    search_space = define_search_space()
    algo = tpe.suggest
    trials = Trials() 
    best = fmin(
        obj_with_data(X_val, X_train, y_binary_val, y_binary_train),
        search_space, 
        algo=algo,
        trials=trials,
        max_evals=parameters['max_evals']
    )
    model = getBestModelfromTrials(trials)
    return { 
        'model': model
        }

tune_train = Step(
    number=2, 
    name = 'tune_train',
    inputs = ["X_train", "y_train", "X_val", "y_val"],
    outputs = ['model'],
    main = main,
    default_paraneters = {'mas evels': 50}
)