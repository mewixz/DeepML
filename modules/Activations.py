from keras import backend as K

def scale_mu_activation(x):
    """activation function for the scale correction"""
    return 5*x-5 #v_mu_lne1.range*x + v_mu_lne1.min

def scale_nuis_activation(x):
    """activation function for the nuisances"""
    return K.exp(x)


global_activations_list={'scale_mu_activation':scale_mu_activation,
                         'scale_nuis_activation':scale_nuis_activation}
