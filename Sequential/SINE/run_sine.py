from recbole.quick_start import run_recbole

parameter_dict = {
   'neg_sampling': None,
}

if __name__ == '__main__':
    run_recbole(model='SINE', dataset='ml-1m', config_dict=parameter_dict)

