model_name = 'TRADES_mnist'
ep = 0.03
batch_size = 10000
random=True
average_number=2000
device=None
kwargs=dict(
    # device=None
    out_restart_num = 1000,
    normal_prom = 0.5,
    out_re_rule = True
)