model_name = 'TRADES_mnist'
dataset='mnist'
ep = 0.3
batch_size = 10000
random=True
average_number=2000
device=None
kwargs=dict(
    # device=None
    out_restart_num = 50,
    normal_prob = 0.5,
    out_re_rule = False,
    max_iter0=200,
    warm_restart=False,
    warm_restart_num=3
)