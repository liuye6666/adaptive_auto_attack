model_name = 'TRADES_mnist'
dataset='mnist'
ep = 0.29999996#0.3
batch_size = 10000
random=True
average_number=2000
device=None
kwargs=dict(
    # device=None
    out_restart_num = 40,
    normal_prob = 0.5,
    max_iter0=200,

    warm_restart=False,
    warm_restart_num=1,

    out_re_rule = True,
    orr_start_iter=20,
    orr_max_iter=200,
    orr_restart_num=50,
    orr_adi_iter_num=8,
    orr_alpha=0.01
)