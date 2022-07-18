# Learning setting
config = dict(setting="supervisedlearning",

              dataset=dict(name="cifar10",
                           datadir="../data",
                           feature="dss",
                           type="pre-defined"),

              dataloader=dict(shuffle=True,
                              batch_size=20,
                              pin_memory=True),

              model=dict(architecture='ResNet18',
                         type='pre-defined',
                         numclasses=10),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

              optimizer=dict(type="sgd",
                             momentum=0.9,
                             lr=0.01,
                             weight_decay=5e-4),

              dss_strategy=dict(type="Full",
                                fraction=1.0,
                                select_every=20),

              train_args=dict(robust_train=True,
                              epsilon=8,
                              alpha=1,
                              attack_iters=7,
                              delta_init='random',
                              num_epochs=200,
                              device="cuda",
                              print_every=10,
                              results_dir='./results/',
                              print_args=["val_loss", "val_robust_loss", "val_acc", "val_robust_acc", "tst_loss", "tst_robust_loss", "tst_acc", "tst_robust_acc", "time"],
                              return_args=[]
                              )
              )