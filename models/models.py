
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'gea_gan':
        # assert(opt.dataset_mode == 'aligned')
        from .gea_gan_model import gea_ganModel
        model = gea_ganModel()
    elif opt.model == 'dea_gan':
        # assert(opt.dataset_mode == 'aligned')
        from .dea_gan_model import dea_ganModel
        model = dea_ganModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
