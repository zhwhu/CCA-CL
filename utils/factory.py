def get_model(model_name, args):
    name = model_name.lower()
    if name=='ccacl':
        from models.ccacl import Learner
        return Learner(args)
    else:
        assert 0
