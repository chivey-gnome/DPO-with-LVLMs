def is_amber(args):
    return args.benchmark is not None and args.benchmark == "amber"

def validate_amber(args):
    return args.amber_path is not None and args.sim_score is not None and args.amber_set_name is not None