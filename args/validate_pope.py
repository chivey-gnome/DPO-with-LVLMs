
def is_pope(args):
    return args.benchmark is not None and args.benchmark == "pope"

def validate_pope(args):
    return args.pope_path is not None and args.coco_path is not None and args.set_name is not None