'''
Copyright (c) 2022 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2022-06-27 10:14:14
Email: haimingzhang@link.cuhk.edu.cn
Description: The utility functions for the model.
'''


from timm.models import create_model


def get_model(args):
    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        decoder_depth=args.decoder_depth
    )
    return model