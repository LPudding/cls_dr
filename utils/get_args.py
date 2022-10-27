import argparse
# print('hello')
# class Config(object):
#     # config params
#     data_dir=None
#     saved_dir=None
#     logs_dir=None
#     num_workers=4
#     model_name='resnext50_32x4d'
#     image_size=512
#     epochs=10
#     lr=1e-4
#     min_lr=1e-7
#     train_batch_size=32
#     val_batch_size=64
#     weight_decay=1e-6
#     gradient_accumulation_steps=1
#     max_grad_norm=1000
#     seed=42
#     label_size=11
#     label_col=['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
#                  'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
#                  'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
#                  'Swan Ganz Catheter Present']
#     n_fold=5
#     trn_fold=[0, 1, 2, 3, 4]

def get_argparse():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--saved_dir", default=None, type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--logs_dir", default=None, type=str, required=False,
                        help="The logs directory.")
    parser.add_argument("--model_name", default=None, type=str, required=False,
                        help="What model to use in training!")
    parser.add_argument("--image_size", default=512, type=int, required=False,
                        help="The image size using!")
    parser.add_argument("--early_stop_patience", default=5, type=int, required=False,
                        help="Early stop patience")

    parser.add_argument("--weight_decay", default=1e-6, type=float, required=False,
                        help="Weight decay if we apply some.")
    parser.add_argument("--lr", default=1e-4, type=float, required=False,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, required=False,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--epochs", default=10, type=int, required=False,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--train_batch_size', default=32, type=int, required=False,
                        help='Train batch size')
    parser.add_argument('--val_batch_size', default=64, type=int, required=False,
                        help='Val batch size')
    parser.add_argument('--num_workers', default=4, type=int, required=False, help='DataLoader num_workers')
    parser.add_argument('--accumulation_steps', default=1, type=int,
                        help='accumulate the grad to update')

    parser.add_argument("--seed", type=int, default=42, required=False,
                        help="random seed for initialization")
    

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # args = get_argparse()
    # args_dict = vars(args)
    print(1)
    # print(args_dict)