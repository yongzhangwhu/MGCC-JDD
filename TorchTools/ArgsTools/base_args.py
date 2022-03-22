import os


class BaseArgs():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--gpu_num', default='0', type=str,  help='gpu num')
        # datasets args
        parser.add_argument('--train_list', type=str,
                            default='',  help='path to train list')
        parser.add_argument('--valid_list', type=str,
                            default='', help='path to valid list')

        parser.add_argument('--max_noise', default=0.0784, type=float, help='noise_level')
        parser.add_argument('--min_noise', default=0.00, type=float, help='noise_level')

        parser.add_argument('--batch_size', default=16, type=int, help='mini-batch size')
        parser.add_argument('--patch_size', default=256, type=int, help='height of patch')

        # train args
        parser.add_argument('--total_epochs', default=10000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--total_iters', default=10000000, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--start_iters', default=0, type=int,
                            help='number of total epochs to run')
        parser.add_argument('--lr', default=1e-4, type=float,
                            help='initial learning rate')
        parser.add_argument('--log_path', default='./train.log', type=str,
                            help='initial log path')

        # valid args
        parser.add_argument('--valid_freq', default=5000, type=int,
                            help='epoch interval for valid (default:5000)')
        # logger parse
        parser.add_argument('--print_freq', default=20, type=int,
                            help='print frequency (default: 100)')
        parser.add_argument('--save_path', type=str, default='',
                            help='path of save model')
        parser.add_argument('--save_freq', default=10000, type=int,
                            help='save ckpoints frequency')

        # model args
        parser.add_argument('--pretrained_model', default='', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--model', default='MGCC', type=str,
                            help='path to pretrained model(default: none)')
        parser.add_argument('--norm_type', default=None, type=str,
                            help='dm_block_type(default: rrdb)')
        parser.add_argument('--block_type', default='rcab', type=str,
                            help='dm_block_type(default: rcab)')
        parser.add_argument('--act_type', default='prelu', type=str,
                            help='activation layer {relu, prelu, leakyrelu}')
        parser.add_argument('--bias', action='store_true',
                            help='bias of layer')
        parser.add_argument('--channels', default=64, type=int,
                            help='channels')
        parser.add_argument('--n_resblocks', default=6, type=int,
                            help='number of basic blocks')

        parser.add_argument('--postname',default='', type=str, help='postname')
        
        self.args = parser.parse_args()

      
        self.args.post = self.args.model + '-dn-'+ self.args.train_list.split('/')[-1].split('.')[0].split('_')[-1]+'x'\
                             + str(self.args.n_resblocks)+'-'+'-'+str(self.args.channels) + '-' + '-' + self.args.block_type
        if self.args.postname:
            self.args.post = self.args.post +'-' + self.args.postname

        self.args.save_path = 'checkpoints/checkpoints'+'-'+self.args.post
        self.args.logdir = 'logs/'+self.args.post

        self.initialized = True
        return self.args

    def print_args(self):
        # print args
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("\n")
        print("==========     CONFIG END    =============")
        
        # check for folders existence 
        if os.path.exists(self.args.logdir):
            cmd = 'rm -rf ' + self.args.logdir
            os.system(cmd)
        os.makedirs(self.args.logdir)

        if not os.path.exists(self.args.save_path):		
            os.makedirs(self.args.save_path)

        assert os.path.exists(self.args.train_list), 'train_list {} not found'.format(self.args.train_list)
        assert os.path.exists(self.args.valid_list), 'valid_list {} not found'.format(self.args.valid_list)


