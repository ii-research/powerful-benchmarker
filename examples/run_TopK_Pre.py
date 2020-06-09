import logging
import argparse
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from util import get_pairwise_stds, get_pairwise_similarity, dist

logging.getLogger().setLevel(logging.INFO)

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--pytorch_home", type=str, default="/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/pretrained")
parser.add_argument("--dataset_root", type=str, default="/media/dl-box/f12286fd-f13c-4fe0-a92d-9f935d6a7dbd/CVPR")
parser.add_argument("--root_experiment_folder", type=str,
                    default="/home/dl-box/WorkBench/CodeBench/PyCharmProject/Project_output/Out_img_metric/pw_bench")
parser.add_argument("--global_db_path", type=str, default=None)
parser.add_argument("--merge_argparse_when_resuming", default=False, action='store_true')
parser.add_argument("--root_config_folder", type=str, default=None)
parser.add_argument("--bayes_opt_iters", type=int, default=0)
parser.add_argument("--reproductions", type=str, default="0")
args, _ = parser.parse_known_args()

if args.bayes_opt_iters > 0:
    from powerful_benchmarker.runners.bayes_opt_runner import BayesOptRunner

    args.reproductions = [int(x) for x in args.reproductions.split(",")]
    runner = BayesOptRunner
else:
    from powerful_benchmarker.runners.single_experiment_runner import SingleExperimentRunner

    runner = SingleExperimentRunner
    del args.bayes_opt_iters
    del args.reproductions

# your custom loss function
ANCHOR_ID = ['Anchor', 'Class']


class TopKPreLoss(losses.BaseMetricLossFunction):
    """
    Sampling Wisely: Deep Image Embedding by Top-K Precision Optimization
    Jing Lu, Chaofan Xu, Wei Zhang, Ling-Yu Duan, Tao Mei; The IEEE International Conference on Computer Vision (ICCV), 2019, pp. 7961-7970
    """

    def __init__(self, k=5, anchor_id='Anchor'): # , use_similarity=False, opt=None):
        super().__init__()

        self.name = 'TopKPreLoss'
        assert anchor_id in ANCHOR_ID

        # self.opt = opt
        self.anchor_id = anchor_id
        # self.use_similarity = use_similarity

        self.k = k
        self.margin = 0.1 # self.opt.margin

        # if 'Class' == anchor_id:置いておく
        #     assert 0 == self.opt.bs % self.opt.samples_per_class
        #     self.num_distinct_cls = int(self.opt.bs / self.opt.samples_per_class)

    def compute_loss(self, embeddings, labels, indices_tuple): # (simi_mat, cls_match_mat, k=5, margin=None):
        '''
        assuming no-existence of classes with a single instance == samples_per_class > 1
        :param sim_mat: [batch_size, batch_size] pairwise similarity matrix, without removing self-similarity
        :param cls_match_mat: [batch_size, batch_size] v_ij is one if d_i and d_j are of the same class, zero otherwise
        :param k: cutoff value
        :param margin:
        :return:
        '''
        # print('conpute loss')
        # print('embeddings size', embeddings.size())
        # print('labels size', labels.size())

        simi_mat = get_pairwise_similarity(batch_reprs=embeddings)
        cls_match_mat = get_pairwise_stds(
            batch_labels=labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise
        # print('simi mat', simi_mat.size())
        # print('class match mat', cls_match_mat.size())

        simi_mat_hat = simi_mat + (1.0 - cls_match_mat) * self.margin  # impose margin

        ''' get rank positions '''
        _, orgp_indice = torch.sort(simi_mat_hat, dim=1, descending=True)
        _, desc_indice = torch.sort(orgp_indice, dim=1, descending=False)
        rank_mat = desc_indice + 1.  # todo using desc_indice directly without (+1) to improve efficiency
        # print('rank_mat', rank_mat)

        # number of true neighbours within the batch
        batch_pos_nums = torch.sum(cls_match_mat, dim=1)

        ''' get proper K rather than a rigid predefined K
        torch.clamp(tensor, min=value) is cmax and torch.clamp(tensor, max=value) is cmin.
        It works but is a little confusing at first.
        '''
        # batch_ks = torch.clamp(batch_pos_nums, max=k)
        '''
        due to no explicit self-similarity filtering.
        implicit assumption: a common L2-normalization leading to self-similarity of the maximum one!
        '''
        batch_ks = torch.clamp(batch_pos_nums, max=self.k + 1)
        k_mat = batch_ks.view(-1, 1).repeat(1, rank_mat.size(1))
        # print('k_mat', k_mat.size())

        '''
        Only deal with a single case: n_{+}>=k
        step-1: determine set of false positive neighbors, i.e., N, i.e., cls_match_std is zero && rank<=k

        step-2: determine the size of N, i.e., |N| which determines the size of P

        step-3: determine set of false negative neighbors, i.e., P, i.e., cls_match_std is one && rank>k && rank<= (k+|N|)
        '''
        # N
        batch_false_pos = (cls_match_mat < 1) & (rank_mat <= k_mat)  # torch.uint8 -> used as indice
        # print('batch_false_pos', batch_false_pos) bool
        batch_fp_nums = torch.sum(batch_false_pos.float(), dim=1)  # used as one/zero
        # print('batch_fp_nums', batch_fp_nums)

        # P
        batch_false_negs = cls_match_mat.bool() & (rank_mat > k_mat)  # all false negative

        ''' just for check '''
        # batch_fn_nums = torch.sum(batch_false_negs.float(), dim=1)
        # print('batch_fn_nums', batch_fn_nums)

        # batch_loss = 0
        batch_loss = torch.tensor(0., requires_grad=True).cuda()
        for i in range(cls_match_mat.size(0)):
            fp_num = int(batch_fp_nums.data[i].item())
            if fp_num > 0:  # error exists, in other words, skip correct case
                # print('fp_num', fp_num)
                all_false_neg = simi_mat_hat[i, :][batch_false_negs[i, :]]
                # print('all_false_neg', all_false_neg)
                top_false_neg, _ = torch.topk(all_false_neg, k=fp_num, sorted=False, largest=True)
                # print('top_false_neg', top_false_neg)

                false_pos = simi_mat_hat[i, :][batch_false_pos[i, :]]

                loss = torch.sum(false_pos - top_false_neg)
                batch_loss += loss

        return batch_loss

    # def forward(self, batch_reprs, batch_labels):
    #     '''
    #     :param batch_reprs:  torch.Tensor() [(BS x embed_dim)], batch of embeddings
    #     :param batch_labels: [(BS x 1)], for each element of the batch assigns a class [0,...,C-1]
    #     :return:
    #     '''
    #     print('forward')
    #
    #     cls_match_mat = get_pairwise_stds(
    #         batch_labels=batch_labels)  # [batch_size, batch_size] S_ij is one if d_i and d_j are of the same class, zero otherwise
    #
    #     if self.use_similarity:
    #         sim_mat = get_pairwise_similarity(batch_reprs=batch_reprs)
    #     else:
    #         dist_mat = dist(batch_reprs=batch_reprs, squared=False)  # [batch_size, batch_size], pairwise distances
    #         sim_mat = -dist_mat
    #
    #     if 'Class' == self.anchor_id:  # vs. anchor wise sorting
    #         cls_match_mat = cls_match_mat.view(self.num_distinct_cls, -1)
    #         sim_mat = sim_mat.view(self.num_distinct_cls, -1)
    #
    #     batch_loss = self.compute_loss(simi_mat=sim_mat, cls_match_mat=cls_match_mat, k=self.k, margin=self.margin)
    #
    #     return batch_loss


if __name__ == '__main__':
    r = runner(**(args.__dict__))
    # make the runner aware of them
    r.register("loss", TopKPreLoss)
    # print('register', r.register("loss", TopKPreLoss))
    # r.register("miner", TopKPreMining)
    r.run()