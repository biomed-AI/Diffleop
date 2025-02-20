import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from tqdm.auto import tqdm

from models.encoders import get_refine_net
from models.common import compose_context, ShiftedSoftplus, GaussianSmearing, \
    to_torch_const, extract
from models.transitions import cosine_beta_schedule, get_beta_schedule, DiscreteTransition, index_to_log_onehot, \
    log_sample_categorical

def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode='protein'):
    if mode == 'none':
        offset = 0.
        pass
    elif mode == 'protein':
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        if ligand_pos is not None:
            ligand_pos = ligand_pos - offset[batch_ligand]
        # print('offset: ', offset)
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset

def center_pos_leop(protein_pos, ligand_pos, batch_protein, batch_ligand, anchor_pos):
    protein_pos = protein_pos - anchor_pos[batch_protein]
    ligand_pos = ligand_pos - anchor_pos[batch_ligand]
    return protein_pos, ligand_pos, anchor_pos


def categorical_kl(log_prob1, log_prob2):
    kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
    return kl


def log_categorical(log_x_start, log_prob):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    KL divergence between normal distributions parameterized by mean and log-variance.
    """
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + (mean1 - mean2) ** 2 * torch.exp(-logvar2))
    return kl.sum(-1)


def log_normal(values, means, log_scales):
    var = torch.exp(log_scales * 2)
    log_prob = -((values - means) ** 2) / (2 * var) - log_scales - np.log(np.sqrt(2 * np.pi))
    return log_prob.sum(-1)


# Time embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Model
class Diffleop(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, num_classes,
                 prior_atom_types=None, prior_bond_types=None):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']

        self.bond_diffusion = getattr(config, 'bond_diffusion', False)
        self.bond_net_type = getattr(config, 'bond_net_type', 'mlp')
        self.center_prox_loss = getattr(config, 'center_prox_loss', False)
        self.armsca_prox_loss = getattr(config, 'armsca_prox_loss', False)
        self.clash_loss = getattr(config, 'clash_loss', False)

        self.sample_time_method = config.sample_time_method  # ['importance', 'symmetric']
        self.loss_pos_type = config.loss_pos_type  # ['mse', 'kl']
        print(f'Loss pos mode {self.loss_pos_type} applied!')

        if config.beta_schedule == 'cosine':
            alphas = cosine_beta_schedule(config.num_diffusion_timesteps, config.pos_beta_s) ** 2
            print('cosine pos alpha schedule applied!')
            betas = 1. - alphas
        else:
            betas = get_beta_schedule(
                beta_schedule=config.beta_schedule,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
                num_diffusion_timesteps=config.num_diffusion_timesteps,
            )
            alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.betas = to_torch_const(betas)
        self.num_timesteps = self.betas.size(0)
        self.alphas_cumprod = to_torch_const(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch_const(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch_const(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_torch_const(np.sqrt(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_torch_const(np.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.posterior_mean_c0_coef = to_torch_const(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_ct_coef = to_torch_const(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_var = to_torch_const(posterior_variance)
        # self.posterior_logvar = to_torch_const(np.log(np.maximum(posterior_variance, 1e-10)))
        self.posterior_logvar = to_torch_const(np.log(np.append(self.posterior_var[1], self.posterior_var[1:])))
        self.pos_score_coef = to_torch_const(betas / np.sqrt(alphas))

        # atom / bond type transition
        self.num_classes = num_classes
        self.num_bond_classes = getattr(config, 'num_bond_classes', 1)

        self.atom_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_classes, prior_probs=prior_atom_types
        )
        self.bond_type_trans = DiscreteTransition(
            config.v_beta_schedule, self.num_timesteps,
            s=config.v_beta_s, num_classes=self.num_bond_classes, prior_probs=prior_bond_types
        )

        self.register_buffer('Lt_history', torch.zeros(self.num_timesteps))
        self.register_buffer('Lt_count', torch.zeros(self.num_timesteps))

        # model definition
        self.hidden_dim = config.hidden_dim
        print('Node indicator: ', self.config.node_indicator)
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        self.refine_net_type = config.model_type

        # time embedding
        self.time_emb_dim = config.time_emb_dim
        self.time_emb_mode = config.time_emb_mode  # ['simple', 'sin']
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + 1, emb_dim)
            elif self.time_emb_mode == 'sin':
                self.time_emb = nn.Sequential(
                    SinusoidalPosEmb(self.time_emb_dim),
                    nn.Linear(self.time_emb_dim, self.time_emb_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.time_emb_dim * 4, self.time_emb_dim)
                )
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim + self.time_emb_dim, emb_dim)
            else:
                raise NotImplementedError
        else:
            if self.refine_net_type == 'uni_o2_bond':
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)
            else:
                self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)

        
        self.refine_net = get_refine_net(self.refine_net_type, config)

        self.ligand_bond_emb = nn.Linear(self.num_bond_classes, self.hidden_dim)

        # atom type prediction
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, self.num_classes),
        )
        if self.bond_diffusion:
            self.distance_expansion = GaussianSmearing(0., 5., num_gaussians=config.num_r_gaussian, fix_offset=False)
            if self.bond_net_type == 'pre_att':
                bond_input_dim = config.num_r_gaussian + self.hidden_dim
            elif self.bond_net_type == 'lin':
                bond_input_dim = self.hidden_dim
            else:
                raise ValueError(self.bond_net_type)
            self.bond_inference = nn.Sequential(
                nn.Linear(bond_input_dim, self.hidden_dim),
                ShiftedSoftplus(),
                nn.Linear(self.hidden_dim, self.num_bond_classes)
            )
        
        self.expert_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )

    def _predict_x0_from_eps(self, xt, eps, t, batch):
        pos0_from_e = extract(self.sqrt_recip_alphas_cumprod, t, batch) * xt - \
                      extract(self.sqrt_recipm1_alphas_cumprod, t, batch) * eps
        return pos0_from_e

    def q_pos_posterior(self, x0, xt, t, batch):
        # Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        pos_model_mean = extract(self.posterior_mean_c0_coef, t, batch) * x0 + \
                         extract(self.posterior_mean_ct_coef, t, batch) * xt
        return pos_model_mean

    def kl_pos_prior(self, pos0, batch):
        num_graphs = batch.max().item() + 1
        a_pos = extract(self.alphas_cumprod, [self.num_timesteps - 1] * num_graphs, batch)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(pos0)
        pos_noise.normal_()
        pos_perturbed = a_pos.sqrt() * pos0 + (1.0 - a_pos).sqrt() * pos_noise
        pos_prior = torch.randn_like(pos_perturbed)
        kl_prior = torch.mean((pos_perturbed - pos_prior) ** 2)
        return kl_prior

    def sample_time(self, num_graphs, device, method):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self.sample_time(num_graphs, device, method='symmetric')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            time_step = torch.multinomial(pt_all, num_samples=num_graphs, replacement=True)
            pt = pt_all.gather(dim=0, index=time_step)
            return time_step, pt

        elif method == 'symmetric':
            time_step = torch.randint(
                0, self.num_timesteps, size=(num_graphs // 2 + 1,), device=device)
            time_step = torch.cat(
                [time_step, self.num_timesteps - time_step - 1], dim=0)[:num_graphs]
            pt = torch.ones_like(time_step).float() / self.num_timesteps
            return time_step, pt

        else:
            raise ValueError

    def compute_pos_Lt(self, pos_model_mean, x0, xt, t, batch):
        # fixed pos variance
        pos_log_variance = extract(self.posterior_logvar, t, batch)
        pos_true_mean = self.q_pos_posterior(x0=x0, xt=xt, t=t, batch=batch)
        kl_pos = normal_kl(pos_true_mean, pos_log_variance, pos_model_mean, pos_log_variance)
        kl_pos = kl_pos / np.log(2.)

        decoder_nll_pos = -log_normal(x0, means=pos_model_mean, log_scales=0.5 * pos_log_variance)
        assert kl_pos.shape == decoder_nll_pos.shape
        mask = (t == 0).float()[batch]
        loss_pos = scatter_mean(mask * decoder_nll_pos + (1. - mask) * kl_pos, batch, dim=0)
        return loss_pos

    def compute_v_Lt(self, log_v_model_prob, log_v0, log_v_true_prob, t, batch):
        kl_v = categorical_kl(log_v_true_prob, log_v_model_prob)  # [num_atoms, ]
        decoder_nll_v = -log_categorical(log_v0, log_v_model_prob)  # L0
        assert kl_v.shape == decoder_nll_v.shape
        mask = (t == 0).float()[batch]
        loss_v = scatter_mean(mask * decoder_nll_v + (1. - mask) * kl_v, batch, dim=0)
        return loss_v

    def get_diffusion_loss(
            self, protein_pos, protein_v, batch_protein,
            ligand_pos, ligand_v, ligand_v_aux, batch_ligand,

            ligand_fc_bond_index=None, ligand_fc_bond_type=None,
            time_step=None,

            anchor_pos = None, mask_mask = None, mask_edge_mask = None,
            ligand_mask_mask_b_batch = None, ligand_mask_edge_mask_b_batch = None,

            anchor_feature = None, true_affinity = None,
    ):
        num_graphs = batch_protein.max().item() + 1
        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            pt = torch.ones_like(time_step).float() / self.num_timesteps
        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos, v, (and bond)
        protein_pos, ligand_pos, offset = center_pos_leop(
            protein_pos, ligand_pos, batch_protein, batch_ligand, anchor_pos)
        ligand_pos_ = ligand_pos.clone()
        mask_pos = ligand_pos_[mask_mask]

        a_pos = a[ligand_mask_mask_b_batch].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(mask_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        
        ligand_mask_pos_perturbed = a_pos.sqrt() * mask_pos + \
                               (1.0 - a_pos).sqrt() * pos_noise
        ligand_pos_[mask_mask] = ligand_mask_pos_perturbed
        ligand_pos_perturbed = ligand_pos_

        # Vt = a * V0 + (1-a) / K
        ligand_v_ = ligand_v.clone()
        mask_v = ligand_v_[mask_mask]
        log_ligand_v0 = index_to_log_onehot(mask_v, self.num_classes)
        ligand_mask_v_perturbed, log_ligand_vt = self.atom_type_trans.q_v_sample(log_ligand_v0, time_step, ligand_mask_mask_b_batch)
        ligand_v_[mask_mask] = ligand_mask_v_perturbed
        ligand_v_perturbed = ligand_v_

        if self.bond_diffusion:
            ligand_fc_bond_type_ = ligand_fc_bond_type.clone()
            mask_b = ligand_fc_bond_type_[mask_edge_mask]
            log_ligand_b0 = index_to_log_onehot(mask_b, self.num_bond_classes)
            ligand_mask_b_perturbed, log_ligand_bt = self.bond_type_trans.q_v_sample(
                log_ligand_b0, time_step, ligand_mask_edge_mask_b_batch)
            ligand_fc_bond_type_[mask_edge_mask] = ligand_mask_b_perturbed
            ligand_b_perturbed = ligand_fc_bond_type_
        else:
            ligand_b_perturbed = None

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self.forward(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,

            init_ligand_pos=ligand_pos_perturbed,
            init_ligand_v=ligand_v_perturbed,
            init_ligand_v_aux=ligand_v_aux,
            batch_ligand=batch_ligand,
            ligand_fc_bond_index=ligand_fc_bond_index,
            init_ligand_fc_bond_type=ligand_b_perturbed,

            time_step=time_step,

            mask_mask = mask_mask,

            anchor_feature = anchor_feature,
        )

        pred_ligand_pos, pred_ligand_v = preds['pred_ligand_pos'][mask_mask], preds['pred_ligand_v'][mask_mask]
        pred_pos_noise = pred_ligand_pos - ligand_pos_perturbed[mask_mask]

        # atom type
        log_ligand_v_recon = F.log_softmax(pred_ligand_v, dim=-1)
        log_v_model_prob = self.atom_type_trans.q_v_posterior(log_ligand_v_recon, log_ligand_vt, time_step, ligand_mask_mask_b_batch)
        log_v_true_prob = self.atom_type_trans.q_v_posterior(log_ligand_v0, log_ligand_vt, time_step, ligand_mask_mask_b_batch)

        # compute kl
        kl_v = self.compute_v_Lt(log_v_model_prob=log_v_model_prob, log_v0=log_ligand_v0,
                                 log_v_true_prob=log_v_true_prob, t=time_step, batch=ligand_mask_mask_b_batch)
        if self.bond_diffusion:
            log_ligand_b_recon = F.log_softmax(preds['pred_bond'][mask_edge_mask], dim=-1)
            log_b_model_prob = self.bond_type_trans.q_v_posterior(
                log_ligand_b_recon, log_ligand_bt, time_step, ligand_mask_edge_mask_b_batch)
            log_b_true_prob = self.bond_type_trans.q_v_posterior(
                log_ligand_b0, log_ligand_bt, time_step, ligand_mask_edge_mask_b_batch)
            kl_b = self.compute_v_Lt(log_v_model_prob=log_b_model_prob, log_v0=log_ligand_b0,
                                     log_v_true_prob=log_b_true_prob, t=time_step, batch=ligand_mask_edge_mask_b_batch)
            loss_bond = torch.mean(kl_b)
        else:
            loss_bond = torch.tensor(0.)

        if self.loss_pos_type == 'mse':
            # unweighted
            if self.model_mean_type == 'C0':
                target, pred = ligand_pos[mask_mask], pred_ligand_pos
            elif self.model_mean_type == 'noise':
                target, pred = pos_noise, pred_pos_noise
            else:
                raise ValueError
            loss_pos = scatter_mean(((pred - target) ** 2).sum(-1), ligand_mask_mask_b_batch, dim=0)
            loss_pos = torch.mean(loss_pos)
        else:
            raise ValueError
        loss_v = torch.mean(kl_v)

        true_affinity = (true_affinity - 0.) / (13. - 0.)
        loss_aff = F.mse_loss(preds['final_aff_pred'].view(-1, 1).float(), true_affinity.view(-1, 1).float())

        results = {
            'losses': {
                'pos': loss_pos,
                'v': loss_v,
                'aff': loss_aff,
            },
            'x0': ligand_pos,
            'pred_ligand_pos': pred_ligand_pos,
            'pred_ligand_v': pred_ligand_v,
            'pred_pos_noise': pred_pos_noise,
            'ligand_v_recon': F.softmax(pred_ligand_v, dim=-1)
        }

        if self.bond_diffusion:
            results['losses']['bond'] = loss_bond
            results['ligand_b_recon'] = F.softmax(preds['pred_bond'][mask_edge_mask], dim=-1)
        return results

    def aff_guide(self, protein_pos, protein_v, batch_protein, 
                init_ligand_pos_, init_ligand_v, init_ligand_v_aux, batch_ligand, 

                ligand_fc_bond_index, init_ligand_fc_bond_type,
                time_step=None, return_all=False,
                mask_mask = None, anchor_feature = None,
        ):
        with torch.enable_grad():
            # init_ligand_v_detach = F.one_hot(init_ligand_v, self.num_classes).float()#.detach().requires_grad_(True)
            init_ligand_v_detach = F.one_hot(init_ligand_v, self.num_classes).float().detach().requires_grad_(True)

            # add decomp indicator feature
            anchor_feature = anchor_feature.view(-1, 1).to(torch.float32)
            init_ligand_v = torch.cat([init_ligand_v_detach, init_ligand_v_aux, anchor_feature], -1)
            # time embedding
            if self.time_emb_dim > 0:
                if self.time_emb_mode == 'simple':
                    input_ligand_feat = torch.cat([
                        init_ligand_v,
                        (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                    ], -1)
                elif self.time_emb_mode == 'sin':
                    time_feat = self.time_emb(time_step)
                    input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
                else:
                    raise NotImplementedError
            else:
                input_ligand_feat = init_ligand_v

            h_protein = self.protein_atom_emb(protein_v)
            init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

            if self.config.node_indicator: # True
                pro_ind = torch.tensor([0]).unsqueeze(0).repeat(len(h_protein), 1).to(h_protein)
                lig_ind = torch.tensor([1]).unsqueeze(0).repeat(len(init_ligand_h), 1).to(h_protein)

                h_protein = torch.cat([h_protein, pro_ind], -1)
                init_ligand_h = torch.cat([init_ligand_h, lig_ind], -1)

            init_ligand_pos = init_ligand_pos_.clone().detach()
            init_ligand_pos = init_ligand_pos.detach().requires_grad_(True)

            h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
                h_protein=h_protein,
                h_ligand=init_ligand_h,
                pos_protein=protein_pos,
                pos_ligand=init_ligand_pos,
                batch_protein=batch_protein,
                batch_ligand=batch_ligand,
                ligand_atom_mask=None,
            )
            group_idx_all = None

            if ligand_fc_bond_index is not None:
                bond_index_in_all = l_index_in_ctx[ligand_fc_bond_index]
            else:
                bond_index_in_all = None

            if self.refine_net_type == 'uni_o2_bond':
                # bond_type = F.one_hot(init_ligand_fc_bond_type, num_classes=self.num_bond_classes).float()#.detach().requires_grad_(True)
                bond_type = F.one_hot(init_ligand_fc_bond_type, num_classes=self.num_bond_classes).float().detach().requires_grad_(True)
                h_bond = self.ligand_bond_emb(bond_type)
                outputs = self.refine_net(
                    h=h_all, x=pos_all, group_idx=group_idx_all,
                    bond_index=bond_index_in_all, h_bond=h_bond,
                    mask_ligand=mask_ligand,
                    mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                    batch=batch_all,
                    return_all=return_all,
                    mask_mask = mask_mask,
                )
            else:
                outputs = self.refine_net(
                    h=h_all, x=pos_all,
                    mask_ligand=mask_ligand,
                    mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                    batch=batch_all,
                    return_all=return_all,
                    mask_mask = mask_mask,
                )
            final_pos, final_h = outputs['x'], outputs['h']

            final_ligand_pos, final_ligand_h = final_pos[mask_ligand_atom], final_h[mask_ligand_atom]

            atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
            pred_affinity = scatter_mean(atom_affinity, batch_ligand)

            pred_affinity_log = pred_affinity.log()
            # if gui_scale_pos is not None:
            v_grad = torch.autograd.grad(pred_affinity, init_ligand_v_detach,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            b_grad = torch.autograd.grad(pred_affinity, bond_type,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]
            pos_grad = torch.autograd.grad(pred_affinity_log, init_ligand_pos,grad_outputs=torch.ones_like(pred_affinity),retain_graph=True)[0]


        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_aff_pred': pred_affinity,
        }

        if self.bond_diffusion:
            # bond inference input
            if self.bond_net_type == 'pre_att':
                src, dst = bond_index_in_all
                dist = torch.norm(final_pos[dst] - final_pos[src], p=2, dim=-1, keepdim=True)
                r_feat = self.distance_expansion(dist)
                if self.bond_net_type == 'pre_att':
                    hi, hj = final_h[dst], final_h[src]
                    bond_inf_input = torch.cat([r_feat, (hi + hj) / 2], -1)
                else:
                    raise NotImplementedError
            elif self.bond_net_type == 'lin':
                bond_inf_input = outputs['h_bond']
            else:
                raise ValueError(self.bond_net_type)
            pred_bond = self.bond_inference(bond_inf_input)
            preds.update({
                'pred_bond': pred_bond
            })

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand_atom] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand_atom]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        # v_grad = 0
        # b_grad = 0
        # pos_grad = 0
        return preds, v_grad, b_grad, pos_grad

    def forward(self, protein_pos, protein_v, batch_protein, 
                init_ligand_pos, init_ligand_v, init_ligand_v_aux, batch_ligand, 

                ligand_fc_bond_index, init_ligand_fc_bond_type,
                time_step=None, return_all=False,
                mask_mask = None, anchor_feature = None,
        ):

        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()

        # add decomp indicator feature
        anchor_feature = anchor_feature.view(-1, 1).to(torch.float32)
        init_ligand_v = torch.cat([init_ligand_v, init_ligand_v_aux, anchor_feature], -1)
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
            elif self.time_emb_mode == 'sin':
                time_feat = self.time_emb(time_step)
                input_ligand_feat = torch.cat([init_ligand_v, time_feat], -1)
            else:
                raise NotImplementedError
        else:
            input_ligand_feat = init_ligand_v

        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator: # True
            pro_ind = torch.tensor([0]).unsqueeze(0).repeat(len(h_protein), 1).to(h_protein)
            lig_ind = torch.tensor([1]).unsqueeze(0).repeat(len(init_ligand_h), 1).to(h_protein)

            h_protein = torch.cat([h_protein, pro_ind], -1)
            init_ligand_h = torch.cat([init_ligand_h, lig_ind], -1)

        h_all, pos_all, batch_all, mask_ligand, mask_ligand_atom, p_index_in_ctx, l_index_in_ctx = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
            ligand_atom_mask=None,
        )
        group_idx_all = None

        if ligand_fc_bond_index is not None:
            bond_index_in_all = l_index_in_ctx[ligand_fc_bond_index]
        else:
            bond_index_in_all = None

        if self.refine_net_type == 'uni_o2_bond':
            bond_type = F.one_hot(init_ligand_fc_bond_type, num_classes=self.num_bond_classes).float()
            h_bond = self.ligand_bond_emb(bond_type)
            outputs = self.refine_net(
                h=h_all, x=pos_all, group_idx=group_idx_all,
                bond_index=bond_index_in_all, h_bond=h_bond,
                mask_ligand=mask_ligand,
                mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                batch=batch_all,
                return_all=return_all,
                mask_mask = mask_mask,
            )
        else:
            outputs = self.refine_net(
                h=h_all, x=pos_all,
                mask_ligand=mask_ligand,
                mask_ligand_atom=mask_ligand_atom,  # dummy node is marked as 0
                batch=batch_all,
                return_all=return_all,
                mask_mask = mask_mask,
            )
        final_pos, final_h = outputs['x'], outputs['h']

        final_ligand_pos, final_ligand_h = final_pos[mask_ligand_atom], final_h[mask_ligand_atom]
        final_ligand_v = self.v_inference(final_ligand_h)

        atom_affinity = self.expert_pred(final_ligand_h).squeeze(-1)
        final_exp_pred = scatter_mean(atom_affinity, batch_ligand)

        preds = {
            'pred_ligand_pos': final_ligand_pos,
            'pred_ligand_v': final_ligand_v,
            'final_aff_pred': final_exp_pred,
        }

        if self.bond_diffusion:
            # bond inference input
            if self.bond_net_type == 'pre_att':
                src, dst = bond_index_in_all
                dist = torch.norm(final_pos[dst] - final_pos[src], p=2, dim=-1, keepdim=True)
                r_feat = self.distance_expansion(dist)
                if self.bond_net_type == 'pre_att':
                    hi, hj = final_h[dst], final_h[src]
                    bond_inf_input = torch.cat([r_feat, (hi + hj) / 2], -1)
                else:
                    raise NotImplementedError
            elif self.bond_net_type == 'lin':
                bond_inf_input = outputs['h_bond']
            else:
                raise ValueError(self.bond_net_type)
            pred_bond = self.bond_inference(bond_inf_input)
            preds.update({
                'pred_bond': pred_bond
            })

        if return_all:
            final_all_pos, final_all_h = outputs['all_x'], outputs['all_h']
            final_all_ligand_pos = [pos[mask_ligand_atom] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand_atom]) for h in final_all_h]
            preds.update({
                'layer_pred_ligand_pos': final_all_ligand_pos,
                'layer_pred_ligand_v': final_all_ligand_v
            })
        return preds

    @torch.no_grad()
    def sample_diffusion(
        self, protein_pos, protein_v, batch_protein, 
        ligand_pos, ligand_v, ligand_v_aux, batch_ligand, 

        ligand_fc_bond_index=None, ligand_fc_bond_type=None, batch_ligand_bond=None,
        num_steps=None,

        anchor_pos = None, mask_mask = None, mask_edge_mask = None,
        ligand_mask_mask_b_batch = None, ligand_mask_edge_mask_b_batch = None,

        anchor_feature = None,

        gui_scale_pos = None, gui_scale_node = None, gui_scale_bond = None, 
    ):

        if num_steps is None:
            num_steps = self.num_timesteps
        num_graphs = batch_protein.max().item() + 1
        protein_pos, ligand_pos, offset = center_pos_leop(
            protein_pos, ligand_pos, batch_protein, batch_ligand, anchor_pos)
        
        # init mask pos
        mask_n_nodes_all = len(ligand_mask_mask_b_batch)
        init_ligand_pos = torch.randn([mask_n_nodes_all, 3]).to(ligand_pos.device)
        # init mask node
        uniform_logits = torch.zeros(len(ligand_mask_mask_b_batch), self.num_classes).to(ligand_pos.device)
        init_ligand_v = log_sample_categorical(uniform_logits)
        # init mask bond
        if self.bond_diffusion:
            uniform_logits = torch.zeros(len(ligand_mask_edge_mask_b_batch), self.num_bond_classes).to(ligand_pos.device)
            init_ligand_fc_bond_type = log_sample_categorical(uniform_logits)

        ligand_pos_ = ligand_pos.clone()
        ligand_pos_[mask_mask] = init_ligand_pos
        ligand_v_ = ligand_v.clone()
        ligand_v_[mask_mask] = init_ligand_v
        if self.bond_diffusion:
            ligand_fc_bond_type_ = ligand_fc_bond_type.clone()
            ligand_fc_bond_type_[mask_edge_mask] = init_ligand_fc_bond_type

        pos_traj, v_traj, bond_traj = [], [], []
        v0_pred_traj, vt_pred_traj = [], []
        bt_pred_traj = []
        ligand_pos, ligand_v = ligand_pos_, ligand_v_
        if self.bond_diffusion:
            ligand_bond = ligand_fc_bond_type_
        else:
            ligand_bond = None

        # time sequence
        time_seq = list(reversed(range(self.num_timesteps - num_steps, self.num_timesteps)))
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)

            preds, v_grad, b_grad, pos_grad = self.aff_guide(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos_=ligand_pos,
                init_ligand_v=ligand_v,
                init_ligand_v_aux=ligand_v_aux,
                batch_ligand=batch_ligand,
                ligand_fc_bond_index=ligand_fc_bond_index,
                init_ligand_fc_bond_type=ligand_bond,
                time_step=t,
                mask_mask = mask_mask,
                anchor_feature = anchor_feature,
            )

            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'][mask_mask] - ligand_pos[mask_mask]
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos[mask_mask], eps=pred_pos_noise, t=t, batch=ligand_mask_mask_b_batch)
                v0_from_e = preds['pred_ligand_v'][mask_mask]
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos'][mask_mask]
                v0_from_e = preds['pred_ligand_v'][mask_mask]
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos[mask_mask], t=t, batch=ligand_mask_mask_b_batch)
            pos_log_variance = extract(self.posterior_logvar, t, ligand_mask_mask_b_batch)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[ligand_mask_mask_b_batch].unsqueeze(-1)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v[mask_mask], self.num_classes)

            if self.bond_diffusion:
                log_ligand_b_recon = F.log_softmax(preds['pred_bond'][mask_edge_mask], dim=-1)
                log_ligand_b = index_to_log_onehot(ligand_bond[mask_edge_mask], self.num_bond_classes)

            v_grad = v_grad[mask_mask]
            b_grad = b_grad[mask_edge_mask]
            pos_grad = pos_grad[mask_mask]
            v_grad = v_grad * gui_scale_node
            b_grad = b_grad * gui_scale_bond
            pos_grad = pos_grad * gui_scale_pos

            pos_model_mean += (0.5 * pos_log_variance).exp() * pos_grad
            log_ligand_v += v_grad
            log_ligand_b += b_grad

            log_model_prob = self.atom_type_trans.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, ligand_mask_mask_b_batch)
            ligand_v_next = log_sample_categorical(log_model_prob)

            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())

            # bond
            if self.bond_diffusion:
                log_b_model_prob = self.bond_type_trans.q_v_posterior(
                    log_ligand_b_recon, log_ligand_b, t, ligand_mask_edge_mask_b_batch)
                ligand_b_next = log_sample_categorical(log_b_model_prob)
                ligand_fc_bond_type_[mask_edge_mask] = ligand_b_next
                ligand_bond = ligand_fc_bond_type_
                bt_pred_traj.append(log_b_model_prob.clone().cpu())
                bond_traj.append(ligand_bond.clone().cpu())

            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos[mask_mask])

            ligand_pos_[mask_mask] = ligand_pos_next
            ligand_pos = ligand_pos_

            ligand_v_[mask_mask] = ligand_v_next
            ligand_v = ligand_v_

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]

        ret_pred = {
            'pos': ligand_pos,
            'v': ligand_v,

            'pos_traj': pos_traj, ##
            'v_traj': v_traj, ##
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }
        if self.bond_diffusion:
            ret_pred.update({
                'bond': ligand_bond,
                'bond_traj': bond_traj, ##
                'bt_traj': bt_pred_traj,
            })

        return ret_pred

    @torch.no_grad()
    def hopping(
        self, protein_pos, protein_v, batch_protein, 
        ligand_pos, ligand_v, ligand_v_aux, batch_ligand, 

        ligand_fc_bond_index=None, ligand_fc_bond_type=None, batch_ligand_bond=None, 
        num_steps=None, time_step=None,

        anchor_pos = None, mask_mask = None, mask_edge_mask = None,
        ligand_mask_mask_b_batch = None, ligand_mask_edge_mask_b_batch = None,

        anchor_feature = None,

        gui_scale_pos = None, gui_scale_node = None, gui_scale_bond = None,
    ):
        num_graphs = batch_protein.max().item() + 1
        # 1. sample noise levels
        if time_step is None:
            time_step, pt = self.sample_time(num_graphs, protein_pos.device, self.sample_time_method)
        else:
            time_step = torch.tensor(time_step, device=protein_pos.device)
            time_step = time_step.expand(num_graphs)
            pt = torch.ones_like(time_step).float() / self.num_timesteps

        a = self.alphas_cumprod.index_select(0, time_step)  # (num_graphs, )

        # 2. perturb pos, v, (and bond)
        protein_pos, ligand_pos, offset = center_pos_leop(
            protein_pos, ligand_pos, batch_protein, batch_ligand, anchor_pos)

        a_pos = a[batch_ligand].unsqueeze(-1)  # (num_ligand_atoms, 1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        
        ligand_pos_perturbed = a_pos.sqrt() * ligand_pos + \
                               (1.0 - a_pos).sqrt() * pos_noise
        ligand_pos_ = ligand_pos.clone()
        ligand_pos_[mask_mask] = ligand_pos_perturbed[mask_mask]

        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, self.num_classes)
        ligand_v_perturbed, log_ligand_vt = self.atom_type_trans.q_v_sample(log_ligand_v0, time_step, batch_ligand)
        ligand_v_ = ligand_v.clone()
        ligand_v_[mask_mask] = ligand_v_perturbed[mask_mask]

        if self.bond_diffusion:
            log_ligand_b0 = index_to_log_onehot(ligand_fc_bond_type, self.num_bond_classes)
            ligand_b_perturbed, log_ligand_bt = self.bond_type_trans.q_v_sample(
                log_ligand_b0, time_step, batch_ligand_bond)
            ligand_fc_bond_type_ = ligand_fc_bond_type.clone()
            ligand_fc_bond_type_[mask_edge_mask] = ligand_b_perturbed[mask_edge_mask]
        else:
            ligand_b_perturbed = None

        pos_traj, v_traj, bond_traj = [], [], []
        v0_pred_traj, vt_pred_traj = [], []
        bt_pred_traj = []
        ligand_pos, ligand_v, ligand_bond = ligand_pos_, ligand_v_, ligand_fc_bond_type_

        # time sequence
        time_seq = list(reversed(range(0, time_step[0].item())))

        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)

            preds, v_grad, b_grad, pos_grad = self.aff_guide(
                protein_pos=protein_pos,
                protein_v=protein_v,
                batch_protein=batch_protein,
                init_ligand_pos_=ligand_pos,
                init_ligand_v=ligand_v,
                init_ligand_v_aux=ligand_v_aux,
                batch_ligand=batch_ligand,
                ligand_fc_bond_index=ligand_fc_bond_index,
                init_ligand_fc_bond_type=ligand_bond,
                time_step=t,
                mask_mask = mask_mask,
                anchor_feature = anchor_feature,
            )

            # Compute posterior mean and variance
            if self.model_mean_type == 'noise':
                pred_pos_noise = preds['pred_ligand_pos'][mask_mask] - ligand_pos[mask_mask]
                pos0_from_e = self._predict_x0_from_eps(xt=ligand_pos[mask_mask], eps=pred_pos_noise, t=t, batch=ligand_mask_mask_b_batch)
                v0_from_e = preds['pred_ligand_v'][mask_mask]
            elif self.model_mean_type == 'C0':
                pos0_from_e = preds['pred_ligand_pos'][mask_mask]
                v0_from_e = preds['pred_ligand_v'][mask_mask]
            else:
                raise ValueError

            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos[mask_mask], t=t, batch=ligand_mask_mask_b_batch)
            pos_log_variance = extract(self.posterior_logvar, t, ligand_mask_mask_b_batch)
            # no noise when t == 0
            nonzero_mask = (1 - (t == 0).float())[ligand_mask_mask_b_batch].unsqueeze(-1)

            log_ligand_v_recon = F.log_softmax(v0_from_e, dim=-1)
            log_ligand_v = index_to_log_onehot(ligand_v[mask_mask], self.num_classes)

            log_ligand_b_recon = F.log_softmax(preds['pred_bond'][mask_edge_mask], dim=-1)
            log_ligand_b = index_to_log_onehot(ligand_bond[mask_edge_mask], self.num_bond_classes)

            # affinity guidance
            v_grad = v_grad[mask_mask]
            b_grad = b_grad[mask_edge_mask]
            pos_grad = pos_grad[mask_mask]
            v_grad = v_grad * gui_scale_node
            b_grad = b_grad * gui_scale_bond
            pos_grad = pos_grad * gui_scale_pos
            pos_model_mean += (0.5 * pos_log_variance).exp() * pos_grad
            log_ligand_v += v_grad
            log_ligand_b += b_grad

            log_model_prob = self.atom_type_trans.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, ligand_mask_mask_b_batch)
            ligand_v_next = log_sample_categorical(log_model_prob)

            v0_pred_traj.append(log_ligand_v_recon.clone().cpu())
            vt_pred_traj.append(log_model_prob.clone().cpu())

            # bond
            log_b_model_prob = self.bond_type_trans.q_v_posterior(
                log_ligand_b_recon, log_ligand_b, t, ligand_mask_edge_mask_b_batch)
            ligand_b_next = log_sample_categorical(log_b_model_prob)
            ligand_fc_bond_type_[mask_edge_mask] = ligand_b_next
            ligand_bond = ligand_fc_bond_type_
            bt_pred_traj.append(log_b_model_prob.clone().cpu())
            bond_traj.append(ligand_bond.clone().cpu())

            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(
                ligand_pos[mask_mask])

            ligand_pos_[mask_mask] = ligand_pos_next
            ligand_pos = ligand_pos_

            ligand_v_[mask_mask] = ligand_v_next
            ligand_v = ligand_v_

            ori_ligand_pos = ligand_pos + offset[batch_ligand]
            pos_traj.append(ori_ligand_pos.clone().cpu())
            v_traj.append(ligand_v.clone().cpu())

        ligand_pos = ligand_pos + offset[batch_ligand]

        return {
            'pos': ligand_pos,
            'v': ligand_v,
            'bond': ligand_bond,
            'bond_traj': bond_traj,
            'bt_traj': bt_pred_traj,

            'pos_traj': pos_traj,
            'v_traj': v_traj,
            'v0_traj': v0_pred_traj,
            'vt_traj': vt_pred_traj
        }
        