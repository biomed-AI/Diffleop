# Copyright 2023 ByteDance and/or its affiliates.
# SPDX-License-Identifier: CC-BY-NC-4.0

import sys
sys.path.append('.')
import argparse
import os
import shutil

import numpy as np
import torch
import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets.pl_data import FOLLOW_BATCH
from datasets.pl_pair_dataset_affinity import get_dataset_linker_aff, get_dataset_dec_aff
from models.diffleop import Diffleop

torch.multiprocessing.set_sharing_strategy('file_system')


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            'basic': trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            'add_aromatic': trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            'full': trans.MAP_INDEX_TO_ATOM_TYPE_FULL
        }
        print(f'atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


def get_bond_auroc(y_true, y_pred):
    avg_auroc = 0.
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        bond_type = {
            0: 'none',
            1: 'single',
            2: 'double',
            3: 'triple',
            4: 'aromatic',
        }
        print(f'bond: {bond_type[c]} \t auc roc: {auroc:.4f}')
    return avg_auroc / len(y_true)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--type', type=str, default='dec')
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)

    # Logging
    log_dir = misc.get_new_log_dir(args.logdir, prefix=config_name, tag=args.tag)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    indicator = trans.AddIndicator()
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        indicator
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    if getattr(config.model, 'bond_diffusion', False):
        transform_list.append(
            trans.FeaturizeLigandBond(mode=config.data.transform.ligand_bond_mode, set_bond_type=True)
        )
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    if args.type == 'linker':
        train, test = get_dataset_linker_aff(
            config=config.data,
            transform=transform,
        )
    elif args.type == 'dec':
        train, test = get_dataset_dec_aff(
            config=config.data,
            transform=transform,
        )
    train_set, val_set = train, test
    logger.info(f'Training: {len(train_set)} Validation: {len(val_set)}')

    collate_exclude_keys = ['ligand_nbh_list']
    train_iterator = utils_train.inf_iterator(DataLoader(
        train_set,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
        pin_memory = True,
    ))
    val_loader = DataLoader(val_set, config.train.batch_size, shuffle=False,
                            follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys)

    # Model
    logger.info('Building model...')
    protein_feature_dim = sum([getattr(t, 'protein_feature_dim', 0) for t in transform_list])
    ligand_feature_dim = sum([getattr(t, 'ligand_feature_dim', 0) for t in transform_list])
    ligand_feature_dim += 1 # anchor feature

    model = Diffleop(
        config.model,
        protein_atom_feature_dim=protein_feature_dim,
        ligand_atom_feature_dim=ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob, prior_bond_types=ligand_featurizer.bond_types_prob
    ).to(args.device)
    print(f'protein feature dim: {protein_feature_dim} '
          f'ligand feature dim: {ligand_feature_dim} ')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')
    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    if config.train.use_load:
        print('load')
        ckpt = torch.load(config.train.ckpt, map_location=args.device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    def train(it):
        model.train()
        optimizer.zero_grad()
        try:
            for _ in range(config.train.n_acc_batch):
                batch = next(train_iterator).to(args.device)
                protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                gt_protein_pos = batch.protein_pos + protein_noise

                results = model.get_diffusion_loss(
                    protein_pos=gt_protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,

                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    ligand_v_aux=batch.ligand_atom_aux_feature.float(),
                    batch_ligand=batch.ligand_element_batch,

                    ligand_fc_bond_index=getattr(batch, 'ligand_fc_bond_index', None),
                    ligand_fc_bond_type=getattr(batch, 'ligand_fc_bond_type', None),

                    anchor_pos = batch.ligand_anchor_pos,
                    mask_mask = batch.ligand_mask_mask,
                    mask_edge_mask = getattr(batch, 'ligand_mask_edge_mask', None),
                    ligand_mask_mask_b_batch = getattr(batch, 'ligand_mask_mask_b_batch', None), 
                    ligand_mask_edge_mask_b_batch = getattr(batch, 'ligand_mask_edge_mask_b_batch', None), 

                    anchor_feature = batch.ligand_anchor_feature,
                    true_affinity = batch.affinity,
                )
                loss_dict = results['losses']
                loss = utils_train.sum_weighted_losses(loss_dict, config.train.loss_weights)
                loss_dict['overall'] = loss

                loss = loss / config.train.n_acc_batch
                loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            if it % args.train_report_iter == 0:
                utils_train.log_losses(loss_dict, it, 'train', args.train_report_iter, logger, writer, others={
                    'grad': orig_grad_norm,
                    'lr': optimizer.param_groups[0]['lr']
                })
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
            else:
                raise e

    def validate(it):
        loss_tape = utils_train.ValidationLossTape()
        # fix time steps
        all_pred_v, all_true_v = [], []
        all_pred_bond_type, all_gt_bond_type = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                batch = batch.to(args.device)
                batch_size = batch.num_graphs
                t_loss, t_loss_pos, t_loss_v = [], [], []
                for t in np.linspace(0, model.num_timesteps - 1, 10).astype(int):
                    time_step = torch.tensor([t] * batch_size).to(args.device)
                    results = model.get_diffusion_loss(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch.protein_element_batch,

                        ligand_pos=batch.ligand_pos,
                        ligand_v=batch.ligand_atom_feature_full,
                        ligand_v_aux=batch.ligand_atom_aux_feature.float(),
                        batch_ligand=batch.ligand_element_batch,

                        ligand_fc_bond_index=getattr(batch, 'ligand_fc_bond_index', None),
                        ligand_fc_bond_type=getattr(batch, 'ligand_fc_bond_type', None),

                        time_step=time_step,


                        anchor_pos = batch.ligand_anchor_pos,
                        mask_mask = batch.ligand_mask_mask,
                        mask_edge_mask = getattr(batch, 'ligand_mask_edge_mask', None), # batch.ligand_mask_edge_mask,
                        ligand_mask_mask_b_batch = getattr(batch, 'ligand_mask_mask_b_batch', None), # batch.ligand_mask_mask_b_batch,
                        ligand_mask_edge_mask_b_batch = getattr(batch, 'ligand_mask_edge_mask_b_batch', None), # batch.ligand_mask_edge_mask_b_batch,

                        anchor_feature = batch.ligand_anchor_feature,
                        true_affinity = batch.affinity,
                    )
                    loss_dict = results['losses']
                    loss = utils_train.sum_weighted_losses(loss_dict, config.train.loss_weights)
                    loss_dict['overall'] = loss
                    loss_tape.update(loss_dict, 1)

                    all_pred_v.append(results['ligand_v_recon'].detach().cpu().numpy())
                    all_true_v.append(batch.ligand_atom_feature_full[batch.ligand_mask_mask].detach().cpu().numpy())
                    if model.bond_diffusion:
                        all_pred_bond_type.append(results['ligand_b_recon'].detach().cpu().numpy())
                        if getattr(batch, 'ligand_bond_mask', None) is not None:
                            gt_bond_type = batch.ligand_fc_bond_type[batch.ligand_bond_mask]
                        else:
                            gt_bond_type = batch.ligand_fc_bond_type
                        all_gt_bond_type.append(gt_bond_type[batch.ligand_mask_edge_mask].detach().cpu().numpy())

        atom_auroc = get_auroc(np.concatenate(all_true_v), np.concatenate(all_pred_v, axis=0),
                               feat_mode=config.data.transform.ligand_atom_mode)
        if model.bond_diffusion:
            bond_auroc = get_bond_auroc(np.concatenate(all_gt_bond_type), np.concatenate(all_pred_bond_type, axis=0))
        else:
            bond_auroc = 0.
        avg_loss = loss_tape.log(it, logger, writer, 'val', others={'Atom aucroc': atom_auroc, 'Bond aucroc': bond_auroc})

        # Trigger scheduler
        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                val_loss = validate(it)
                if best_loss is None or val_loss < best_loss:
                    logger.info(f'[Validate] Best val loss achieved: {val_loss:.6f}')
                    best_loss, best_iter = val_loss, it
                    ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                else:
                    logger.info(f'[Validate] Val loss is not improved. '
                                f'Best val loss: {best_loss:.6f} at iter {best_iter}')
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
