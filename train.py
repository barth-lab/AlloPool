import torch
import numpy as np
from einops import rearrange, repeat
import argparse
from argparse import ArgumentParser, _ArgumentGroup, Namespace
from arg_parser import *
import pickle
from os import path
import os
from torch.utils.tensorboard import SummaryWriter
from torch import Tensor
import io
from typing import *
import sys
from enum import Enum
from collections import UserDict
import random
import json  # for logging

# import pygit2
from datetime import datetime

# from torchdistill.core.forward_hook import ForwardHookManager
from Data.Data_utils import *
from Modules.Model import EdgePoolModel

import torch.nn.functional as F
import torch.nn as nn

epsilon = 1e-8
# todo : implement arbitrary offset between btaches
# todo : optional normalization


class Logger:
    """Objects to record and save relevant data"""

    LOG_PATH = "results/"

    def __init__(self, name) -> None:
        self.name = name
        self.dir = os.path.join(self.LOG_PATH, self.name)
        if not os.path.exists(self.dir):
            os.mkdir(
                self.dir,
            )
        self.train_loss: List[Tensor] = []
        # self.valid_loss: List[float] = []
        self.epoch = 0
        self.att_map: Tensor = None
        self.edge_index: Tensor = None
        self.edge_class: Tensor = None
        self.loss_by_res: Tensor = None  # (T, N)
        self.batch_i: List[Tensor] = []  # batch indices
        self.repl_i: List[Tensor] = []  # replicate indices
        self.edge_ref: List[
            Tensor
        ] = []  # list of reference edges by batch, e.g. persistence edges

    def pjoin(self, path: str):
        return os.path.join(self.dir, path)

    def log(self):
        (f"Logging at epoch {self.epoch}")
        torch.save(self.att_map, os.path.join(self.dir, f"att_map.pt"))
        torch.save(self.train_loss, os.path.join(self.dir, "train_loss.pt"))
        torch.save(self.repl_i, os.path.join(self.dir, "repl_i.pt"))
        torch.save(self.batch_i, os.path.join(self.dir, "batch_i.pt"))
        torch.save(self.loss_by_res, self.pjoin("loss_res.pt"))
        if self.edge_ref:
            torch.save(self.edge_ref, self.pjoin("edge_ref.pt"))
        if self.edge_index is not None:
            torch.save(self.edge_index, os.path.join(self.dir, f"att_map_edges.pt"))
        if self.edge_class is not None:
            torch.save(self.edge_class, os.path.join(self.dir, f"edge_class.pt"))
        ser = json.dumps({"name": self.name, "epoch": self.epoch})
        with open(os.path.join(self.dir, "meta.json"), "w") as w_:
            w_.write(ser)

    def log_args(self, args, parser: argparse.ArgumentParser):
        p = os.path.join(self.dir, "args.json")
        print(f"Logging args at {p}")
        ser = json.dumps(vars(args), indent="\t", sort_keys=True)
        with open(p, "w") as writer:
            writer.write(ser)


def get_grad(module: nn.Module):
    """Return average gradient of module"""
    grad = 0
    n = 1
    for i, p in enumerate(module.parameters()):
        if p.grad is None:
            continue
        grad = grad + p.grad.norm().cpu().detach()
        n += 1
    return grad / n


def get_model(args, device="cpu", checkpoint=None):
    input_dim = 6  # + int(args.rot_embedding)*args.rot_embed_dim
    if not args.drop_cbs:
        input_dim += 3
    if not args.drop_angles:
        print("Not using angles")
        input_dim += 4

    print("input dim: ", input_dim)

    if args.checkpoint is not None:  # args.checkpoint is None, '', or a checkpoint name
        if not args.checkpoint:  # args.use checkpoint is empty string : use run name
            args.checkpoint = args.run_name
        model_path = path.join("checkpoints", args.checkpoint, "model.pkl")
        model = torch.load(model_path, map_location=device)
        print("Model has been loaded from checkpoint", model_path)

    else:
        print("Initializing model")
        model = EdgePoolModel(
            seq_len=args.seq_len,  # Total sequence len
            input_dim=args.input_dim,
            num_time_steps=args.num_time_steps,  # Number of prediction to perform
            temporal_in=args.temporal_in,
            temporal_out=args.temporal_out,
            pool_in=args.pool_in,
            pool_ratio=args.pool_ratio,
            pool_attn_heads=args.pool_attn_heads,
            pool_depth=args.pool_depth,
            transformer_depth=args.transformer_depth,
            transformer_K=args.transformer_heads,
            transformer_nb_cheb_filter=args.decoder_dim,
            transformer_time_cheb_filter=args.decoder_dim,
            len_input=args.batch_size - args.num_time_steps,  # History
        )
    return model


def get_dataset(
    sim_type,
    data_dir,
    use_cache=True,
    cache_name="",
    rotary_embed=None,
    replicates=1,
    timesteps=20,
    stride=1,
    threshold=1.5,
    first_frame=None,
    last_frame=None,
    batch_size=20,
    overlap=None,
    shuffle=True,
    persistence=0.5,
    normalize=False,
    test_set=False,
    test_n=1,
    drop_angles=False,
    drop_cbs=False,
    test_only=False,
    args=None,
) -> Tuple[PersistenceLoader, Normalizer, None | PersistenceLoader]:
    """
    Either sim_type or data_dir should be None.
    If test set, returns last replicate as a test dataloader
    """
    assert (
        (sim_type is None) ^ (data_dir is None)
    ) or use_cache  # sim_type xor data_dir is None
    cache_dir = path.join(
        "cache",
        cache_name,
    )
    cache_path = path.join(cache_dir, "data_preprocessed.pkl")
    if use_cache:
        print("Looking for cached data at", cache_dir)
        if path.isfile(cache_path):
            print("Loading data from ", cache_path)
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            with open(path.join(cache_dir, "replicates.pkl"), "rb") as reader:
                repl_len = pickle.load(reader)
        else:
            raise RuntimeError(f"There is no cache named {cache_name}")
    else:
        if sim_type == "smd":
            sim_path = "traj_smd"
        elif sim_type == "eq":
            sim_path = "traj"
            if first_frame is not None:
                print(f"Warning : first_frame is {first_frame} on eq simulation")
        elif sim_type is None:
            sim_path = data_dir
        print(f"Extracting data from simulations at {sim_path}")
        if os.path.exists(cache_path):
            print(f"\nWARNING : CACHE ALREADY EXISTS AT {cache_path}\n")
        frames, frame_idx, repl_len, data_files = apply_to_all_simulation(
            sim_path,
            replicates,
            timesteps,
            first_frame,
            last_frame,
            use_displacement=True,
            use_dihedral=not drop_angles,
            use_cbs=not drop_cbs,
        )
        coords, edges, weights = get_graph(frames, threshold=threshold)
        repl_idx = []
        for i, n in enumerate(repl_len):
            repl_idx.extend([i] * n)
        data = make_dataset(
            coords, edges, weights, frame_idx=frame_idx, repl_idx=repl_idx
        )

        if rotary_embed is not None:  # add position embedding
            len_seq = int(data[0].x.shape[0])
            seq_pos_embed = rotary_embed(torch.arange(len_seq))
            print("seq pos embed", seq_pos_embed)
            for d in data:
                print("test", d.x.shape)
                d.x = torch.cat([d.x, seq_pos_embed], dim=-1)

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, "wb") as f:
            print("Saving data to cache at ", cache_path)
            pickle.dump(data, f)
        with open(path.join(cache_dir, "args.json"), "w") as writer:
            json.dump(vars(args), writer, indent="\t")
        with open(path.join(cache_dir, "replicates.pkl"), "wb") as writer:
            pickle.dump(repl_len, writer)
        with open(path.join(cache_dir, "data_src.txt"), "w") as writer:
            writer.writelines(
                [t + "\n" for t in data_files],
            )

    if test_set:
        assert test_n <= len(
            repl_len
        ), f"test_n = {test_n} > replicates_n = {len(repl_len)}"
        repl_test = repl_len[-test_n:]
        repl_len = repl_len[:-test_n]
        n_train = sum(repl_len)
        data_test = data[n_train:]
        data = data[:n_train]

    normalizer = Normalizer(data, blank=not normalize)

    # if normalize:
    # for d in data:
    # normalizer.normalize(d) #in-place operation
    loader = PersistenceLoader(
        data,
        repl_len,
        batch_size=batch_size,
        offset=overlap,
        shuffle=shuffle,
        threshold=persistence,
        stride=stride,
    )
    print("train set")
    print("Number of frames ", len(data))
    print(f"Average number of edges {loader.avg_edges:.0f}")
    print("Number of batches", len(loader))
    if test_set:

        if normalize:
            for d in data_test:
                normalizer.normalize(d)  # in-place operation
        print("\ntest set")
        print("Number of frames ", len(data_test))
        print(
            f"Average number of edges {np.mean([d.edge_index.shape[1] for d in data_test]):.0f}"
        )
        test_only = test_only or (
            len(data_test) < 1000
        )  # if dataset is small use all test set
        loader_test = PersistenceLoader(
            data_test,
            repl_test,
            batch_size=batch_size,
            stride=stride,
            offset=max(
                overlap, 1 if test_only else 20
            ),  # don't spend too much time on test set
            shuffle=False,
            threshold=persistence,
        )
        print("Number of batches", len(loader_test))
    else:
        loader_test = None

    return (
        loader,
        normalizer,
        loader_test,
    )


def main(args, logger: Logger, data_args, device="cpu"):
    # Create path for model checkpoints and tensorboard folder
    os.makedirs(path.join("checkpoints", args.run_name), exist_ok=True)
    tsb_path = path.join("runs", args.run_name)
    if (
        os.path.isdir(tsb_path) and args.checkpoint is None
    ):  # empty directory of previous runs if necessary
        i = 0
        for file in glob(path.join(tsb_path, "events.*")):
            i += 1
            os.remove(file)
        print(f"Emptying Tensorboard logdir of {i} events files.")
        os.rmdir(tsb_path)
    writer = SummaryWriter(log_dir=tsb_path)

    print("Preparing dataset...")
    dataloader, normalizer, test_dataloader = get_dataset(
        args.sim_type,
        args.dir,
        use_cache=not args.no_cache,
        rotary_embed=None,
        replicates=args.replicates,
        timesteps=args.timesteps,
        threshold=args.threshold,
        first_frame=args.first_frame,
        last_frame=args.last_frame,
        batch_size=args.batch_size,
        overlap=args.overlap,
        stride=args.stride,
        shuffle=args.shuffle,
        persistence=args.persistence,
        normalize=args.normalize,
        cache_name=args.cache_name,
        drop_angles=args.drop_angles,
        drop_cbs=args.drop_cbs,
        test_set=True,
        args=data_args,
    )
    print("\nGetting model ready...", end=" ")
    model = get_model(args, device)
    with open(f"results/{args.run_name}/model.txt", "w") as w_:
        print(model, file=w_)
    print(f"{sum(t.numel() for t in model.parameters()):_} parameters in model")
    print("Model is ready. See description at ", f"results/{args.run_name}/model.txt")
    if args.cuda:
        model = model.to(device)
    print("Using ", device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        # momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min',factor=0.5, patience=500,
    #     min_lr= args.lr if args.cst_lr else 0
    #     )

    print("\nStart Training")
    best_epoch_loss = 15000

    # TEMP : to remove later
    if not hasattr(model, "counter"):
        model.counter = 0
        model.counter_test = 0

    counter = model.counter
    test_counter = model.counter_test

    nanflag = False
    start_epoch = 0
    progress = tqdm(range(start_epoch, args.n_epochs))

    for epoch_i in progress:
        progress.set_postfix({"wall": datetime.now().strftime("%d/%m %H:%M")})
        logger.epoch += 1
        av_loss_epoch = []
        batch_retrieve = []
        out_attn = []
        out_attn_e = []
        out_edge_class = []
        batch_index = []
        repl_index = []
        frac_edges = []
        num_edges = []
        grad_struct = []
        grad_conv = []
        grad_temp = []
        grad_decoder = []
        masked_loss_total = []
        all_pair_attn = []
        edge_ref = []  # persistence edges over epoch
        acc_loss = 0

        if not args.test_only:  # train loop
            model.train()
            for batch_i, batch in tqdm(
                enumerate(dataloader), "Train loop", len(dataloader)
            ):
                if args.cuda:
                    batch = batch.to(device)

                batch_size = len(batch.y)
                batch_id = batch.y
                batch_retrieve.append(batch_id.detach().cpu().numpy())
                attn_e = batch.persistence_edges

                # Run Model forward pass
                all_loss, pred_coords, edge_class, attn_w, pair_attn = model(batch)

                if epoch_i == batch_i == 0:

                    num_edge_tot = batch.persistence_edges.size(1)

                num_edge_real = edge_class.size(1)
                frac_edges.append((num_edge_real / num_edge_tot))
                num_edges.append((num_edge_real))

                loss = (
                    all_loss["CA_bond_loss"] * args.rmsd_coef
                    + all_loss["rmsd_loss"] * args.transl_coef
                    + all_loss["contact_loss"] * args.contact_coef
                )
                # loss = all_loss["loss_x"] + all_loss["loss_y"] + all_loss["loss_z"]
                for key, loss_value in all_loss.items():
                    writer.add_scalar(f"train/loss/{key}", loss_value, counter)

                acc_loss += loss
                av_loss_epoch.append(loss)

                writer.flush()
                counter += 1

                edge_ref.append(batch.persistence_edges.cpu())

                loss = loss / args.accumulation_step
                loss.backward()

                if torch.isnan(loss):
                    nanflag = True
                if ((batch_i + 1) % args.accumulation_step == 0) or (
                    (batch_i + 1) == len(dataloader)
                ):
                    #  torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                    optimizer.step()
                    optimizer.zero_grad()
                    # scheduler.step(loss) #this is unorthodox but we don't do enough epochs to put it outside the loop

                batch_index.append(batch.y[0].cpu())
                repl_index.append(batch.repl[0].cpu())
                out_attn.append(attn_w.detach().cpu().numpy())
                out_attn_e.append(attn_e.detach().cpu().numpy())
                out_edge_class.append(edge_class.detach().cpu())
                # all_pair_attn.append(pair_attn.detach().cpu().numpy())

            train_epoch_loss = torch.mean(torch.stack(av_loss_epoch), dim=0)
            for param_group in optimizer.param_groups:
                lr_ = param_group["lr"]

            # region train loop logging

            writer.add_scalar("train/Epoch_loss", train_epoch_loss, epoch_i)

            logger.batch_i = batch_index
            logger.repl_i = repl_index
            # logger.loss_by_res = loss_unreduced_l.cpu().detach() # not useful to keep
            logger.edge_ref = edge_ref
        # endregion
        # test loop
        if test_dataloader is not None:
            epoch_loss = []
            model.eval()
            with torch.no_grad():
                record_act = DefaultDict(list)
                for batch_i, batch in tqdm(
                    enumerate(test_dataloader),
                    desc="Test loop",
                    total=len(test_dataloader),
                ):
                    if args.cuda:
                        batch = batch.to(device)
                    save_act: Dict[Tensor] = {}

                    batch_size = len(batch.y)
                    batch_id = batch.y
                    batch_retrieve.append(batch_id.detach().cpu().numpy())
                    attn_e = batch.persistence_edges

                    # Run Model forward pass
                    all_loss, pred_coords, edge_class, attn_w, pair_attn = model(batch)

                    loss = (
                        all_loss["rmsd_loss"] * args.rmsd_coef
                        + all_loss["translation_loss"] * args.transl_coef
                        + all_loss["contact_loss"] * args.contact_coef
                    )

                    epoch_loss.append(loss)

                    for key, loss_value in all_loss.items():
                        writer.add_scalar(f"test/loss/{key}", loss_value, test_counter)

                    test_counter += 1

        epoch_loss = torch.stack(epoch_loss).mean()

        if train_epoch_loss <= best_epoch_loss:  #
            print()
            savepath = path.join("checkpoints", args.run_name, "model.pkl")
            print("Saving model in ", savepath)
            model.last_epoch = epoch_i
            best_epoch_loss = train_epoch_loss
            logger.att_map = out_attn
            logger.edge_class = out_edge_class
            np.save("batch_index.npy", np.concatenate(batch_retrieve))
            # np.save(f'results/{args.run_name}/pair_attention.npy', np.stack(all_pair_attn,axis=0).mean(axis=0))
            torch.save(model.state_dict(), savepath)
            print("Saving edges ")
            logger.log()
        if nanflag:
            print("Loss is NaN, stopping.")


if __name__ == "__main__":
    # print("Using branch ", pygit2.Repository('.').head.shorthand )
    # Create parsers
    parser = argparse.ArgumentParser(
        epilog="Will use passed arguments, then arguments loaded from load-args, then default arguments."
    )
    parser.add_argument(
        "--load-args",
        default=None,
        nargs="*",
        help="If passed, ignore all other arguments except run-name and use arguments from run-name or passed path or run name",
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="If passed, create and save cache"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        nargs="?",
        const="",
        help="Use saved model. \
        If no name is passed, use checkpoint with name identical to run name",
        metavar="model",
    )

    dataloder_parser = make_dataloader_parser()
    dataset_parser = make_dataset_parser()
    model_parser = make_model_parser()
    training_parser = make_training_parser()
    main_args_parser = ArgumentParser(
        parents=[
            dataloder_parser,
            model_parser,
            training_parser,
        ],
        conflict_handler="resolve",
    )
    parser = ArgumentParser(
        parents=[
            dataloder_parser,
            dataset_parser,
            model_parser,
            training_parser,
            parser,
        ],
        conflict_handler="resolve",
    )
    # parse once to get all arguments, raise errors and print help if needed
    args_l = sys.argv[1:]  # Not sure whether parse_args() consumes args
    args_default = parser.parse_args(args_l)  # args with defaultvalues
    fill_parser_default(parser, args_default)  # replaces all defaults with  flag

    # Re-parse args, with flag values instead of default
    dataset_args, _ = dataset_parser.parse_known_args(args_l)
    main_args, _ = main_args_parser.parse_known_args(args_l)
    all_args = parser.parse_args(args_l)
    # initialize argdict, take passed args
    argdict = ArgDict()
    argdict.init_with_args(
        all_args,
    )  # initialize args with flag where nothing was passed
    argdict.init_with_args(
        main_args,
        tags={"main"},
    )
    argdict.init_with_args(
        dataset_args,
        tags={"data"},
    )
    # if needed, load saved args
    load_args = args_default.load_args
    run_name = args_default.run_name
    if load_args is not None:  # loading args from file
        if not load_args:
            path_or_run_name = [run_name]
        else:
            # new_run_name = run_name
            path_or_run_name = load_args
        for arg_file in path_or_run_name:
            if os.path.isfile(arg_file):
                p = arg_file
            else:
                p = os.path.join("results", arg_file, "args.json")
            print(f"Using args as saved at {p}")
            with open(p, "r") as reader:
                new_args = json.load(reader)
            argdict.update_default(new_args)
    # finally, update with default values

    argdict.update_default(vars(args_default))
    args = argdict.get_namespace()
    dataset_args = argdict.get_namespace("data")

    ## args checks

    if args.seed is not None:
        torch.random.manual_seed(args.seed)

    if args.drop_x and args.decoder != "mlp":
        raise NotImplementedError("Dropping x is implemented for mlp decoder only")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.shuffle = not args.no_shuffle
    if args.cuda:
        device = "cuda"
    else:
        device = "cpu"
    logger = Logger(args.run_name)
    print(f"Logging to {logger.dir}")
    logger.log_args(argdict.get_namespace("main"), parser)
    main(args, logger, data_args=dataset_args, device=device)
