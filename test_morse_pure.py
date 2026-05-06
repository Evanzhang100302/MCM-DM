import sys, os, argparse, math, pickle, random
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--dataset",              type=str,  default="Earthquake")
p.add_argument("--zoom",                 type=int,  default=12)
p.add_argument("--timesteps",            type=int,  default=500)
p.add_argument("--samplingsteps",        type=int,  default=500)
p.add_argument("--batch_size",           type=int,  default=64)
p.add_argument("--cuda_id",             type=str,  default="0")
p.add_argument("--model_ckpt",           type=str,  default="")
p.add_argument("--morse_ckpt",           type=str,  default="")
p.add_argument("--gate_ckpt",            type=str,  default="")
p.add_argument("--temporal_morse_ckpt",  type=str,  default="")
p.add_argument("--emb_file",             type=str,  default="")
p.add_argument("--morse_file",           type=str,  default="")
p.add_argument("--n_samples",            type=int,  default=3)
p.add_argument("--seed",                 type=int,  default=42)
p.add_argument("--img_dim",              type=int,  default=64)
p.add_argument("--no_vlm",              action="store_true")
p.add_argument("--no_morse",            action="store_true")
p.add_argument("--no_temporal_morse",   action="store_true")
p.add_argument("--sigma_t",             type=float, default=0.05)
p.add_argument("--morse_k_t",           type=int,   default=5)
p.add_argument("--top_n_t",             type=int,   default=3)
args = p.parse_args()

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from transformer_st import Transformer_ST, get_dataloader
from model_mm import (Model_all_MM, GaussianDiffusion_MM, ST_Diffusion_MM,
                      ImageProjector, MorseTransformer, TriModalFusionGate, TemporalMorseEncoder)

# ── helpers (mirrors train_mm_morse_temporal.py) ──────────────────────────────

def normalization(x, MAX, MIN): return (x - MIN) / (MAX - MIN)

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def load_data():
    data_root = f"dataset/{args.dataset}"
    def read(split):
        with open(f"{data_root}/{split}.pkl", "rb") as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        d = [[[i[0], i[0]-u[idx-1][0] if idx>0 else i[0]] + i[1:]
              for idx, i in enumerate(u)] for u in d]
        return d
    train_data = read("data_train")
    val_data   = read("data_val")
    test_data  = read("data_test")
    data_all   = train_data + val_data + test_data
    dim = 2
    Max, Min = [], []
    for m in range(dim + 2):
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1); Min.append(0)
    def norm(d):
        return [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in d]
    return (get_dataloader(norm(test_data), min(len(test_data), 1000), D=dim, shuffle=False),
            (Max, Min), data_all)

def build_emb_lookup(data_all, emb_cache):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw = float(event[2]); lat_raw = float(event[3])
            tx, ty = lat_lon_to_tile(lat_raw, lon_raw, args.zoom)
            key = (args.zoom, tx, ty)
            if key in emb_cache:
                val = emb_cache[key]
                lookup[(round(lon_raw,5), round(lat_raw,5))] = val["embedding"] if isinstance(val, dict) else val
    return lookup

def build_tile_features(morse_cache):
    tile_keys = list(morse_cache.keys())
    all_mv = [morse_cache[k]["morse_value"] for k in tile_keys]
    mv_min, mv_max = min(all_mv), max(all_mv) + 1e-6
    feats = []
    for k in tile_keys:
        mv = (morse_cache[k]["morse_value"] - mv_min) / (mv_max - mv_min)
        ic = float(morse_cache[k]["is_critical"])
        feats.append([mv, ic])
    tile_features = torch.tensor(feats, dtype=torch.float32)
    tile_key_to_idx = {k: i for i, k in enumerate(tile_keys)}
    return tile_keys, tile_features, tile_key_to_idx

def build_morse_lookup(data_all, morse_cache, tile_key_to_idx):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw = float(event[2]); lat_raw = float(event[3])
            tx, ty = lat_lon_to_tile(lat_raw, lon_raw, args.zoom)
            key = (args.zoom, tx, ty)
            if key in tile_key_to_idx:
                lookup[(round(lon_raw,5), round(lat_raw,5))] = tile_key_to_idx[key]
    return lookup

def batch_to_model(batch, transformer, emb_lookup, morse_lookup, temporal_morse_encoder, MAX, MIN):
    event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
    event_loc = torch.cat((lng.unsqueeze(2), lat.unsqueeze(2)), dim=-1)
    enc_out, mask = transformer(event_loc, event_time_origin)
    B = mask.shape[0]

    seq_crit_padded = []; seq_valid = []
    for b_idx in range(B):
        length = int(mask[b_idx].sum().item())
        if length <= 1:
            seq_crit_padded.append(None); seq_valid.append(False); continue
        vm = (event_time[b_idx] > 0).clone(); vm[0] = True
        abs_t   = event_time_origin[b_idx][vm].cpu()
        inter_t = event_time[b_idx][vm].cpu()
        if len(abs_t) < 2:
            seq_crit_padded.append(None); seq_valid.append(False); continue
        N = len(abs_t)
        # Vectorized KNN: compute full pairwise distance matrix, mask upper triangle
        dists_sq_mat = (abs_t.unsqueeze(0) - abs_t.unsqueeze(1)) ** 2  # [N, N]
        past_mask = torch.tril(torch.ones(N, N, dtype=torch.bool), diagonal=-1)
        inf_fill = torch.full((N, N), float('inf'))
        dists_masked = torch.where(past_mask, dists_sq_mat, inf_fill)  # [N, N]
        knn_k = min(args.morse_k_t, N - 1)
        top_k_dists, _ = torch.topk(dists_masked, knn_k, dim=1, largest=False)  # [N, k]
        g_t = -torch.sum(torch.exp(-top_k_dists / (2 * args.sigma_t ** 2)), dim=1)  # [N]
        g_t[0] = 0.0  # no past for first event
        g_t += torch.rand(N) * 1e-6
        top_n_actual = min(args.top_n_t, N)
        _, idx = torch.topk(g_t, top_n_actual, largest=True)
        padded = torch.zeros(args.top_n_t)
        padded[:top_n_actual] = inter_t[idx]
        seq_crit_padded.append(padded); seq_valid.append(True)

    t_emb_per_seq = [torch.zeros(args.img_dim, device=device)] * B
    if not args.no_temporal_morse:
        valid_idxs = [i for i in range(B) if seq_valid[i]]
        if valid_idxs:
            batch_crit = torch.stack([seq_crit_padded[i] for i in valid_idxs])
            batch_crit = batch_crit.unsqueeze(-1).to(device)
            batch_t_emb = temporal_morse_encoder(batch_crit)
            for j, b_idx in enumerate(valid_idxs):
                t_emb_per_seq[b_idx] = batch_t_emb[j]

    enc_out_nm, time_nm, loc_nm, img_nm, morse_idx_list, t_morse_emb_list = [], [], [], [], [], []
    zero_emb = torch.zeros(1536); zero_morse_idx = 0
    for b_idx in range(B):
        length = int(mask[b_idx].sum().item())
        if length <= 1: continue
        lng_raw = (lng[b_idx, :length].cpu() * (MAX[2]-MIN[2])) + MIN[2]
        lat_raw = (lat[b_idx, :length].cpu() * (MAX[3]-MIN[3])) + MIN[3]
        t_emb_seq = t_emb_per_seq[b_idx]
        for ev_idx in range(length - 1):
            enc_out_nm.append(enc_out[b_idx][ev_idx].unsqueeze(0))
            time_nm.append(event_time[b_idx][ev_idx+1].unsqueeze(0))
            loc_nm.append(event_loc[b_idx][ev_idx+1].unsqueeze(0))
            lon_k = round(lng_raw[ev_idx+1].item(), 5)
            lat_k = round(lat_raw[ev_idx+1].item(), 5)
            img_nm.append(emb_lookup.get((lon_k, lat_k), zero_emb).unsqueeze(0))
            morse_idx_list.append(morse_lookup.get((lon_k, lat_k), zero_morse_idx))
            t_morse_emb_list.append(t_emb_seq.unsqueeze(0))

    if len(enc_out_nm) == 0:
        return None, None, None, None, None, None
    enc_out_nm  = torch.cat(enc_out_nm,  dim=0).reshape(-1, 1, enc_out.shape[-1])
    time_nm     = torch.cat(time_nm,     dim=0).reshape(-1, 1, 1)
    loc_nm      = torch.cat(loc_nm,      dim=0).reshape(-1, 1, 2)
    img_nm      = torch.cat(img_nm,      dim=0).unsqueeze(1).to(device)
    morse_idx   = torch.tensor(morse_idx_list, dtype=torch.long).to(device)
    t_morse_emb = torch.cat(t_morse_emb_list, dim=0).unsqueeze(1)
    return time_nm, loc_nm, enc_out_nm, img_nm, morse_idx, t_morse_emb

# ── load data & embeddings ────────────────────────────────────────────────────

emb_file   = args.emb_file   or f"tile_embeddings_{args.dataset}_z{args.zoom}.pt"
morse_file = args.morse_file or f"morse_features_{args.dataset}_z{args.zoom}.pt"

testloader, (MAX, MIN), data_all = load_data()
emb_cache   = torch.load(emb_file,   map_location="cpu")
morse_cache = torch.load(morse_file, map_location="cpu")
emb_lookup  = build_emb_lookup(data_all, emb_cache)
tile_keys, tile_features, tile_key_to_idx = build_tile_features(morse_cache)
tile_features = tile_features.to(device)
morse_lookup  = build_morse_lookup(data_all, morse_cache, tile_key_to_idx)

# ── build models ──────────────────────────────────────────────────────────────

transformer = Transformer_ST(
    d_model=64, d_rnn=256, d_inner=128,
    n_layers=4, n_head=4, d_k=16, d_v=16,
    dropout=0.1, device=device, loc_dim=2, CosSin=True
).to(device)
denoiser  = ST_Diffusion_MM(n_steps=args.timesteps, dim=3, condition=True, cond_dim=64, img_dim=args.img_dim).to(device)
diffusion = GaussianDiffusion_MM(
    denoiser, seq_length=3,
    timesteps=args.timesteps, sampling_timesteps=args.samplingsteps,
    loss_type="l2", objective="pred_noise", beta_schedule="cosine"
).to(device)
img_projector          = ImageProjector(input_dim=1536, img_dim=args.img_dim).to(device)
morse_transformer      = MorseTransformer(in_dim=2, d_model=64, nhead=4, num_layers=2, out_dim=args.img_dim).to(device)
temporal_morse_encoder = TemporalMorseEncoder(top_n=args.top_n_t, d_model=64, out_dim=args.img_dim).to(device)
fusion_gate            = TriModalFusionGate(dim=args.img_dim).to(device)
Model = Model_all_MM(transformer, diffusion, img_projector).to(device)

# ── load checkpoints ──────────────────────────────────────────────────────────

model_ckpt         = args.model_ckpt
morse_ckpt         = args.morse_ckpt
gate_ckpt          = args.gate_ckpt
temporal_morse_ckpt = args.temporal_morse_ckpt

Model.load_state_dict(torch.load(model_ckpt, map_location=device))
morse_transformer.load_state_dict(torch.load(morse_ckpt, map_location=device))
fusion_gate.load_state_dict(torch.load(gate_ckpt, map_location=device))
temporal_morse_encoder.load_state_dict(torch.load(temporal_morse_ckpt, map_location=device))

Model.eval(); morse_transformer.eval(); fusion_gate.eval(); temporal_morse_encoder.eval()
print(f"Loaded model from {model_ckpt}")

# ── evaluation ────────────────────────────────────────────────────────────────

mae_spatial, mae_temporal, rmse_temporal, total_num = 0., 0., 0., 0
with torch.no_grad():
    for batch in testloader:
        t_nm, loc_nm, enc_nm, img_nm, morse_idx, t_morse_emb = batch_to_model(
            batch, Model.transformer, emb_lookup, morse_lookup, temporal_morse_encoder, MAX, MIN)
        if t_nm is None: continue

        if args.no_vlm:
            img_proj = torch.zeros(img_nm.shape[0], 1, args.img_dim, device=device)
        else:
            img_proj = Model.img_projector(img_nm)
        if args.no_morse:
            morse_emb = torch.zeros(img_proj.shape[0], 1, args.img_dim, device=device)
        else:
            morse_tile = morse_transformer(tile_features)
            morse_emb  = morse_tile[morse_idx].unsqueeze(1)
        img_proj = fusion_gate(img_proj, morse_emb, t_morse_emb)

        gen_t_all, gen_s_all = [], []
        for _ in range(args.n_samples):
            sampled = Model.diffusion.sample(batch_size=t_nm.shape[0], cond=enc_nm, img_cond=img_proj)
            gen_t_all.append(sampled[:, 0, :1].cpu())
            gen_s_all.append(sampled[:, 0, -2:].cpu())
        gen_t = torch.stack(gen_t_all, dim=0).mean(dim=0)
        gen_s = torch.stack(gen_s_all, dim=0).mean(dim=0)

        real_t = (t_nm[:, 0, :].cpu() * (MAX[1] - MIN[1])) + MIN[1]
        gen_t  = (gen_t * (MAX[1] - MIN[1])) + MIN[1]
        mae_temporal  += torch.abs(real_t - gen_t).sum().item()
        rmse_temporal += ((real_t - gen_t) ** 2).sum().item()

        real_s = (loc_nm[:, 0, :].cpu() * torch.tensor([MAX[2]-MIN[2], MAX[3]-MIN[3]])) + torch.tensor([MIN[2], MIN[3]])
        gen_s  = (gen_s * torch.tensor([MAX[2]-MIN[2], MAX[3]-MIN[3]])) + torch.tensor([MIN[2], MIN[3]])
        mae_spatial += torch.sqrt(((real_s - gen_s) ** 2).sum(dim=-1)).sum().item()
        total_num   += t_nm.shape[0]

print(f"Test events   : {total_num}")
print(f"MAE  temporal : {mae_temporal/total_num:.4f}")
print(f"RMSE temporal : {(rmse_temporal/total_num)**0.5:.4f}")
print(f"MAE  spatial  : {mae_spatial/total_num:.4f}")
