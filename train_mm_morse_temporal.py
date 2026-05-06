import os, sys, math, pickle, random, time, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
import setproctitle

from transformer_st import Transformer_ST, get_dataloader
from model_mm import ST_Diffusion_MM, GaussianDiffusion_MM, Model_all_MM, ImageProjector, MorseTransformer, TriModalFusionGate, TemporalMorseEncoder

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seed',          type=int, default=1234)
    p.add_argument('--dataset',       type=str, default='Citibike')
    p.add_argument('--total_epochs',  type=int, default=2000)
    p.add_argument('--alpha',         type=float, default=0.05)
    p.add_argument('--beta_topo',     type=float, default=0.1,   help='spatial topo loss weight')
    p.add_argument('--beta_topo_t',   type=float, default=0.001, help='temporal topo loss weight')
    p.add_argument('--sigma_t',       type=float, default=0.05,  help='temporal Morse kernel bandwidth')
    p.add_argument('--morse_k_t',     type=int,   default=5,     help='KNN neighbors for temporal Morse')
    p.add_argument('--top_n_t',       type=int,   default=3,     help='# temporal critical nodes per sequence')
    p.add_argument('--batch_size',    type=int, default=64)
    p.add_argument('--timesteps',     type=int, default=500)
    p.add_argument('--samplingsteps', type=int, default=500)
    p.add_argument('--objective',     type=str, default='pred_noise')
    p.add_argument('--loss_type',     type=str, default='l2')
    p.add_argument('--beta_schedule', type=str, default='cosine')
    p.add_argument('--cuda_id',       type=str, default='0')
    p.add_argument('--img_dim',       type=int, default=64)
    p.add_argument('--zoom',          type=int, default=15)
    p.add_argument('--emb_file',      type=str, default='')
    p.add_argument('--morse_file',    type=str, default='')
    p.add_argument('--kde_sigma_t',   type=float, default=1.0)
    p.add_argument('--no_vlm',           action='store_true', help='disable VLM image path')
    p.add_argument('--no_morse',         action='store_true', help='disable spatial Morse transformer path')
    p.add_argument('--no_temporal_morse',action='store_true', help='disable temporal Morse encoder path')
    p.add_argument('--exp_name', type=str, default='', help='experiment name for model save dir')
    p.add_argument('--kde_sigma_s',   type=float, default=1.0)
    args = p.parse_args()
    args.cuda = torch.cuda.is_available()
    args.dim  = 2
    return args

opt = get_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)
device = torch.device('cuda:0' if opt.cuda else 'cpu')

def compute_temporal_critical_nodes(times, sigma_t=0.05, k=5, top_n=3):
    N = len(times)
    if N <= 1:
        return times.clone()
    g_t = torch.zeros(N, device=times.device)
    for i in range(1, N):
        past = times[:i]
        dists_sq = (times[i] - past) ** 2
        knn_k = min(k, len(past))
        knn_dists_sq, _ = torch.topk(dists_sq, knn_k, largest=False)
        density = torch.sum(torch.exp(-knn_dists_sq / (2 * sigma_t ** 2)))
        g_t[i] = -density
    eps = torch.rand(N, device=times.device) * 1e-6
    g_t = g_t + eps
    top_n = min(top_n, N)
    _, idx = torch.topk(g_t, top_n, largest=True)
    return times[idx]

def compute_temporal_topo_loss(pred_t, t_crit_nodes):
    if t_crit_nodes is None or len(t_crit_nodes) == 0:
        return torch.tensor(0.0, device=pred_t.device)
    dists = torch.abs(pred_t.unsqueeze(1) - t_crit_nodes.unsqueeze(0).to(pred_t.device))
    min_dists = dists.min(dim=1).values
    return min_dists.mean()

def get_temporal_critical_nodes_from_batch(batch, sigma_t, k, top_n):
    """
    sigma_t computed from batch std of valid absolute times (data-adaptive).
    """
    event_time_origin = batch[0]  # [B, max_len] absolute normalized times
    event_time = batch[1]         # [B, max_len] inter-event times

    # data-adaptive sigma_t
    sigma_t = opt.sigma_t

    all_crit = []
    for b in range(event_time_origin.shape[0]):
        valid_mask = (event_time[b] > 0)
        valid_mask[0] = True
        abs_times   = event_time_origin[b][valid_mask]
        inter_times = event_time[b][valid_mask]
        if len(abs_times) < 2:
            continue

        N = len(abs_times)
        g_t = torch.zeros(N)
        for i in range(1, N):
            past = abs_times[:i]
            dists_sq = (abs_times[i] - past) ** 2
            knn_k = min(k, len(past))
            knn_dists_sq, _ = torch.topk(dists_sq, knn_k, largest=False)
            density = torch.sum(torch.exp(-knn_dists_sq / (2 * sigma_t ** 2)))
            g_t[i] = -density
        eps = torch.rand(N) * 1e-6
        g_t = g_t + eps
        top_n_actual = min(top_n, N)
        _, idx = torch.topk(g_t, top_n_actual, largest=True)
        all_crit.append(inter_times[idx])

    if len(all_crit) == 0:
        return None
    return torch.cat(all_crit, dim=0)

def mcl_loss(hist_emb, img_emb, temperature=0.1):
    h = F.normalize(hist_emb, dim=-1)
    v = F.normalize(img_emb,  dim=-1)
    sim = torch.mm(h, v.T) / temperature
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2
    return loss

def setup_init():
    random.seed(opt.seed); np.random.seed(opt.seed); torch.manual_seed(opt.seed)
    if opt.cuda: torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalization(x, MAX, MIN): return (x - MIN) / (MAX - MIN)

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def get_critical_nodes_normalized(morse_cache, MAX, MIN):
    def tile_center(z, x, y):
        n = 2**z
        lon = x / n * 360 - 180
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2*y/n))))
        return lon, lat
    nodes = []
    for k, v in morse_cache.items():
        if v['is_critical']:
            lon, lat = tile_center(*k)
            lon_nm = (lon - MIN[2]) / (MAX[2] - MIN[2])
            lat_nm = (lat - MIN[3]) / (MAX[3] - MIN[3])
            nodes.append([lon_nm, lat_nm])
    if not nodes:
        return torch.zeros(0, 2, device=device)
    return torch.tensor(nodes, dtype=torch.float32).to(device)

def load_data():
    data_root = f'dataset/{opt.dataset}'
    def read(split):
        with open(f'{data_root}/{split}.pkl', 'rb') as f:
            d = pickle.load(f)
        d = [[list(i) for i in u] for u in d]
        d = [[[i[0], i[0]-u[idx-1][0] if idx>0 else i[0]] + i[1:]
              for idx, i in enumerate(u)] for u in d]
        return d
    train_data = read('data_train')
    val_data   = read('data_val')
    test_data  = read('data_test')
    data_all   = train_data + val_data + test_data
    Max, Min = [], []
    for m in range(opt.dim + 2):
        if m > 0:
            Max.append(max(i[m] for u in data_all for i in u))
            Min.append(min(i[m] for u in data_all for i in u))
        else:
            Max.append(1); Min.append(0)
    def norm(d):
        return [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in d]
    return (get_dataloader(norm(train_data), opt.batch_size, D=opt.dim, shuffle=True),
            get_dataloader(norm(val_data),   min(len(val_data), 1000),  D=opt.dim, shuffle=False),
            get_dataloader(norm(test_data),  min(len(test_data), 1000), D=opt.dim, shuffle=False),
            (Max, Min), data_all)

def build_emb_lookup(data_all, emb_cache, zoom):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw = float(event[2])
            lat_raw = float(event[3])
            tx, ty = lat_lon_to_tile(lat_raw, lon_raw, zoom)
            key = (zoom, tx, ty)
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

def build_morse_lookup(data_all, morse_cache, zoom, tile_key_to_idx):
    lookup = {}
    for seq in data_all:
        for event in seq:
            lon_raw = float(event[2])
            lat_raw = float(event[3])
            tx, ty = lat_lon_to_tile(lat_raw, lon_raw, zoom)
            key = (zoom, tx, ty)
            if key in tile_key_to_idx:
                lookup[(round(lon_raw,5), round(lat_raw,5))] = tile_key_to_idx[key]
    return lookup

def batch_to_model(batch, transformer, emb_lookup, morse_lookup, temporal_morse_encoder, MAX, MIN):
    event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)
    event_loc = torch.cat((lng.unsqueeze(2), lat.unsqueeze(2)), dim=-1)
    enc_out, mask = transformer(event_loc, event_time_origin)
    B = mask.shape[0]

    # Compute per-sequence temporal Morse critical nodes
    seq_crit_padded = []
    seq_valid = []
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
        g_t = torch.zeros(N)
        for i in range(1, N):
            past = abs_t[:i]
            dists_sq = (abs_t[i] - past) ** 2
            knn_k = min(opt.morse_k_t, len(past))
            knn_dists_sq, _ = torch.topk(dists_sq, knn_k, largest=False)
            g_t[i] = -torch.sum(torch.exp(-knn_dists_sq / (2 * opt.sigma_t ** 2)))
        g_t += torch.rand(N) * 1e-6
        top_n_actual = min(opt.top_n_t, N)
        _, idx = torch.topk(g_t, top_n_actual, largest=True)
        padded = torch.zeros(opt.top_n_t)
        padded[:top_n_actual] = inter_t[idx]
        seq_crit_padded.append(padded); seq_valid.append(True)

    # Batch-encode temporal Morse embeddings per sequence
    t_emb_per_seq = [torch.zeros(opt.img_dim, device=device)] * B
    if not opt.no_temporal_morse:
        valid_idxs = [i for i in range(B) if seq_valid[i]]
        if valid_idxs:
            batch_crit = torch.stack([seq_crit_padded[i] for i in valid_idxs])
            batch_crit = batch_crit.unsqueeze(-1).to(device)   # [V, top_n_t, 1]
            batch_t_emb = temporal_morse_encoder(batch_crit)   # [V, img_dim]
            for j, b_idx in enumerate(valid_idxs):
                t_emb_per_seq[b_idx] = batch_t_emb[j]

    # Collect per-event data
    enc_out_nm, time_nm, loc_nm, img_nm, morse_idx_list, t_morse_emb_list = [], [], [], [], [], []
    zero_emb = torch.zeros(1536)
    zero_morse_idx = 0
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
    loc_nm      = torch.cat(loc_nm,      dim=0).reshape(-1, 1, opt.dim)
    img_nm      = torch.cat(img_nm,      dim=0).unsqueeze(1).to(device)
    morse_idx   = torch.tensor(morse_idx_list, dtype=torch.long).to(device)
    t_morse_emb = torch.cat(t_morse_emb_list, dim=0).unsqueeze(1)  # [N, 1, img_dim]
    return time_nm, loc_nm, enc_out_nm, img_nm, morse_idx, t_morse_emb

if __name__ == '__main__':
    setup_init()
    setproctitle.setproctitle('MM-STPP-Temporal-Morse')

    emb_file = opt.emb_file or f'tile_embeddings_{opt.dataset}_z{opt.zoom}.pt'
    print(f'Loading tile embeddings from {emb_file}...')
    emb_cache = torch.load(emb_file, map_location='cpu')

    morse_file = opt.morse_file or f'morse_features_{opt.dataset}_z{opt.zoom}.pt'
    print(f'Loading Morse features from {morse_file}...')
    morse_cache = torch.load(morse_file, map_location='cpu')

    trainloader, valloader, testloader, (MAX, MIN), data_all = load_data()
    emb_lookup = build_emb_lookup(data_all, emb_cache, opt.zoom)
    tile_keys, tile_features, tile_key_to_idx = build_tile_features(morse_cache)
    critical_nodes = get_critical_nodes_normalized(morse_cache, MAX, MIN)
    print(f'  Spatial critical nodes: {critical_nodes.shape[0]}')
    print(f'  Temporal Morse: sigma_t=auto(batch_std), k={opt.morse_k_t}, top_n={opt.top_n_t}, beta_t={opt.beta_topo_t}')

    tile_features = tile_features.to(device)
    morse_lookup  = build_morse_lookup(data_all, morse_cache, opt.zoom, tile_key_to_idx)

    logdir     = f'./logs_mm/{opt.dataset}_ts{opt.timesteps}_temporal'
    if opt.exp_name:
        model_path = f'./models_mm/{opt.dataset}_{opt.exp_name}_seed{opt.seed}/'
    else:
        model_path = f'./models_mm/{opt.dataset}_ts{opt.timesteps}_morse_temporal_seed{opt.seed}/'
    os.makedirs('./models_mm', exist_ok=True)
    os.makedirs(model_path,   exist_ok=True)
    writer = SummaryWriter(log_dir=logdir, flush_secs=5)

    transformer = Transformer_ST(
        d_model=64, d_rnn=256, d_inner=128,
        n_layers=4, n_head=4, d_k=16, d_v=16,
        dropout=0.1, device=device, loc_dim=opt.dim, CosSin=True
    ).to(device)

    denoiser = ST_Diffusion_MM(
        n_steps=opt.timesteps, dim=1+opt.dim,
        condition=True, cond_dim=64, img_dim=opt.img_dim
    ).to(device)

    diffusion = GaussianDiffusion_MM(
        denoiser, seq_length=1+opt.dim,
        timesteps=opt.timesteps, sampling_timesteps=opt.samplingsteps,
        loss_type=opt.loss_type, objective=opt.objective,
        beta_schedule=opt.beta_schedule
    ).to(device)

    img_projector          = ImageProjector(input_dim=1536, img_dim=opt.img_dim).to(device)
    morse_transformer      = MorseTransformer(in_dim=2, d_model=64, nhead=4, num_layers=2, out_dim=opt.img_dim).to(device)
    temporal_morse_encoder = TemporalMorseEncoder(top_n=opt.top_n_t, d_model=64, out_dim=opt.img_dim).to(device)
    fusion_gate            = TriModalFusionGate(dim=opt.img_dim).to(device)
    Model = Model_all_MM(transformer, diffusion, img_projector)
    Model.hist_proj    = Model.hist_proj.to(device)
    Model.img_proj_mcl = Model.img_proj_mcl.to(device)

    optimizer = AdamW([
        {"params": Model.parameters(),                    "lr": 1e-3},
        {"params": morse_transformer.parameters(),        "lr": 1e-4},
        {"params": fusion_gate.parameters(),              "lr": 1e-4},
        {"params": temporal_morse_encoder.parameters(),   "lr": 1e-4},
    ], betas=(0.9, 0.99))

    warmup_steps = 5
    step, early_stop, min_loss_val = 0, 0, 1e20

    for itr in range(opt.total_epochs):
        print(f'epoch: {itr}', flush=True)

        if itr % 10 == 0:
            Model.eval()
            with torch.no_grad():
                loss_val, mae_spatial, total_num = 0., 0., 0
                for batch in valloader:
                    t_nm, l_nm, enc_nm, img_nm, morse_idx, t_morse_emb = batch_to_model(
                        batch, Model.transformer, emb_lookup, morse_lookup, temporal_morse_encoder, MAX, MIN)
                    if t_nm is None: continue
                    if opt.no_vlm:
                        img_proj = torch.zeros(img_nm.shape[0], 1, opt.img_dim, device=device)
                    else:
                        img_proj = Model.img_projector(img_nm)
                    if opt.no_morse:
                        morse_emb = torch.zeros(img_proj.shape[0], 1, opt.img_dim, device=device)
                    else:
                        morse_tile = morse_transformer(tile_features)
                        morse_emb  = morse_tile[morse_idx].unsqueeze(1)
                    img_proj = fusion_gate(img_proj, morse_emb, t_morse_emb)
                    x_input = torch.cat((t_nm, l_nm), dim=-1)
                    loss = Model.diffusion(x_input, enc_nm, img_cond=img_proj)
                    loss_val += loss.item() * t_nm.shape[0]
                    sampled = Model.diffusion.sample(
                        batch_size=t_nm.shape[0], cond=enc_nm, img_cond=img_proj)
                    real = (l_nm[:,0,:2].cpu() + torch.tensor([MIN[2:]])) * \
                           (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]]))
                    gen  = (sampled[:,0,1:3].cpu() + torch.tensor([MIN[2:]])) * \
                           (torch.tensor([MAX[2:]]) - torch.tensor([MIN[2:]]))
                    mae_spatial += torch.sqrt(((real-gen)**2).sum(dim=-1)).sum().item()
                    total_num   += t_nm.shape[0]
                if total_num > 0:
                    lv = loss_val / total_num
                    ds = mae_spatial / total_num
                    print(f'  [val] loss={lv:.4f}  dist_spatial={ds:.4f}', flush=True)
                    writer.add_scalar('Eval/loss_val',         lv, itr)
                    writer.add_scalar('Eval/distance_spatial', ds, itr)
                    if loss_val > min_loss_val:
                        early_stop += 1
                        if early_stop >= 100:
                            print('Early stopping.'); break
                    else:
                        early_stop = 0
                        torch.save(Model.state_dict(),                    model_path + f'model_{itr}.pkl')
                        torch.save(morse_transformer.state_dict(),        model_path + f'morse_{itr}.pkl')
                        torch.save(fusion_gate.state_dict(),              model_path + f'gate_{itr}.pkl')
                        torch.save(temporal_morse_encoder.state_dict(),   model_path + f'temporal_morse_{itr}.pkl')
                        torch.save(Model.state_dict(),                    model_path + 'model_best.pkl')
                        torch.save(morse_transformer.state_dict(),        model_path + 'morse_best.pkl')
                        torch.save(fusion_gate.state_dict(),              model_path + 'gate_best.pkl')
                        torch.save(temporal_morse_encoder.state_dict(),   model_path + 'temporal_morse_best.pkl')
                    min_loss_val = min(min_loss_val, loss_val)

        base_lr = 1e-3*(itr+1)/warmup_steps if itr < warmup_steps \
                   else 1e-3 - (1e-3-5e-5)*(itr-warmup_steps)/opt.total_epochs
        optimizer.param_groups[0]['lr'] = base_lr
        optimizer.param_groups[1]['lr'] = base_lr / 10
        optimizer.param_groups[2]['lr'] = base_lr / 10
        optimizer.param_groups[3]['lr'] = base_lr / 10
        writer.add_scalar('Stats/lr', base_lr, itr)

        Model.train()
        loss_epoch, total_num = 0., 0
        for batch in trainloader:
            t_nm, l_nm, enc_nm, img_nm, morse_idx, t_morse_emb = batch_to_model(
                batch, Model.transformer, emb_lookup, morse_lookup, temporal_morse_encoder, MAX, MIN)
            if t_nm is None: continue

            if opt.no_vlm:
                img_proj = torch.zeros(img_nm.shape[0], 1, opt.img_dim, device=device)
            else:
                img_proj = Model.img_projector(img_nm)
            if opt.no_morse:
                morse_emb = torch.zeros(img_proj.shape[0], 1, opt.img_dim, device=device)
            else:
                morse_tile = morse_transformer(tile_features)
                morse_emb  = morse_tile[morse_idx].unsqueeze(1)
            img_proj = fusion_gate(img_proj, morse_emb, t_morse_emb)
            x_input    = torch.cat((t_nm, l_nm), dim=-1)
            loss_diff  = Model.diffusion(x_input, enc_nm, img_cond=img_proj)

            hist_for_mcl = Model.hist_proj(enc_nm[:, 0, :64].to(device))
            img_for_mcl  = Model.img_proj_mcl(img_proj[:, 0, :].to(device))
            loss_mcl = mcl_loss(hist_for_mcl, img_for_mcl)

            # Spatial topo loss
            if opt.beta_topo > 0 and len(critical_nodes) > 0:
                t_rand   = torch.randint(0, opt.timesteps, (x_input.shape[0],), device=device)
                noise    = torch.randn_like(x_input)
                x_t      = Model.diffusion.q_sample(x_input, t_rand, noise=noise)
                preds    = Model.diffusion.model_predictions(x_t, t_rand, cond=enc_nm, img_cond=img_proj)
                pred_loc = preds.pred_x_start[:, 0, -2:]
                dists    = torch.cdist(pred_loc.unsqueeze(1),
                                       critical_nodes.unsqueeze(0).expand(pred_loc.shape[0], -1, -1))
                loss_topo_s = dists.squeeze(1).min(dim=-1).values.mean()
            else:
                loss_topo_s = torch.tensor(0.0, device=device)
                preds = None

            # Temporal topo loss
            if opt.beta_topo_t > 0:
                t_crit_nodes = get_temporal_critical_nodes_from_batch(
                    batch, sigma_t=opt.sigma_t, k=opt.morse_k_t, top_n=opt.top_n_t)
                if t_crit_nodes is not None:
                    if preds is None:
                        t_rand = torch.randint(0, opt.timesteps, (x_input.shape[0],), device=device)
                        noise  = torch.randn_like(x_input)
                        x_t    = Model.diffusion.q_sample(x_input, t_rand, noise=noise)
                        preds  = Model.diffusion.model_predictions(x_t, t_rand, cond=enc_nm, img_cond=img_proj)
                    pred_t = preds.pred_x_start[:, 0, 0]
                    loss_topo_t = compute_temporal_topo_loss(pred_t, t_crit_nodes)
                else:
                    loss_topo_t = torch.tensor(0.0, device=device)
            else:
                loss_topo_t = torch.tensor(0.0, device=device)

            loss = (loss_diff
                    + opt.alpha       * loss_mcl
                    + opt.beta_topo   * loss_topo_s
                    + opt.beta_topo_t * loss_topo_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step()

            loss_epoch += loss.item() * t_nm.shape[0]
            total_num  += t_nm.shape[0]
            writer.add_scalar('Train/loss_step',   loss.item(),        step)
            writer.add_scalar('Train/loss_topo_s', loss_topo_s.item(), step)
            writer.add_scalar('Train/loss_topo_t', loss_topo_t.item(), step)
            step += 1

        writer.add_scalar('Train/loss_epoch', loss_epoch / max(total_num, 1), itr)
        if opt.cuda:
            torch.cuda.empty_cache()
