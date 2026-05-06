"""
Pre-extract Qwen2-VL embeddings for every unique OSM tile in the dataset.
Saves: tile_embeddings_{DATASET}_z{ZOOM}.pt  →  dict { (zoom, tx, ty): tensor[1536] }
Run once before training.

Usage:
    python extract_tile_embeddings.py --dataset Crime2026 --zoom 12
    python extract_tile_embeddings.py --dataset Twitter   --zoom 14
"""
import argparse, math, os, pickle, time, torch, requests
from io import BytesIO
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',   type=str, required=True)
parser.add_argument('--zoom',      type=int, required=True)
parser.add_argument('--data_root', type=str, default='dataset')
parser.add_argument('--out_file',  type=str, default='')
opt = parser.parse_args()

OUT_FILE = opt.out_file or f'tile_embeddings_{opt.dataset}_z{opt.zoom}.pt'

def lat_lon_to_tile(lat, lon, zoom):
    n = 2 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(math.radians(lat)) + 1.0 / math.cos(math.radians(lat))) / math.pi) / 2.0 * n)
    return x, y

def download_tile_img(zoom, tx, ty):
    url = f"https://tile.openstreetmap.org/{zoom}/{tx}/{ty}.png"
    r = requests.get(url, headers={"User-Agent": "STPP-Research/1.0"}, timeout=10)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")

tile_keys = set()
for split in ['data_train', 'data_val', 'data_test']:
    path = f'{opt.data_root}/{opt.dataset}/{split}.pkl'
    if not os.path.exists(path):
        continue
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for seq in data:
        for event in seq:
            lon, lat = float(event[1]), float(event[2])
            tx, ty = lat_lon_to_tile(lat, lon, opt.zoom)
            tile_keys.add((opt.zoom, tx, ty))

print(f"Unique tiles to embed: {len(tile_keys)}")

print("Loading Qwen2-VL-2B...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2-VL-2B-Instruct', torch_dtype=torch.float16, device_map="auto")
model.eval()
for p in model.parameters():
    p.requires_grad = False
processor = AutoProcessor.from_pretrained('Qwen/Qwen2-VL-2B-Instruct')
print("Model loaded.")

cache = {}
tmp_img_path = '/tmp/tile_tmp.png'
for i, (zoom, tx, ty) in enumerate(tile_keys):
    print(f"[{i+1}/{len(tile_keys)}] tile ({zoom},{tx},{ty})", flush=True)
    try:
        img = download_tile_img(zoom, tx, ty)
        img.save(tmp_img_path)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": tmp_img_path},
            {"type": "text",  "text": "Describe this map tile briefly."}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            emb = out.hidden_states[-1].mean(dim=1).squeeze(0).float().cpu()
        cache[(zoom, tx, ty)] = emb
        time.sleep(0.05)
    except Exception as e:
        print(f"  warn: {e}")

torch.save(cache, OUT_FILE)
print(f"Saved {len(cache)} embeddings → {OUT_FILE}")
