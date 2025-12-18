from sklearn.neighbors import LocalOutlierFactor
import torch
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label as cc_label

from transformers import (
    AutoProcessor,
    Owlv2TextModel,
    Owlv2Processor,
    Owlv2ForObjectDetection,
    Owlv2VisionModel,
    CLIPProcessor,
    CLIPModel
)
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import pil_to_tensor, resize, crop
from torchvision.ops import box_iou, nms

from dataset_coco import DatasetCOCO
import utils_seed
from evaluation import Evaluator
from logger import Logger, AverageMeter


@torch.no_grad()
def _get_tokens(model, pixel_values, layer_idx):
    """Extract hidden states from specified layer of vision model."""
    out = model.vision_model(pixel_values=pixel_values,
                             output_hidden_states=True, return_dict=True)
    return out.hidden_states[layer_idx]


def _resolve_anomaly(tokens, G, top_k=10):
    """Detect and resolve anomaly tokens using Local Outlier Factor. Replaces outlier tokens with 3x3 neighborhood average."""
    pt = tokens[:, 1:, :]
    B, N, D = pt.shape
    new_pt = []
    for b in range(B):
        feat = pt[b]
        lof = LocalOutlierFactor(n_neighbors=30,
                                 contamination=float(top_k)/N)
        lbl = lof.fit_predict(feat.cpu().numpy())
        out_idx = np.where(lbl == -1)[0]
        fm = feat.view(G, G, D)
        for idx in out_idx:
            r,c = divmod(idx, G)
            neigh = []
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    if dr==0 and dc==0: continue
                    rr,cc = r+dr, c+dc
                    if 0<=rr<G and 0<=cc<G:
                        neigh.append(fm[rr,cc])
            if neigh:
                fm[r,c] = torch.stack(neigh).mean(0)
        new_pt.append(fm.reshape(-1,D))
    tokens[:,1:,:] = torch.stack(new_pt)
    return tokens

def _self_adjust(deep_tok, mid_tok, beta=0.4):
    """Re-aggregate deep tokens using intermediate layer similarity."""
    deep_norm = F.normalize(deep_tok, dim=-1)
    mid_norm  = F.normalize(mid_tok,  dim=-1)
    sim = mid_norm @ mid_norm.transpose(1,2)
    sim = sim.masked_fill(sim < beta, 0.)
    sim = sim / (sim.sum(-1, keepdim=True)+1e-6)
    return sim @ deep_tok

@torch.inference_mode()
def extract_scclip_box_features(model, image_pil:Image.Image,
                                boxes:torch.Tensor,
                                *, device="cpu",
                                mid_idx=8, beta=0.4, top_k=10):
    """Extract SC-CLIP box features with anomaly resolution and self-adjustment."""
    inp = resize(image_pil, (224, 224), interpolation=Image.BICUBIC)
    px  = pil_to_tensor(inp).float() / 255.
    px  = px.unsqueeze(0).to(device)

    penult = _get_tokens(model, px, layer_idx=-2)
    last   = _get_tokens(model, px, layer_idx=-1)
    mid    = _get_tokens(model, px, layer_idx=mid_idx)

    G = int(math.sqrt(penult.size(1)-1))
    penult = _resolve_anomaly(penult.clone(), G, top_k)

    deep_tok = last[:,1:,:]
    mid_tok  = mid[:,1:,:]
    patch_tok = _self_adjust(deep_tok, mid_tok, beta).squeeze(0)

    mid_norm = F.normalize(mid_tok.squeeze(0), dim=-1)
    mid_simi = mid_norm @ mid_norm.t()
    mid_simi[mid_simi < beta] = 0.4   
    
    patch_sz = 224 // G
    feats = []
    for box in boxes.cpu():
        x0,y0,x1,y1 = box.tolist()
        sx, sy = 224/image_pil.width, 224/image_pil.height
        x0*=sx; x1*=sx; y0*=sy; y1*=sy
        cx0, cy0 = int(x0//patch_sz), int(y0//patch_sz)
        cx1, cy1 = int((x1-1)//patch_sz), int((y1-1)//patch_sz)
        ids = [r*G+c
               for r in range(cy0, cy1+1)
               for c in range(cx0, cx1+1)
               if 0<=r<G and 0<=c<G]
        if not ids:
            feats.append(torch.zeros(patch_tok.size(-1)))
            continue
        w = mid_simi[ids][:, ids].sum(1)
        f = (w[:, None] * patch_tok[ids]).sum(0) / w.sum()
        feats.append(F.normalize(f, dim=0))        
    
    return torch.stack(feats).to(device)

def clip_text_feat(text, processor, model, device):
    """Extract CLIP text features and normalize."""
    enc = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        txt = model.get_text_features(**enc)
    return F.normalize(txt.squeeze(0), dim=-1)

def scclip_box_feat(pil_img, box, scclip, clip_model, device):
    """Extract SC-CLIP box features and project to CLIP embedding space."""
    tok768 = extract_scclip_box_features(
                scclip, pil_img, box.unsqueeze(0), device=device
             ).squeeze(0)

    tok768 = tok768.clone()

    if hasattr(clip_model, "visual"):
        img512 = tok768 @ clip_model.visual.proj
    else:
        img512 = clip_model.visual_projection(tok768)

    return F.normalize(img512, dim=-1)

def scclip_box_feat_to_class(pil_img, box, scclip, clip_model, device):
    """Extract SC-CLIP box features with LayerNorm and project to CLIP embedding space."""
    tok768 = extract_scclip_box_features(
                scclip, pil_img, box.unsqueeze(0), device=device
             ).squeeze(0)

    tok768 = tok768.clone()

    if hasattr(clip_model, "visual"):
        tok768 = clip_model.visual.ln_post(tok768)
        img512 = tok768 @ clip_model.visual.proj
    else:
        tok768 = clip_model.vision_model.post_layernorm(tok768)
        img512 = clip_model.visual_projection(tok768)

    return F.normalize(img512, dim=-1)



@torch.inference_mode()
def extract_scclip_mask_features_per_object(
        model: torch.nn.Module,
        image_pil: Image.Image,
        masks: torch.Tensor,
        *,
        device="cpu",
        mid_idx=8,
        beta=0.4,
        top_k=10
    ) -> torch.Tensor:
    """Extract SC-CLIP embeddings for each connected component in binary masks."""
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    K, H, W = masks.shape
    masks = masks.bool()

    inp = resize(image_pil, (224,224), interpolation=Image.BICUBIC)
    px  = pil_to_tensor(inp).float().unsqueeze(0).to(device) / 255.

    penult = _get_tokens(model, px, layer_idx=-2)
    last   = _get_tokens(model, px, layer_idx=-1)
    mid    = _get_tokens(model, px, layer_idx=mid_idx)

    G = int(math.sqrt(penult.size(1)-1))
    penult = _resolve_anomaly(penult, G, top_k)
    patch_tok = _self_adjust(
        last[:,1:,:], mid[:,1:,:], beta
    ).squeeze(0)

    m224 = F.interpolate(
        masks.float().unsqueeze(1),
        size=(224,224), mode='nearest'
    ).squeeze(1)
    patch_masks = F.interpolate(
        m224.unsqueeze(1), size=(G,G),
        mode='nearest'
    ).squeeze(1).bool()

    feats = []
    for k in range(K):
        lab, num = cc_label(patch_masks[k].cpu().numpy().astype(np.uint8))
        for lbl in range(1, num+1):
            comp = (lab == lbl)
            comp_ids = torch.tensor(comp, device=device).nonzero(as_tuple=False)
            ids = comp_ids[:,0] * G + comp_ids[:,1]
            if ids.numel()>0:
                f = patch_tok[ids].mean(0)
                feats.append(F.normalize(f,dim=0))
    if len(feats)==0:
        return torch.zeros((0, patch_tok.size(-1)), device=device)
    return torch.stack(feats)



@torch.inference_mode()
def build_support_box_feats(
        clip_model,
        support_imgs,
        support_boxes,
        *,
        device="cpu"):
    """Build support box features by concatenating all box features from all shots."""
    all_feats = []
    shot = support_imgs.size(0)

    for s in range(shot):
        img_tensor  = support_imgs[s]
        boxes_tensor = support_boxes[s]

        img_pil = Image.fromarray(
            (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype("uint8")
        )

        if boxes_tensor.ndim == 1:
            boxes_tensor = boxes_tensor.unsqueeze(0)

        if boxes_tensor.numel() == 0:
            continue

        feats = extract_scclip_box_features(
                    clip_model,
                    img_pil,
                    boxes_tensor.to(device),
                    device=device)
        all_feats.append(feats)

    if len(all_feats) == 0:
        D = clip_model.config.projection_dim
        return torch.zeros((0, D), device=device)

    return torch.cat(all_feats, dim=0)


def visualize_filtered_boxes(image_pil, boxes, title="Filtered Boxes"):
    """Visualize filtered boxes on image."""
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_pil)

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.tolist()

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)
        width, height = x_max - x_min, y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 5, f"Box: [{x_min}, {y_min}, {x_max}, {y_max}]", color='white',
                bbox=dict(facecolor='red', alpha=0.5), fontsize=8)

    ax.set_title(title)
    ax.axis("off")
    plt.show()

    


@torch.inference_mode()
def clip_mask_emb(pil_img: Image.Image,
                  masks,
                  clip_processor, clip_model,
                  device="cpu"):
    """Extract CLIP embeddings for each mask by cropping and masking image regions."""
    if isinstance(masks, Image.Image):
        masks = pil_to_tensor(masks)

    m = masks.clone()
    if m.dim() == 2:
        m = m.unsqueeze(0)
    m = (m.float() > 0).to(torch.bool)

    K, H, W = m.shape
    feats   = []

    np_img = np.array(pil_img)

    for k in range(K):
        mk = m[k]

        if mk.sum() == 0:
            feats.append(torch.zeros(512, device=device))
            continue

        ys, xs  = torch.where(mk)
        y0, y1  = ys.min().item(), ys.max().item() + 1
        x0, x1  = xs.min().item(), xs.max().item() + 1

        crop_arr          = np_img[y0:y1, x0:x1].copy()
        mk_crop           = mk[y0:y1, x0:x1].cpu().numpy()
        crop_arr[~mk_crop] = 0

        crop_pil = Image.fromarray(crop_arr)

        enc = clip_processor(images=crop_pil,
                             return_tensors="pt").to(device)
        img_feat = clip_model.get_image_features(**enc)
        feats.append(F.normalize(img_feat.squeeze(0), dim=-1))

    return torch.stack(feats)


def extract_local_features_with_mask(model, processor, image_pil, mask, device):
    """Extract local features by masking image and extracting CLIP features."""
    mask_resized = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), 
                                                  size=image_pil.size[::-1], 
                                                  mode='nearest').squeeze().long()
    mask_np = mask_resized.cpu().numpy()
    masked_image = np.array(image_pil).copy()
    masked_image[mask_np == 0] = 0
    
    masked_image_pil = Image.fromarray(masked_image)
    
    with torch.no_grad():
        inputs = processor(images=masked_image_pil, return_tensors="pt").to(device)
        outputs = model.get_image_features(pixel_values=inputs["pixel_values"])
        local_features = outputs.squeeze(0)
    
    return local_features



def visualize_pred_mask(support_img, support_mask, alpha=0.5, title='title'):
    """Visualize support mask overlaid on support image with red color."""
    if isinstance(support_img, torch.Tensor):
        support_img_pil = Image.fromarray(
            (support_img.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).astype('uint8')
        )
    else:
        support_img_pil = support_img

    if isinstance(support_mask, torch.Tensor):
        mask_np = support_mask.cpu().numpy()
    elif isinstance(support_mask, np.ndarray):
        mask_np = support_mask
    else:
        raise TypeError("support_mask must be a torch.Tensor or numpy.ndarray")

    assert support_img_pil.size == (mask_np.shape[1], mask_np.shape[0]), "Image and mask size mismatch"

    img_np = np.array(support_img_pil)

    red_mask = np.zeros_like(img_np)
    red_mask[..., 0] = 255

    overlay = np.where(mask_np[..., None] > 0, red_mask, img_np)

    blended = (img_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

    plt.figure(figsize=(6, 6))
    plt.imshow(blended)
    plt.title(title)
    plt.axis("off")
    plt.show()
    

def extract_global_features_from_text(class_name, processor, model, device):
    """Extract global features from class name text using CLIP."""
    text = f"a photo of a {class_name}"
    inputs = processor(text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_text_features(input_ids=inputs["input_ids"])
        global_features = outputs.squeeze(0)
    
    return global_features

def extract_global_features_from_background(class_name, processor, model, device):
    """Extract global features from background text using CLIP."""
    text = f"a photo of a background"    
    inputs = processor(text=text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.get_text_features(input_ids=inputs["input_ids"])
        global_features = outputs.squeeze(0)
    
    return global_features

def extract_clip_box_features(clip_model, query_img_tensor, boxes, device):
    """Extract CLIP box features by cropping image regions and extracting [CLS] token features."""
    clip_model.eval()
    query_img_tensor = query_img_tensor.to(device)
    box_features = []

    for box in boxes:
        x1, y1, x2, y2 = box.int()
        cropped = crop(query_img_tensor, y1.item(), x1.item(), (y2 - y1).item(), (x2 - x1).item())
        cropped_pil = transforms.ToPILImage()(cropped.cpu())

        inputs = clip_processor(images=cropped_pil, return_tensors="pt").to(device)

        with torch.no_grad():
            vision_outputs = clip_model.vision_model(**inputs.vision_model_input)
            feature = vision_outputs.last_hidden_state[:, 0, :]
            feature = F.normalize(feature, dim=-1)

        box_features.append(feature.squeeze(0))

    if box_features:
        return torch.stack(box_features, dim=0)
    else:
        return torch.zeros((0, clip_model.config.projection_dim), device=device)



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


if __name__ == "__main__":
    utils_seed.fix_randseed(0)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformations and dataset
    transform = Compose([
        ToTensor()])

    coco_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    datapath = "./data"
    fold = 1
    split = "val"
    shot = 1
    use_original_imgsize = False

    dataset = DatasetCOCO(datapath, fold, transform, split, shot, use_original_imgsize)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    processor = Owlv2Processor.from_pretrained("google/owlv2-large-patch14-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-large-patch14-ensemble")

    model.to(device)
    model.eval()

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    clip_model.config.output_hidden_states = True
    clip_model.vision_model.return_dict     = True

    sam_checkpoint = "./sam_checkpoints/sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    total_miou = 0.0
    num_batches = 0
    total_batches = len(dataloader)
    Evaluator.initialize()
    average_meter = AverageMeter(dataloader.dataset)
    

    for i, batch in enumerate(dataloader, start=1):
        batch = utils_seed.to_cuda(batch)
        query_img = batch['query_img'].squeeze(0).to(device)
        query_mask = batch['query_mask'].squeeze(0).to(device)
        query_img_pil = Image.fromarray((query_img.cpu().permute(1, 2, 0).numpy() * 255).astype('uint8'))
        
        support_imgs   = batch['support_imgs'][0].to(device)
        support_boxes  = batch['support_boxes'][0]
        support_masks  = batch['support_masks'][0].to(device)

        support_embs = []
        for s in range(support_imgs.size(0)):
            sup_pil = Image.fromarray(
                (support_imgs[s].cpu().permute(1,2,0).numpy() * 255).astype("uint8")
            )
            mask_np = support_masks[s].cpu().numpy().astype(bool)
            img_np  = np.array(sup_pil)
            img_np[~mask_np] = 0
            sup_crop = Image.fromarray(img_np)

            enc_sup = clip_processor(images=sup_crop, return_tensors="pt").to(device)
            with torch.no_grad():
                emb = clip_model.get_image_features(**enc_sup)
            support_embs.append(F.normalize(emb.squeeze(0), dim=-1))

        support_emb = torch.stack(support_embs, dim=0).mean(0)

        class_id = batch['class_id']
        class_name = coco_class_names[class_id]
        lines = [[f'a photo of a {class_name}']]

        text_inputs = clip_processor(
            text=[f"a photo of a {class_name}"], return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            class_emb = clip_model.get_text_features(**text_inputs)
        class_emb = F.normalize(class_emb.squeeze(0), dim=-1)



        with torch.no_grad():
            inputs = processor(text=lines, images=query_img_pil, return_tensors="pt").to(device)
            outputs = model(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"]
            )

        target_sizes = torch.tensor([query_img_pil.size[::-1]], dtype=torch.float32).to(device)
        results = processor.post_process_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.08
        )

        pred_boxes = results[0]["boxes"].to(device)
        pred_scores = results[0]["scores"].to(device)            

        if pred_boxes.shape[0] > 0:
            s_min, s_max = pred_scores.min().item(), pred_scores.max().item()
            denom = max(s_max - s_min, 1e-6)

            reweighted_scores = []
            for box, orig_score in zip(pred_boxes, pred_scores):
                x0, y0, x1, y1 = map(int, box.tolist())
                crop = query_img_pil.crop((x0, y0, x1, y1))

                cropped_np = np.array(crop)
                if cropped_np.ndim == 3:
                    h, w, c = cropped_np.shape
                    if h == 1 and c == 3:
                        cropped_np = cropped_np[0]
                    elif c == 1:
                        cropped_np = np.concatenate([cropped_np]*3, axis=2)
                elif cropped_np.ndim == 2:
                    cropped_np = np.stack([cropped_np]*3, axis=-1)
                else:
                    raise ValueError(f"Unsupported cropped image shape: {cropped_np.shape}")
                crop_rgb = Image.fromarray(cropped_np.astype(np.uint8)).convert("RGB")

                enc = clip_processor(images=crop_rgb, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}

                with torch.no_grad():
                    img_emb = clip_model.get_image_features(**enc)
                img_emb = F.normalize(img_emb.squeeze(0), dim=-1)

                sim_text    = (img_emb @ class_emb).item()
                sim_support = (img_emb @ support_emb).item()
                score_norm  = (orig_score.item() - s_min) / denom
                
                final_score = (sim_text + sim_support + score_norm) / 3
                reweighted_scores.append(final_score)

            reweighted = torch.tensor(reweighted_scores, device=pred_boxes.device)
            tau = reweighted.mean() + 0.3 * reweighted.std()
            keep = reweighted > tau
            filtering_box = pred_boxes[keep]
    
    
        if len(filtering_box) > 0:
            filtering_box = filtering_box
        else:
            filtering_box = pred_boxes
            print('not_filterd_box')

            
        predictor.set_image(np.array(query_img_pil))
        
        masks_list = []

        if isinstance(filtering_box, torch.Tensor) and filtering_box.numel() > 0:
            for box in filtering_box.cpu().numpy():
                x_min, y_min, x_max, y_max = map(int, box)
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2

                masks, scores, low_res_logits = predictor.predict(
                    box=box,
                    multimask_output=True
                )
                best0 = np.argmax(scores)  
                init_mask      = masks[best0]
                init_logits    = low_res_logits[best0:best0+1]

                masks1, scores1, logits1 = predictor.predict(
                    box=box,
                    mask_input=init_logits,
                    multimask_output=True
                )
                
    
                best1 = np.argmax(scores1)
                mask1      = masks1[best1]
                logits1_in = logits1[best1:best1+1]
                     
                                
                ys, xs = np.nonzero(mask1)
                if xs.size == 0 or ys.size == 0:
                    masks_list.append(mask1)
                    break
                
                x0, x1 = xs.min(), xs.max()
                y0, y1 = ys.min(), ys.max()
                refine_box = np.array([x0, y0, x1, y1])[None, :]

                masks2, scores2, logits2 = predictor.predict(
                    box=refine_box,
                    mask_input=logits1_in,
                    multimask_output=True
                )
                
                mask_tensor = torch.from_numpy(masks2).to(device)

                all_mask_feats = []
                for k in range(mask_tensor.shape[0]):
                    single_mask = mask_tensor[k]
                    feat_k = extract_local_features_with_mask(
                        clip_model,
                        clip_processor,
                        query_img_pil,
                        single_mask,
                        device=device
                    )
                    all_mask_feats.append(feat_k)

                mask_feats = torch.stack(all_mask_feats, dim=0)
                scores2_t = torch.from_numpy(scores2).to(mask_feats.device)
                
                
                mask_feats = F.normalize(mask_feats, dim=-1)
                class_emb  = F.normalize(class_emb, dim=-1)

                mask_sims = torch.matmul(mask_feats, class_emb)

                
                best2      = np.argmax(scores2)
                best_idx  = mask_sims.argmax().item()
                final_mask = masks2[best2]
       
                
                
                best_sim = mask_sims[best_idx]
                
                if best_sim < 0.2:
                    continue
                                
                

                if final_mask.sum() == 0:
                    masks_list.append(mask1)
                else:
                    masks_list.append(final_mask)
    
                

                    
        
        else:
            height, width = query_img_pil.size[::-1]
            full_image_box = np.array([[0, 0, width, height]])
            mask, _, _ = predictor.predict(box=full_image_box)
            masks_list.append(mask[0])

            
        if isinstance(filtering_box, torch.Tensor) and filtering_box.numel() > 0 and len(masks_list) == 0:
            for box in filtering_box.cpu().numpy():

                x_min, y_min, x_max, y_max = map(int, box)
                cx = (x_min + x_max) // 2
                cy = (y_min + y_max) // 2
                topk_xy = np.array([[cx, cy]])
                topk_label = np.array([1])

                masks, scores, logits = predictor.predict(
                    box=box,
                    multimask_output=True
                )            
                masks_list.extend(masks)   
            
        

        combined_mask = np.any(np.stack(masks_list), axis=0).astype(np.float32)
        combined_mask_tensor = torch.tensor(combined_mask, dtype=torch.float32).to(device)
        
        
        if query_mask.shape != combined_mask_tensor.shape:
            query_mask_resized = torch.nn.functional.interpolate(
                query_mask.unsqueeze(0).unsqueeze(0).float(),
                size=combined_mask_tensor.shape,
                mode="nearest"
            ).squeeze().to(device)            
        else:
            query_mask_resized = query_mask
    

        pred_mask_batch = combined_mask_tensor.unsqueeze(0)
        query_mask_batch = query_mask_resized.unsqueeze(0)
        batch['query_mask'] = query_mask_batch
        
        pred_mask_batch = pred_mask_batch.to(device)
        query_mask_batch = query_mask_batch.to(device)
        
        area_inter, area_union = Evaluator.classify_prediction(pred_mask_batch, batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(i, total_batches, epoch=-1, write_batch_idx=1)

    average_meter.write_result('Test', 0)
    miou, fb_iou, _ = average_meter.compute_iou()

    logger.info(f"Fold {fold} mIoU: {miou:.2f} \t FB-IoU: {fb_iou:.2f}")
    logger.info('==================== Finished Testing ====================')
    
    
