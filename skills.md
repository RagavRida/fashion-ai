# skills.md — CNN–Diffusion Powered Fashion Reuse (DeepFashion)

## 0) Project Name
**CNN–Diffusion Powered Fashion Reuse**  
A multimodal fashion upcycling and redesign system trained on **DeepFashion**.

---

## 1) Mission
Build a production-ready AI system that supports:

- **Prompt-only fashion generation**
- **Image-only garment redesign**
- **Image + prompt controlled redesign**
- **Reference-image style transfer**
- **Multi-output generation + ranking**
- **Interactive refinement**
- **Tailoring instruction generation**

The system must be trained on **DeepFashion** and optimized for:
- garment structure consistency  
- texture realism  
- upcycling suitability  

---

## 2) Target Outputs
### Model Artifacts
- `checkpoints/fashion_lora.safetensors`
- `checkpoints/fashion_controlnet/`
- `checkpoints/fashion_ip_adapter/`
- `checkpoints/garment_segmentation.pth`
- `checkpoints/fabric_classifier.pth` *(optional)*

### Datasets (Generated)
- `data/processed/metadata.jsonl`
- `data/processed/images_512/`
- `data/processed/masks/`
- `data/processed/edges/`
- `data/processed/prompts/`

### Demo App
- Flask API server with endpoints:
  - `/generate`
  - `/redesign`
  - `/redesign_prompt`
  - `/style_transfer`
  - `/refine`
- Simple React UI

---

## 3) Repository Structure (Must Create)
```
fashion-reuse-ai/
  README.md
  skills.md
  requirements.txt
  configs/
    dataset.yaml
    train_lora.yaml
    train_controlnet.yaml
    train_ip_adapter.yaml
    train_segmentation.yaml
    inference.yaml
  data/
    raw/
    processed/
  dataset_builder/
    download_deepfashion.py
    preprocess.py
    build_prompts.py
    build_edges.py
    build_masks.py
    export_jsonl.py
  models/
    segmentation/
    fabric_classifier/
    diffusion/
    controlnet/
    ip_adapter/
    ranking/
  training/
    train_segmentation.py
    train_lora.py
    train_controlnet.py
    train_ip_adapter.py
    train_discriminator.py
  inference/
    pipeline.py
    generate.py
    redesign.py
    refine.py
    ranker.py
  evaluation/
    fid.py
    clip_score.py
    structure_score.py
    texture_score.py
    user_study_template.md
  api/
    app.py
    schemas.py
  frontend/
    (react app)
  outputs/
    samples/
    logs/
```

---

## 4) Skills Required (Agent Must Perform)

### Skill A — Dataset Acquisition (DeepFashion)
**Goal:** Download and organize DeepFashion.

**Actions:**
- Download DeepFashion (Category & Attribute + In-shop retrieval)
- Verify file integrity
- Store under `data/raw/`

**Success Criteria:**
- Images accessible
- Labels parsed successfully

---

### Skill B — Preprocessing + Prompt Synthesis
**Goal:** Convert DeepFashion into diffusion-friendly training data.

**Must produce:**
- 512×512 garment crops
- metadata JSONL
- auto-generated prompts
- canny edges
- segmentation masks (true or pseudo)

**Prompt generation rules:**
- Use category + attributes
- Extract dominant colors
- Add style keywords:
  - “streetwear”, “formal”, “casual”, “summer”, “winter”
- Add sustainability tags:
  - “recycled”, “upcycled”, “eco-friendly”
- Example prompt:
  - `"a recycled denim jacket, long sleeves, blue, streetwear style, high realism"`

**Success Criteria:**
- `metadata.jsonl` created with >= 50k valid samples (or as many as dataset allows)
- each sample has:
  - image_path
  - prompt
  - category
  - attributes
  - edge_path
  - mask_path

---

### Skill C — Garment Segmentation CNN
**Goal:** Train or fine-tune a segmentation model.

**Model options:**
- U-Net (fast, simple)
- Mask R-CNN (better but heavier)

**Inputs:**
- images_512

**Outputs:**
- garment mask

**Training Output:**
- `checkpoints/garment_segmentation.pth`

**Success Criteria:**
- Good segmentation on test samples
- IoU target: >0.70 on garment region (if ground truth exists)
- If no ground truth, use visual verification + pseudo-label stability

---

### Skill D — Fabric / Material Classifier (Optional)
**Goal:** Predict fabric types (denim/cotton/wool/leather).

**Approach:**
- Weakly supervised from attributes if available
- Or skip if dataset insufficient

**Output:**
- `checkpoints/fabric_classifier.pth`

---

## 5) Core Training Skills (GPU Work)

### Skill E — LoRA Fine-Tuning (Diffusion)
**Goal:** Make diffusion model fashion-aware using DeepFashion prompts.

**Base model:**
- SDXL if GPU strong
- SD 1.5 if GPU limited

**Method:**
- LoRA fine-tuning via diffusers + accelerate

**Training Config:**
- resolution: 512
- mixed precision: fp16
- batch size: GPU-dependent
- gradient checkpointing: enabled
- xformers: enabled
- learning rate: 1e-4 to 5e-5
- max steps: 20k–80k depending on budget

**Outputs:**
- `checkpoints/fashion_lora.safetensors`

**Success Criteria:**
- Prompts like “denim jacket” generate fashion-like outputs
- Better fashion realism than base SD

---

### Skill F — ControlNet Training
**Goal:** Improve garment structure and consistency.

**Control signals:**
- edges (canny)
- segmentation masks
- pose (optional)

**Output:**
- `checkpoints/fashion_controlnet/`

**Success Criteria:**
- Given garment edges, output respects silhouette
- Reduced structural artifacts vs LoRA-only

---

### Skill G — IP-Adapter / Reference Conditioning
**Goal:** Preserve identity of uploaded garments or reference styles.

**Training:**
- Condition diffusion on reference image embeddings (CLIP image encoder)

**Outputs:**
- `checkpoints/fashion_ip_adapter/`

**Success Criteria:**
- Reference style transfer works reliably
- Input garment identity preserved during redesign

---

### Skill H — Optional GAN Discriminator Refinement
**Goal:** Improve seam sharpness and texture realism.

**Discriminators:**
- patch discriminator on garment region
- seam/edge discriminator
- texture discriminator

**Training:**
- Freeze diffusion mostly
- Fine-tune lightly with adversarial loss

**Outputs:**
- `checkpoints/fashion_discriminator.pth`
- updated diffusion weights *(optional)*

**Success Criteria:**
- Reduced blur in fabric textures
- Better seam realism

---

## 6) Inference Skills (Must Implement)

### Skill I — Multi-Mode Inference
Implement:

#### 1) Text-only
- prompt → N images

#### 2) Image-only redesign
- image → auto prompt suggestions
- generate N variations

#### 3) Image + prompt
- preserve garment identity
- apply prompt changes

#### 4) Reference style transfer
- garment + reference style → redesigned garment

---

### Skill J — Multi-Sample Ranking Engine
Generate 8–16 samples and rank by:

- CLIPScore(prompt, image)
- mask alignment score
- edge alignment score
- landmark deviation score *(optional)*
- texture realism score (LPIPS patch-based)

Return top 4.

**Success Criteria:**
- Top outputs look consistently better than random samples
- User sees stable high-quality outputs

---

### Skill K — Interactive Refinement Engine
User can say:
- “make it sleeveless”
- “change color to pastel”
- “add pockets”
- “make sleeves shorter”
- “make it more formal”

System must:
- parse request
- select region masks
- use inpainting to edit only requested regions
- preserve identity embeddings

---

### Skill L — Tailoring Instruction Generator
Given final output:
Generate step-by-step instructions:
- what changes were applied
- what to cut/stitich
- what to add
- fabric suggestions
- difficulty rating

This module is text-only but must be grounded in:
- garment category
- edits applied
- visible features

---

## 7) Evaluation Skills (Must Report)

### Automatic Metrics
- FID (baseline)
- CLIPScore
- Structure Consistency:
  - mask IoU between generated and control mask
- Edge Alignment:
  - correlation between canny edges and output edges
- Texture Score:
  - LPIPS on fabric patches

### Human Evaluation
- realism (1–5)
- wearability (1–5)
- prompt satisfaction (1–5)
- upcycling usefulness (1–5)

---

## 8) Deployment Skills

### Backend API
- Flask or FastAPI
- GPU inference
- async job queue *(optional)*

Endpoints:
- `POST /generate`
- `POST /redesign`
- `POST /redesign_prompt`
- `POST /style_transfer`
- `POST /refine`

### Frontend
- React
- upload image + prompt
- display 4 outputs
- refine with chat-style instructions

---

## 9) GPU Instance Execution Rules
### Training Best Practices
- Use fp16 mixed precision
- enable gradient checkpointing
- enable xformers
- log to W&B
- save checkpoints every N steps
- resume training from checkpoint automatically

### Efficiency
- use LoRA first (cheapest)
- only train ControlNet after LoRA works
- only train IP-Adapter after ControlNet works
- GAN refinement is optional

---

## 10) Acceptance Criteria (Project Completion)
Project is considered complete when:

1) User can generate fashion from prompt  
2) User can upload garment and get 4 redesign variations  
3) User can refine designs with follow-up prompts  
4) Output maintains garment structure reliably  
5) Model is trained on DeepFashion and documented  
6) Repo runs end-to-end on a GPU instance  

---

## 11) Example Prompts for Testing
- “Upcycle this denim jacket into a cropped streetwear jacket with patches.”
- “Make it eco-friendly, pastel colors, minimal design.”
- “Convert into a formal blazer style.”
- “Add embroidery patterns on sleeves only.”
- “Make it sleeveless and summer style.”
