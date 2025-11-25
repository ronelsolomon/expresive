import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif
from PIL import Image

import os
from diffusers import StableDiffusionPipeline

# Configuration - USING ANIME MODEL
MODEL_PATH = "./Counterfeit-V3.0.safetensors"  # Local model file
MOTION_ADAPTER_ID = "guoyww/animatediff-motion-adapter-v1-5"
LORA_PATH = "ghibli_style.safetensors"  # Optional: add Ghibli LoRA for extra style

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Using device: {device}")
print(f"Using local model: {os.path.basename(MODEL_PATH)}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {os.path.abspath(MODEL_PATH)}")

# Step 1: Load Motion Adapter
print("Loading Motion Adapter...")
motion_adapter = MotionAdapter.from_pretrained(
    MOTION_ADAPTER_ID,
    torch_dtype=dtype
)

# Step 2: Load AnimateDiff Pipeline with Anime Model
print("Loading AnimateDiff Pipeline with Anime Model...")
# First load the base model
print("Loading base model...")
pipe = StableDiffusionPipeline.from_single_file(
    MODEL_PATH,
    torch_dtype=dtype,
)

# Then add motion adapter
pipe = AnimateDiffPipeline(
    vae=pipe.vae,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    unet=pipe.unet,
    scheduler=pipe.scheduler,
    feature_extractor=pipe.feature_extractor,
    motion_adapter=motion_adapter,
)

# Step 3: Load Ghibli Style LoRA (Optional but recommended)
print("Loading Style LoRA...")
try:
    pipe.load_lora_weights(LORA_PATH)
    pipe.fuse_lora(lora_scale=0.75)  # Adjust 0.5-0.9 for style strength
    print("✅ LoRA loaded successfully!")
except Exception as e:
    print(f"⚠️  No LoRA found: {e}")
    print("   Continuing with base anime model...")

# Step 4: Optimize pipeline
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_slicing()
if device == "cuda":
    pipe.enable_model_cpu_offload()
else:
    pipe = pipe.to(device)

print("Pipeline ready!")

# Step 5: Enhanced Anime/Ghibli Prompt with Spiderman
prompt = """
Spiderman swinging between skyscrapers at night, 
city lights reflecting off wet streets, dynamic motion blur,
studio ghibli style, anime art, makoto shinkai style,
hand-drawn animation, cel shaded, soft pastel colors,
cinematic lighting, detailed background painting,
atmospheric, dramatic action, high quality anime,
japanese animation, masterpiece, 4K, dynamic pose,
web swinging action, flowing red and blue costume,
reflective raindrops, urban nightscape
"""

negative_prompt = """
low quality, worst quality, blurry, distorted, ugly,
realistic, photo, photorealistic, 3d render, cgi,
watermark, text, signature, jpeg artifacts,
deformed, bad anatomy, bad hands,
western cartoon, disney style, pixar style
"""

print("\nGenerating Ghibli-style animation...")
print(f"Prompt: {prompt[:100]}...")

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_frames=16,  # 16 frames = 2 seconds at 8fps
    guidance_scale=7.5,  # Lower (6-7) for softer style, higher (8-9) for more detail
    num_inference_steps=30,  # More steps = better quality
    generator=torch.Generator(device).manual_seed(42)
)

# Step 6: Export to GIF
frames = output.frames[0]
export_to_gif(frames, "ghibli_animation.gif", fps=8)
print("\n✅ Animation saved as 'ghibli_animation.gif'")

# Optional: Save individual frames
for i, frame in enumerate(frames):
    frame.save(f"frame_{i:03d}.png")
print(f"✅ Saved {len(frames)} individual frames")

# ============================================
# TIPS FOR BETTER GHIBLI STYLE
# ============================================
print("\n" + "="*50)
print("🎨 TIPS FOR PERFECT GHIBLI STYLE:")
print("="*50)
print("""
1. MODELS TO TRY:
   - andite/anything-v4.0 (used here)
   - Linaqruf/anything-v3.0
   - stabilityai/stable-diffusion-xl-base-1.0 + Ghibli LoRA
   
2. GET GHIBLI LORA:
   - Visit https://civitai.com
   - Search "Studio Ghibli LoRA SD1.5"
   - Download .safetensors file
   - Place in same folder and update LORA_PATH
   
3. PROMPT KEYWORDS:
   - "studio ghibli style"
   - "makoto shinkai style" 
   - "cel shaded"
   - "hand-drawn animation"
   - "soft colors"
   - "painted background"
   
4. SETTINGS:
   - guidance_scale: 6-8 (lower = softer style)
   - num_inference_steps: 25-40 (more = better)
   - lora_scale: 0.6-0.8 (adjust strength)
""")

print("\n📁 Output files:")
print("   - ghibli_animation.gif (animated)")
print("   - frame_*.png (individual frames)")
print("\nEnjoy your Ghibli-style animation! 🎬✨")