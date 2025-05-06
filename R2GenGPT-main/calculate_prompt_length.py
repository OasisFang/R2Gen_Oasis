import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoImageProcessor
import argparse
import sys
import os
import traceback # <-- Moved import traceback here

# --- Try to import the config parser ---
try:
    # Assuming config.py is in the root or a 'configs' subdir accessible via path
    from config import parser as config_parser
    print("Imported parser from config.py")
except ModuleNotFoundError:
    try:
        from configs.config import parser as config_parser
        print("Imported parser from configs/config.py")
    except ModuleNotFoundError:
        print("Error: Cannot find 'config.py'. Make sure it's in the project root or 'configs/' directory.", file=sys.stderr)
        # Define minimal args if config not found, requiring user to provide all needed args via command line
        config_parser = argparse.ArgumentParser(description="Manual Args for Prompt Length Calculation")
        config_parser.add_argument('--vision_model', required=True, type=str)
        config_parser.add_argument('--llama_model', required=True, type=str)
        config_parser.add_argument('--task', required=True, choices=['classification', 'report'])
        config_parser.add_argument('--precision', default='bf16-mixed', type=str)
        config_parser.add_argument('--max_new_tokens', type=int, default=150) # Default based on run script


def main():
    args = config_parser.parse_args()
    print("--- Configuration ---")
    print(f"Vision Model: {args.vision_model}")
    print(f"LLM Model: {args.llama_model}")
    print(f"Task: {args.task}")
    print(f"Precision Hint: {args.precision}")
    print(f"Max New Tokens (for suggestion): {args.max_new_tokens}")
    print("---------------------\n")

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Determine model dtype based on precision hint
    model_dtype = torch.bfloat16 if 'bf16' in args.precision else torch.float16 if '16' in args.precision else torch.float32
    print(f"Using model dtype: {model_dtype}")

    # --- Load Models and Tokenizer (Minimal) ---
    try:
        print("Loading Vision Components...")
        # Load Image Processor for size info
        image_processor = AutoImageProcessor.from_pretrained(args.vision_model)
        # Load Vision Encoder Config to get feature dim
        vision_config = AutoConfig.from_pretrained(args.vision_model)
        # Load Vision Encoder Model - place directly on device
        visual_encoder = AutoModel.from_pretrained(args.vision_model, torch_dtype=model_dtype).to(device)
        visual_encoder.eval() # Set to eval mode

        # Determine vision feature dimension
        vis_feature_dim = getattr(vision_config, 'hidden_size', getattr(vision_config, 'embed_dim', None))
        if vis_feature_dim is None and hasattr(visual_encoder, 'num_features'):
            vis_feature_dim = visual_encoder.num_features
        if vis_feature_dim is None:
             raise ValueError("Cannot determine vision feature dimension.")
        print(f"Vision Feature Dim: {vis_feature_dim}")

        print("\nLoading LLM Components...")
        # Load LLM Tokenizer
        llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=True)
        if llama_tokenizer.pad_token is None:
            llama_tokenizer.pad_token = llama_tokenizer.eos_token
            print(f"Set pad_token to: {llama_tokenizer.pad_token}")

        # Load LLM Config to get hidden size
        llama_config = AutoConfig.from_pretrained(args.llama_model)
        llm_hidden_size = llama_config.hidden_size
        print(f"LLM Hidden Dim: {llm_hidden_size}")

        # --- Get LLM Embedding Layer ---
        # Load the minimal model structure containing embeddings if possible
        print("Loading LLM (potentially only structure for embeddings)...")
        # Avoid device_map='auto' here if possible to simplify, place manually
        llama_model_for_embeddings = AutoModel.from_pretrained(
            args.llama_model,
            config=llama_config,
            torch_dtype=model_dtype,
        ).to(device)
        llama_model_for_embeddings.eval()

        # Get the embedding layer
        try:
             embed_tokens = llama_model_for_embeddings.get_input_embeddings()
        except AttributeError:
             # Fallback for models where embeddings are nested (e.g., under model.embed_tokens)
             if hasattr(llama_model_for_embeddings, 'model') and hasattr(llama_model_for_embeddings.model, 'embed_tokens'):
                  embed_tokens = llama_model_for_embeddings.model.embed_tokens
             # Fallback for Llama models specifically
             elif hasattr(llama_model_for_embeddings, 'embed_tokens'):
                 embed_tokens = llama_model_for_embeddings.embed_tokens
             else:
                 print("Could not automatically find embedding layer. Trying common names...")
                 # Add more potential paths if needed based on the specific LLM architecture
                 found = False
                 if hasattr(llama_model_for_embeddings, 'embeddings'): # e.g., BERT
                     if hasattr(llama_model_for_embeddings.embeddings, 'word_embeddings'):
                        embed_tokens = llama_model_for_embeddings.embeddings.word_embeddings
                        found = True
                 if not found:
                     raise RuntimeError("Cannot access LLM input embedding layer. Check model structure.")
        print("Accessed LLM embedding layer.")
        # Embeddings should be on the same device as the model
        embed_tokens = embed_tokens.to(device)


    except Exception as e:
        print(f"Error loading models/tokenizer: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # --- Initialize Projection Layer ---
    # Use the same dtype as the models
    llama_proj = nn.Linear(vis_feature_dim, llm_hidden_size, dtype=model_dtype).to(device)
    projection_layer_norm = nn.LayerNorm(llm_hidden_size, dtype=model_dtype).to(device)
    llama_proj.eval()
    projection_layer_norm.eval()


    # --- Define Core Prompt ---
    if args.task == 'report':
        prompt_text_core = 'Generate a comprehensive and detailed diagnostic report for this chest X-ray image.'
    elif args.task == 'classification':
        # Recreate the disease list - Ensure this matches R2GenGPT.py
        diseases = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
        disease_list_str = ", ".join(diseases)
        prompt_text_core = (
            "Analyze the provided chest X-ray image. For each of the following 14 conditions, state whether it is 'Positive' or 'Negative'. "
            f"The conditions are: {disease_list_str}. "
            "Format the output as a comma-separated list, e.g., 'Atelectasis:Negative, Cardiomegaly:Positive, ...'."
        )
    else:
        print(f"Error: Unsupported task '{args.task}'", file=sys.stderr)
        sys.exit(1)
    print(f"\nUsing Core Prompt (start): {prompt_text_core[:100]}...")


    # --- Create Dummy Image Tensor ---
    try:
        # Get size from image processor config
        # Use .get() with defaults for robustness
        proc_size = image_processor.size
        img_height = proc_size.get('shortest_edge', proc_size.get('height', 224))
        img_width = proc_size.get('shortest_edge', proc_size.get('width', 224))

        # *** FIXED: Use torch.rand to generate values in [0, 1) range ***
        dummy_input_for_processor = torch.rand(1, 3, img_height, img_width)
        print(f"\nCreated dummy image tensor (before processing) with shape: {dummy_input_for_processor.shape}")

        # Process with image_processor to get the exact input format for the visual encoder
        # The processor handles resizing, normalization etc.
        # Convert dummy tensor to list of images or correct format processor expects
        # Often, processors take PIL Images or numpy arrays, but some take tensors.
        # Check the specific processor's requirements if issues persist.
        # Assuming it can handle a tensor in [0, 1] range:
        inputs = image_processor(images=dummy_input_for_processor, return_tensors="pt")

        pixel_values = inputs['pixel_values'].to(device, dtype=model_dtype) # Move to device and set dtype
        print(f"Created processed dummy image tensor (pixel_values) with shape: {pixel_values.shape}")

    except Exception as e:
        print(f"Error creating/processing dummy image: {e}", file=sys.stderr)
        traceback.print_exc() # Now traceback should be defined
        sys.exit(1)


    # --- Simulate `encode_img` ---
    print("\nSimulating encode_img...")
    with torch.no_grad():
        visual_outputs = visual_encoder(pixel_values=pixel_values)

        # Extract embeddings (adjust based on actual model output keys)
        if hasattr(visual_outputs, 'last_hidden_state'):
            image_embeds = visual_outputs.last_hidden_state
        elif hasattr(visual_outputs, 'pooler_output'):
            image_embeds = visual_outputs.pooler_output.unsqueeze(1)
        elif isinstance(visual_outputs, torch.Tensor):
             image_embeds = visual_outputs
             if image_embeds.dim() == 2: image_embeds = image_embeds.unsqueeze(1)
        else:
            # Attempt to access common keys if direct attributes fail
            output_keys = visual_outputs.keys() if hasattr(visual_outputs, 'keys') else []
            if 'last_hidden_state' in output_keys:
                 image_embeds = visual_outputs['last_hidden_state']
            elif 'pooler_output' in output_keys:
                 image_embeds = visual_outputs['pooler_output'].unsqueeze(1)
            else:
                 print(f"Error: Cannot find suitable vision embeddings in output. Available keys: {output_keys}", file=sys.stderr)
                 sys.exit(1)

        image_embeds_proj = llama_proj(image_embeds.to(llama_proj.weight.device, dtype=llama_proj.weight.dtype))
        image_embeds_norm = projection_layer_norm(image_embeds_proj)
        # atts_img = torch.ones(image_embeds_norm.size()[:-1], dtype=torch.long).to(device)
        print(f"Image Embeddings Norm Shape (Batch, ImgSeqLen, HiddenDim): {image_embeds_norm.shape}")


    # --- Simulate `prompt_wrap` ---
    print("\nSimulating prompt_wrap...")
    # Use the specific tokens for Llama 3 format
    prompt_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<Img>"
    prompt_end = f"</Img> {prompt_text_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    batch_size = image_embeds_norm.shape[0]

    with torch.no_grad():
        # Tokenize text parts - ensure they are on the same device as the embedding layer
        p_before_tokens = llama_tokenizer(prompt_start, return_tensors="pt", add_special_tokens=False).to(device)
        p_after_tokens = llama_tokenizer(prompt_end, return_tensors="pt", add_special_tokens=False).to(device)

        # Get embeddings using the loaded embedding layer
        p_before_embeds = embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)

        # Ensure all parts are on the same device before concatenation
        image_embeds_norm = image_embeds_norm.to(p_before_embeds.device)

        # Concatenate embeddings
        wrapped_emb = torch.cat([p_before_embeds, image_embeds_norm, p_after_embeds], dim=1)

    # --- Final Result ---
    prompt_length = wrapped_emb.shape[1]
    image_tokens_length = image_embeds_norm.shape[1]
    text_prompt_length = prompt_length - image_tokens_length

    print("\n--- Results ---")
    print(f"Shape of final wrapped prompt embeddings (Batch, SeqLen, HiddenDim): {wrapped_emb.shape}")
    print(f"===> Calculated Prompt Sequence Length (`prompt_embeds.shape[1]`): {prompt_length}")
    print(f"     - Vision Model Output Sequence Length (ImgSeqLen): {image_tokens_length}")
    print(f"     - Textual Prompt Parts Length (Tokenized): {text_prompt_length}")
    print("---------------")
    print(f"\nRecommendation:")
    print(f"Your '--max_length' parameter in run_classification.sh MUST be greater than this prompt length ({prompt_length}).")
    print(f"It needs to accommodate the prompt PLUS the maximum length of the text you want the model to generate.")
    print(f"Suggested minimum value = Prompt Length ({prompt_length}) + Max New Tokens (--max_new_tokens = {args.max_new_tokens})")
    suggested_min_max_length = prompt_length + args.max_new_tokens
    print(f"===> Consider setting --max_length in your run script to at least: {suggested_min_max_length}")
    # Try accessing max_length from args if config was loaded, otherwise show N/A
    current_max_len_val = getattr(args, 'max_length', 'N/A (config not fully parsed)')
    print(f"     (Current value in script according to parsed args: {current_max_len_val})")
    print("     You might want to add a small buffer (e.g., 16 or 32 tokens) to this minimum value.")
    print("     Common values like 768, 1024, or 2048 might be suitable depending on the result.")
    print("---------------")

if __name__ == "__main__":
    main()