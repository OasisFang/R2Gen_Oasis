import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModel
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import precision_recall_fscore_support, multilabel_confusion_matrix, roc_auc_score, accuracy_score
import torchvision.utils
import numpy as np
import warnings

# Suppress specific warnings if needed
# warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

class R2GenGPT(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # Saving hyperparameters allows accessing them with self.hparams
        self.save_hyperparameters(args)
        # Keep a direct reference to args if needed, though hparams is preferred
        self.args = args

        self.task = self.hparams.task
        self.learning_rate = self.hparams.learning_rate
        self.max_length = self.hparams.max_length # Max length for tokenizer encoding
        # Default end symbol if not provided
        self.end_sym = self.hparams.end_sym if hasattr(self.hparams, 'end_sym') and self.hparams.end_sym else '<|eot_id|>'

        # Generation parameters
        self.num_beams = self.hparams.beam_size
        self.do_sample = self.hparams.do_sample
        self.max_new_tokens = self.hparams.max_new_tokens # Max tokens for generation
        self.temperature = self.hparams.temperature if self.do_sample else 1.0
        self.repetition_penalty = self.hparams.repetition_penalty
        self.length_penalty = self.hparams.length_penalty

        # Override generation params for classification evaluation (use greedy search)
        if self.task == 'classification':
            print("Classification task: Overriding generation to greedy search (num_beams=1, do_sample=False).")
            self.num_beams = 1
            self.do_sample = False
            # Use a potentially shorter max_new_tokens for classification output if specified
            # self.max_new_tokens = getattr(self.hparams, 'eval_max_new_tokens', self.hparams.max_new_tokens)

        # --- Vision Encoder Setup ---
        print(f'Loading vision encoder: {self.hparams.vision_model}')
        try:
            vision_config = AutoConfig.from_pretrained(self.hparams.vision_model)
            self.visual_encoder = AutoModel.from_pretrained(self.hparams.vision_model)
            # Determine feature dimension robustly
            vis_feature_dim = getattr(vision_config, 'hidden_size', None)
            if vis_feature_dim is None: # Fallback for models like ViT
                 vis_feature_dim = getattr(vision_config, 'embed_dim', None)
            if vis_feature_dim is None and hasattr(self.visual_encoder, 'num_features'): # Another fallback
                 vis_feature_dim = self.visual_encoder.num_features
            if vis_feature_dim is None:
                 raise ValueError(f"Cannot determine feature dimension for vision model {self.hparams.vision_model}. Config: {vision_config}")
            print(f"Detected vision feature dimension: {vis_feature_dim}")
            self.vis_feature_dim = vis_feature_dim # Store for projection layer

        except Exception as e: print(f"Error loading vision model {self.hparams.vision_model}: {e}"); raise

        # Apply LoRA to vision encoder if configured
        if self.hparams.vis_use_lora:
            print("Applying LoRA to vision encoder...")
            # Ensure target modules exist in the model
            target_modules = self.hparams.vis_lora_target_modules
            print(f"Vision LoRA target modules: {target_modules}")
            peft_config_visual = LoraConfig(
                r=self.hparams.vis_r,
                lora_alpha=self.hparams.vis_alpha,
                target_modules=target_modules,
                lora_dropout=self.hparams.lora_dropout,
                bias="none" # or "all" or "lora_only"
            )
            self.visual_encoder = get_peft_model(self.visual_encoder, peft_config_visual)
            self.visual_encoder.print_trainable_parameters()
        elif self.hparams.freeze_vm:
            print(f'Freezing vision encoder: {self.hparams.vision_model}')
            for param in self.visual_encoder.parameters(): param.requires_grad = False
            # Ensure the final layer norm is trainable if it exists and VM is frozen
            # Example: if hasattr(self.visual_encoder, 'layernorm'): for param in self.visual_encoder.layernorm.parameters(): param.requires_grad = True
        else:
            print(f'Vision encoder is fully trainable: {self.hparams.vision_model}')
        # --- End Vision Encoder Setup ---

        # --- LLM Setup ---
        print(f'Loading LLM model: {self.hparams.llama_model}')
        try:
            # Load tokenizer
            self.llama_tokenizer = AutoTokenizer.from_pretrained(self.hparams.llama_model, use_fast=False)
            # Load LLM config first to get hidden size
            llama_config = AutoConfig.from_pretrained(self.hparams.llama_model)
            llm_hidden_size = llama_config.hidden_size
            print(f"LLM hidden size: {llm_hidden_size}")

            # Set pad token if missing (common for Llama models)
            if self.llama_tokenizer.pad_token is None:
                self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
                # Also update the config if necessary, though setting on tokenizer is usually sufficient
                # llama_config.pad_token_id = llama_config.eos_token_id
                print(f"Set LLM pad_token to eos_token: {self.llama_tokenizer.pad_token} (ID: {self.llama_tokenizer.pad_token_id})")

            # Check consistency of end_sym
            if hasattr(self.hparams, 'end_sym') and self.hparams.end_sym != self.llama_tokenizer.eos_token:
                 print(f"Warning: Configured end_sym '{self.hparams.end_sym}' differs from tokenizer EOS '{self.llama_tokenizer.eos_token}'. Using tokenizer's EOS for consistency.")
                 self.end_sym = self.llama_tokenizer.eos_token # Ensure consistency

            # Determine model dtype based on precision
            model_dtype = torch.bfloat16 if 'bf16' in self.hparams.precision else torch.float16 if '16' in self.hparams.precision else torch.float32
            print(f"Using model dtype: {model_dtype} based on precision '{self.hparams.precision}'")

            # Prepare arguments for from_pretrained
            from_pretrained_kwargs = {
                "config": llama_config,
                "torch_dtype": model_dtype,
                # Add attn_implementation="flash_attention_2" if applicable and installed
                # "attn_implementation": "flash_attention_2",
            }

            # Check for accelerate and add device_map='auto' if available
            try:
                import accelerate
                # device_map = "auto" is crucial for multi-GPU or large models
                from_pretrained_kwargs["device_map"] = "auto"
                print("Accelerate library found, using device_map='auto' for LLM loading.")
            except ImportError:
                print("Warning: accelerate library not installed. Loading LLM without device_map. This might lead to OOM errors on large models or multi-GPU setups.")

            # Handle low_resource (8-bit) loading
            if self.hparams.low_resource:
                 print("Loading LLM in low resource mode (8-bit)")
                 from_pretrained_kwargs["load_in_8bit"] = True
                 try: import bitsandbytes
                 except ImportError: print("Warning: bitsandbytes library not installed. 8-bit loading requires bitsandbytes.")
            else:
                print(f"Loading LLM in standard mode (dtype: {model_dtype})")

            # Load the LLM
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                self.hparams.llama_model,
                **from_pretrained_kwargs
            )

        except Exception as e: print(f"Error loading LLM model {self.hparams.llama_model}: {e}"); raise

        # Apply LoRA to LLM if configured
        if self.hparams.llm_use_lora:
            print("Applying LoRA to LLM...")
            lora_target_modules = self.hparams.llm_lora_target_modules
            print(f"LLM LoRA target modules: {lora_target_modules}")
            peft_config_llm = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.hparams.llm_r,
                lora_alpha=self.hparams.llm_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=lora_target_modules
                # Potentially add 'modules_to_save' if you want to train other parts like the lm_head
                # modules_to_save=["lm_head", "embed_tokens"] # Example
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config_llm)
            self.llama_model.print_trainable_parameters()
        elif not self.hparams.llm_use_lora:
            print('Freezing LLM base model layers (LLM LoRA is disabled)')
            # Freeze all parameters except potentially embeddings and the final head
            for name, param in self.llama_model.named_parameters():
                 # Adjust condition based on Llama model's layer names if needed
                 if "embed_tokens" not in name and "lm_head" not in name:
                    param.requires_grad = False
            # Optionally, print trainable parameters after freezing
            # total_params = sum(p.numel() for p in self.llama_model.parameters())
            # trainable_params = sum(p.numel() for p in self.llama_model.parameters() if p.requires_grad)
            # print(f"LLM Trainable parameters: {trainable_params} / {total_params} ({trainable_params/total_params*100:.2f}%)")

        # --- End LLM Setup ---

        # --- Projection Layer ---
        print(f"Creating projection layer from vision dim ({self.vis_feature_dim}) to LLM dim ({llm_hidden_size})")
        self.llama_proj = nn.Linear(self.vis_feature_dim, llm_hidden_size)
        # Add LayerNorm for stability
        self.projection_layer_norm = nn.LayerNorm(llm_hidden_size)
        # --- End Projection Layer ---

        # --- Task Specific Setup ---
        # Define disease list for classification
        self.diseases = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
            "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax", "Support Devices"
        ]
        self.diseases_lower = [d.lower() for d in self.diseases] # For case-insensitive matching
        self.num_classes = len(self.diseases)

        # Define prompts based on task
        if self.task == 'report':
            # Use a clear instruction for report generation
            self.prompt_text_core = 'Generate a comprehensive and detailed diagnostic report for this chest X-ray image.'
        elif self.task == 'classification':
            disease_list_str = ", ".join(self.diseases)
            # Explicitly define the expected output format
            self.prompt_text_core = (
"This is a 14-class classification task related to medical imaging, where the 14 disease categories are as follows: Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged Cardiomediastinum,Fracture,Lung Lesion,Lung Opacity,No Finding,Pleural Effusion,Pleural Other,Pneumonia,Pneumothorax,Support Devices. Please output \"Positive\" or \"Negative\" for each category." )
            
        else: raise ValueError(f"Unsupported task: {self.task}. Choose 'classification' or 'report'.")
        print(f"Core prompt for task '{self.task}': {self.prompt_text_core}")
        # --- End Task Specific Setup ---

        # --- Evaluation Setup ---
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.best_val_score = -float('inf') # Initialize best validation score

        # Setup scorers for report generation task
        if self.task == 'report':
             self.scorers = [
                 (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                 (Rouge(), "ROUGE_L"),
                 (Cider(), "CIDEr")
             ]
        # --- End Evaluation Setup ---

        # Load delta checkpoint if provided (AFTER model initialization)
        if hasattr(self.hparams, 'delta_file') and self.hparams.delta_file is not None:
            self.load_delta_checkpoint(self.hparams.delta_file)


    def load_delta_checkpoint(self, delta_path):
        """Loads weights from a delta checkpoint file (adapter weights + projection)."""
        if not os.path.exists(delta_path):
            print(f"Warning: Delta file not found at {delta_path}. Skipping.")
            return
        print(f"Loading delta checkpoint from {delta_path}...")
        try:
           # Load checkpoint to CPU first to avoid device mismatches
           map_location = 'cpu'
           state_dict = torch.load(delta_path, map_location=map_location)

           # Handle nested state dicts (common in PL checkpoints)
           if 'model' in state_dict: state_dict = state_dict['model']
           elif 'state_dict' in state_dict: state_dict = state_dict['state_dict']

           # Checkpoint keys for debugging
           # print(f"Delta file keys (first 5): {list(state_dict.keys())[:5]}...")

           # Load projection layer weights
           proj_state = {
               k: v for k, v in state_dict.items()
               if k.startswith("llama_proj.") or k.startswith("projection_layer_norm.")
           }
           if proj_state:
               load_result_proj = self.load_state_dict(proj_state, strict=False)
               print(f"Loaded projection layer weights from delta file. Result: {load_result_proj}")

           # Load LoRA adapter weights for LLM if applicable
           if isinstance(self.llama_model, PeftModel):
                # Extract weights with the PeftModel prefix (e.g., base_model.model...)
                # The exact prefix depends on how PeftModel wraps the base model
                # Common prefixes might be 'base_model.model.' or just the adapter name like 'default.'
                # Let's try to load directly assuming keys match the adapter structure
                llm_adapter_state = {}
                for k, v in state_dict.items():
                    # Check for common adapter weight patterns
                    if "lora_A" in k or "lora_B" in k:
                         # Try to remove potential prefixes added during saving
                         key_no_prefix = k.split('base_model.model.')[-1] # Common prefix
                         llm_adapter_state[key_no_prefix] = v

                if llm_adapter_state:
                     # It's often better to load the adapter using PeftModel's methods
                     # This assumes the adapter name is 'default'
                     try:
                         # self.llama_model.load_adapter(llm_adapter_state, "default") # This might expect a directory
                         # Loading the state dict directly into the PeftModel might be simpler
                         load_result_llm_adapter = self.llama_model.load_state_dict(state_dict, strict=False)
                         print(f"Loaded LLM adapter weights (using load_state_dict). Result: {load_result_llm_adapter}")
                     except Exception as e_adapter:
                         print(f"Error loading LLM adapter weights: {e_adapter}. Keys might not match.")
                         # print("LLM adapter state keys found:", list(llm_adapter_state.keys())[:5])
                         # print("Model adapter state keys expected:", list(self.llama_model.state_dict().keys())[:5])

                else: print("No LoRA adapter weights found for LLM in the delta file.")
           else: print("LLM is not a PeftModel, skipping adapter weight loading.")

           # Load LoRA adapter weights for Vision Encoder if applicable
           if isinstance(self.visual_encoder, PeftModel):
               # Similar logic as LLM adapter loading
               try:
                   load_result_vis_adapter = self.visual_encoder.load_state_dict(state_dict, strict=False)
                   print(f"Loaded Vision adapter weights (using load_state_dict). Result: {load_result_vis_adapter}")
               except Exception as e_vis_adapter:
                   print(f"Error loading Vision adapter weights: {e_vis_adapter}. Keys might not match.")
           else: print("Visual encoder is not a PeftModel, skipping adapter weight loading.")

        except Exception as e:
            print(f"Error loading delta checkpoint from {delta_path}: {e}")
            # Optionally re-raise or handle the error appropriately
            # raise e


    def parse_classification_label(self, text):
        """Parses the LLM output text into a multi-label binary vector."""
        labels_vector = [0] * self.num_classes
        if not isinstance(text, str) or not text.strip():
            # print("Warning: Received empty text for classification parsing.")
            return labels_vector

        # Normalize text: lowercase, remove extra spaces
        text_norm = text.lower().strip()
        labels_dict = {}

        try:
            # Split by comma, handling potential extra whitespace
            parts = [p.strip() for p in text_norm.split(',') if p.strip()]
            for part in parts:
                if ':' in part:
                    # Split only on the first colon
                    disease_part, label_part = part.split(':', 1)
                    disease_lower = disease_part.strip()
                    label = label_part.strip()

                    # Check if the disease name matches (case-insensitive)
                    if disease_lower in self.diseases_lower:
                        # Find the original case disease name
                        disease_idx = self.diseases_lower.index(disease_lower)
                        original_disease = self.diseases[disease_idx]
                        # Check if the label is 'positive'
                        if label == 'positive':
                            labels_dict[original_disease] = 1
                        # Optional: Handle 'negative' explicitly if needed, otherwise default is 0
                        # elif label == 'negative':
                        #     labels_dict[original_disease] = 0 # Ensure it's explicitly 0 if found negative
                    # else:
                        # print(f"Warning: Parsed disease '{disease_lower}' not in known list.")

        except Exception as e:
            # Log parsing errors without crashing
            print(f"Error parsing classification text: '{text[:100]}...': {e}")

        # Fill the vector based on the parsed dictionary
        for i, disease in enumerate(self.diseases):
            labels_vector[i] = labels_dict.get(disease, 0) # Default to 0 if not found or not positive

        # print(f"Parsed '{text[:50]}...' -> {labels_vector}") # Debugging print
        return labels_vector

    def decode(self, output_token_ids):
        """Decodes token IDs to text, handling potential tensor inputs."""
        if output_token_ids is None: return ""
        # Move tensor to CPU and convert to numpy if necessary
        if isinstance(output_token_ids, torch.Tensor):
            # Handle potential 0-dim tensor (single token ID)
            if output_token_ids.dim() == 0:
                output_token_ids = output_token_ids.unsqueeze(0)
            output_token_ids = output_token_ids.cpu().numpy()

        # Decode using the tokenizer
        try:
            # skip_special_tokens=True removes EOS, BOS, PAD etc.
            output_text = self.llama_tokenizer.decode(output_token_ids, skip_special_tokens=True)
            # Clean up potential leading/trailing whitespace
            output_text = output_text.strip()
        except Exception as e:
             print(f"Error during decoding: {e}")
             output_text = "" # Return empty string on error
        return output_text

    def score(self, ref, hypo):
        """Calculates evaluation metrics based on the task type."""
        if not ref or not hypo:
             print("Warning: Empty reference or hypothesis list passed to score function.")
             return {}

        # --- Report Generation Scoring ---
        if self.task == 'report':
            if not hasattr(self, 'scorers'):
                print("Warning: Scorers not initialized for report task.")
                return {}
            final_scores = {}
            # Ensure ref and hypo are in the correct format (dict of id: [str])
            if not isinstance(ref, dict) or not isinstance(hypo, dict):
                print("Error: ref and hypo must be dictionaries for report scoring.")
                return {}

            # Filter to common IDs
            common_ids = ref.keys() & hypo.keys()
            if not common_ids:
                 print("Warning: No common IDs between references and hypotheses.")
                 return {}
            filtered_ref = {idx: ref[idx] for idx in common_ids}
            filtered_hypo = {idx: hypo[idx] for idx in common_ids}

            print(f"Scoring {len(filtered_hypo)} samples for report generation...")
            for scorer, method in self.scorers:
                try:
                     # compute_score expects dict {id: [string]}
                     score, scores = scorer.compute_score(filtered_ref, filtered_hypo)
                     if isinstance(method, list): # Handle multiple scores from one scorer (like BLEU)
                         for m, s in zip(method, score): final_scores[m] = s
                     else: final_scores[method] = score
                except Exception as e: print(f"Error during scoring with {scorer.__class__.__name__}: {e}")
            return final_scores

        # --- Classification Scoring ---
        elif self.task == 'classification':
            all_true_labels, all_pred_labels = [], []
            all_true_texts, all_pred_texts = [], [] # Store raw texts for debugging if needed

            # Ensure ref and hypo are dicts {id: [label_string]}
            if not isinstance(ref, dict) or not isinstance(hypo, dict):
                 print("Error: ref and hypo must be dictionaries for classification scoring.")
                 return {}

            common_ids = ref.keys() & hypo.keys()
            if not common_ids:
                 print("Warning: No common IDs between reference and hypothesis for classification.")
                 return {}

            print(f"Parsing and scoring {len(common_ids)} samples for classification...")
            for sid in common_ids:
                # Assume the first element in the list is the relevant string
                true_text = ref[sid][0] if ref.get(sid) else ""
                pred_text = hypo[sid][0] if hypo.get(sid) else ""

                all_true_texts.append(true_text)
                all_pred_texts.append(pred_text)

                true_vector = self.parse_classification_label(true_text)
                pred_vector = self.parse_classification_label(pred_text)

                all_true_labels.append(true_vector)
                all_pred_labels.append(pred_vector)

            # If parsing failed for all samples, return default metrics
            if not all_true_labels:
                print("Warning: Could not parse any labels for classification scoring.")
                return {'F1_mac': 0.0, 'F1_mic': 0.0, 'P_mac': 0.0, 'R_mac': 0.0, 'Acc_mic': 0.0}

            # Convert lists of vectors to numpy arrays
            all_true_np = np.array(all_true_labels)
            all_pred_np = np.array(all_pred_labels)

            # Debugging: Print shapes and a few examples
            # print(f"Classification scoring: True labels shape={all_true_np.shape}, Predicted labels shape={all_pred_np.shape}")
            # print("Sample True vs Pred Vectors:")
            # for i in range(min(3, len(all_true_np))):
            #      print(f"  True[{i}]: {all_true_np[i]}, Pred[{i}]: {all_pred_np[i]}")
            #      print(f"    True Text: {all_true_texts[i][:50]}...")
            #      print(f"    Pred Text: {all_pred_texts[i][:50]}...")


            # Calculate Micro and Macro metrics using sklearn
            # zero_division=0 handles cases where a class has no true/pred samples
            p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(all_true_np, all_pred_np, average='micro', zero_division=0)
            p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(all_true_np, all_pred_np, average='macro', zero_division=0)

            # Calculate overall accuracy (Exact Match Ratio)
            # This calculates the fraction of samples where all labels were predicted correctly
            accuracy_subset = accuracy_score(all_true_np, all_pred_np)

            # --- Calculate Per-Class Metrics ---
            per_class_metrics = {}
            try:
                # Calculate metrics per class (average=None)
                p_per, r_per, f1_per, s_per = precision_recall_fscore_support(all_true_np, all_pred_np, average=None, zero_division=0, labels=list(range(self.num_classes)))

                # Calculate confusion matrix per class
                mcm = multilabel_confusion_matrix(all_true_np, all_pred_np, labels=list(range(self.num_classes)))

                # Calculate AUC per class if possible (requires probabilities, not implemented here)
                # auc_per = roc_auc_score(all_true_np, all_pred_np, average=None, multi_class='ovr') # Needs predicted probabilities

                for i, disease in enumerate(self.diseases):
                    tn, fp, fn, tp = mcm[i].ravel()
                    total = tn + fp + fn + tp
                    acc_cls = (tp + tn) / total if total > 0 else 0 # Per-class accuracy

                    per_class_metrics[disease] = {
                        'Acc': acc_cls,
                        'P': p_per[i],
                        'R': r_per[i],
                        'F1': f1_per[i],
                        'Support': int(s_per[i]), # Number of true instances for the class
                        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
                        # 'AUC': auc_per[i] # Add if AUC is calculated
                    }
                    # print(f"  Metrics for {disease}: P={p_per[i]:.3f}, R={r_per[i]:.3f}, F1={f1_per[i]:.3f}, Acc={acc_cls:.3f}, Support={s_per[i]}")
            except Exception as e_cls:
                print(f"Error calculating per-class metrics: {e_cls}")
            # --- End Per-Class Metrics ---

            return {
                'P_mic': p_mic, 'R_mic': r_mic, 'F1_mic': f1_mic,
                'P_mac': p_mac, 'R_mac': r_mac, 'F1_mac': f1_mac,
                'Acc_subset': accuracy_subset, # Exact match accuracy
                'Per_Class': per_class_metrics # Dictionary of per-class results
            }
        else:
            print(f"Warning: Unknown task '{self.task}' for scoring.")
            return {}

    def encode_img(self, images):
        """Encodes images using the visual encoder and projects them."""
        # Ensure images are on the correct device and dtype
        device = self.visual_encoder.device
        images = images.to(device, dtype=self.visual_encoder.dtype if hasattr(self.visual_encoder, 'dtype') else torch.float32)

        try:
            # Pass images through the visual encoder
            # Use output_hidden_states=True if you need intermediate layers
            visual_outputs = self.visual_encoder(pixel_values=images) # output_hidden_states=False by default

            # Extract the final layer's hidden states (or appropriate embedding)
            # Common attribute names: 'last_hidden_state', 'pooler_output', sometimes just the tensor itself
            if hasattr(visual_outputs, 'last_hidden_state'):
                # Typically (batch_size, sequence_length, hidden_size) for transformers
                image_embeds = visual_outputs.last_hidden_state
            elif hasattr(visual_outputs, 'pooler_output'):
                # Typically (batch_size, hidden_size) - unsqueeze to add sequence dim
                image_embeds = visual_outputs.pooler_output.unsqueeze(1)
                # print("Note: Using pooler_output from vision encoder.")
            elif isinstance(visual_outputs, torch.Tensor):
                # If the output is just the tensor (e.g., from some CNNs)
                 image_embeds = visual_outputs
                 # Add sequence dimension if it's missing, assuming (batch, features) -> (batch, 1, features)
                 if image_embeds.dim() == 2: image_embeds = image_embeds.unsqueeze(1)
            else:
                 raise KeyError(f"Cannot find suitable vision embeddings in output keys: {visual_outputs.keys()}")

            # Project image embeddings to LLM's hidden size
            # Ensure dtype consistency before projection if needed
            image_embeds_proj = self.llama_proj(image_embeds.to(self.llama_proj.weight.device, dtype=self.llama_proj.weight.dtype))

            # Apply LayerNorm to the projected embeddings
            image_embeds_norm = self.projection_layer_norm(image_embeds_proj)

            # Create attention mask for the image embeddings (all ones)
            atts_llama = torch.ones(image_embeds_norm.size()[:-1], dtype=torch.long).to(image_embeds_norm.device)

            return image_embeds_norm, atts_llama

        except Exception as e:
            print(f"Error during image encoding/projection: {e}")
            # Return dummy tensors on error with expected shapes and device
            b = images.shape[0]
            h = self.llama_model.config.hidden_size # LLM hidden dimension
            # Dummy shape: (batch_size, num_image_tokens (e.g., 1 for pooled), hidden_size)
            dummy_embeds = torch.zeros(b, 1, h, device=device, dtype=self.dtype)
            dummy_atts = torch.zeros(b, 1, dtype=torch.long, device=device)
            return dummy_embeds, dummy_atts

    def prompt_wrap(self, img_embeds, atts_img):
        """Wraps image embeddings with text prompts for LLM input."""
        # Define the text parts of the prompt using Llama 3 format
        # Reference: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3_1/
        # Use the specific tokens for Llama 3.1/3.2
        prompt_start = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n<Img>" # Placeholder for image
        prompt_end = f"</Img> {self.prompt_text_core}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" # Core instruction + start of assistant turn

        # Get batch size and device from image embeddings
        batch_size, device = img_embeds.shape[0], img_embeds.device

        # Tokenize the text parts
        # add_special_tokens=False because we are manually adding <|begin_of_text|> etc.
        p_before_tokens = self.llama_tokenizer(prompt_start, return_tensors="pt", add_special_tokens=False).to(device)
        p_after_tokens = self.llama_tokenizer(prompt_end, return_tensors="pt", add_special_tokens=False).to(device)

        # Convert token IDs to embeddings using the LLM's embedding layer
        # Ensure the embedding layer is accessible (might be inside self.llama_model.model)
        try:
            embed_tokens = self.llama_model.get_input_embeddings()
        except AttributeError: # Fallback if get_input_embeddings() isn't directly available
             if hasattr(self.llama_model, 'model') and hasattr(self.llama_model.model, 'embed_tokens'):
                 embed_tokens = self.llama_model.model.embed_tokens
             else: raise RuntimeError("Cannot access LLM input embedding layer.")

        p_before_embeds = embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1) # Expand to batch size
        p_after_embeds = embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1) # Expand to batch size

        # Concatenate text embeddings and image embeddings: [prompt_start, img_embeds, prompt_end]
        wrapped_emb = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)

        # Create corresponding attention masks
        atts_before = torch.ones(p_before_embeds.size()[:-1], dtype=torch.long).to(device)
        atts_after = torch.ones(p_after_embeds.size()[:-1], dtype=torch.long).to(device)
        # Concatenate attention masks: [atts_before, atts_img, atts_after]
        wrapped_atts = torch.cat([atts_before, atts_img, atts_after], dim=1)

        return wrapped_emb, wrapped_atts


    def forward(self, samples):
        """Performs a forward pass for training, calculating the loss."""
        # Extract data from samples, move image and labels to correct device/dtype
        # Image dtype should match visual encoder's expected dtype
        image = samples["image"].to(self.dtype if hasattr(self, 'dtype') else torch.float32)
        # Labels dtype should be float for BCEWithLogitsLoss
        labels_tensor = samples["labels"].to(self.device, dtype=torch.float)
        # Target text is used for teacher forcing during training
        target_text = samples["target_text"] # List of strings

        # 1. Encode Image Features
        img_embeds, atts_img = self.encode_img(image) # (B, N_img_tok, H), (B, N_img_tok)

        # 2. Wrap Image Features with Prompt Text
        prompt_embeds, prompt_atts = self.prompt_wrap(img_embeds, atts_img) # (B, N_prompt_tok, H), (B, N_prompt_tok)

        # 3. Prepare Target Text for LLM Input (Teacher Forcing)
        self.llama_tokenizer.padding_side = "right" # Ensure padding is on the right for training
        # Add EOS token to target text for training loss calculation
        text_for_loss = [t + self.end_sym for t in target_text]

        # Tokenize target text
        to_regress_tokens = self.llama_tokenizer(
            text_for_loss,
            return_tensors="pt",
            padding="longest", # Pad to the longest sequence in the batch
            truncation=True,   # Truncate if longer than max_length
            max_length=self.max_length - prompt_embeds.shape[1], # Ensure space for prompt
            add_special_tokens=False # We added end_sym manually
        ).to(self.device)

        # Create targets for loss calculation: shift input_ids and mask padding tokens
        # Mask padding tokens with -100 (ignored by CrossEntropyLoss)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.attention_mask == 0, -100
        )

        # --- Combine Prompt and Target for LLM Input ---
        # Prepend -100 labels for the prompt part (we don't calculate loss on prompt tokens)
        prompt_len = prompt_embeds.shape[1]
        empty_targets = torch.full((prompt_embeds.shape[0], prompt_len), -100, dtype=torch.long, device=self.device)
        # Final targets tensor includes prompt masking: [prompt_mask, target_labels]
        targets = torch.cat([empty_targets, targets], dim=1) # (B, N_prompt_tok + N_target_tok)

        # Get embeddings for the target text tokens
        try: text_embeds = self.llama_model.get_input_embeddings()(to_regress_tokens.input_ids)
        except AttributeError: text_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

        # Concatenate prompt embeddings and target text embeddings
        inputs_embeds = torch.cat([prompt_embeds, text_embeds], dim=1) # (B, N_prompt_tok + N_target_tok, H)
        # Concatenate prompt attention mask and target text attention mask
        attention_mask = torch.cat([prompt_atts, to_regress_tokens.attention_mask], dim=1) # (B, N_prompt_tok + N_target_tok)

        # Ensure shapes match before passing to LLM (can mismatch due to truncation)
        current_max_len = inputs_embeds.shape[1]
        if targets.shape[1] > current_max_len:
             targets = targets[:, :current_max_len]
             # print(f"Warning: Truncated targets to match input embeds length: {current_max_len}")
        elif targets.shape[1] < current_max_len:
             pad_len = current_max_len - targets.shape[1]
             padding = torch.full((targets.shape[0], pad_len), -100, dtype=torch.long, device=self.device)
             targets = torch.cat([targets, padding], dim=1)
             # print(f"Warning: Padded targets to match input embeds length: {current_max_len}")

        # 4. Forward Pass through LLM
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets, # Pass labels for automatic loss calculation
        )

        # Extract loss
        loss = outputs.loss

        # Optional: Add custom loss components if needed (e.g., for classification directly)
        # if self.task == 'classification':
        #     # Could add a BCE loss on the generated logits vs labels_tensor here,
        #     # but the primary loss is usually the Causal LM loss.
        #     pass

        return {"loss": loss}

    def training_step(self, batch, batch_idx):
        """Performs a single training step."""
        result = self(batch) # Forward pass calculates loss
        loss = result["loss"]
        batch_size = batch["image"].shape[0] # Get batch size for logging

        # Log training loss
        # sync_dist=True ensures correct logging in multi-GPU setups
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        # Log learning rate
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/learning_rate', lr, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        return loss # Return loss for optimization

    def generate_step(self, batch):
         """Generates text for a given batch using the LLM."""
         # Ensure model is in evaluation mode for generation
         self.eval()

         # Prepare image input
         # Image dtype should match visual encoder
         image = batch["image"].to(self.dtype if hasattr(self, 'dtype') else torch.float32)

         # Encode images and wrap with prompts
         # Ensure generation happens on the correct device
         with torch.no_grad(): # Disable gradient calculation for generation
            img_embeds, atts_img = self.encode_img(image)
            prompt_embeds, prompt_atts = self.prompt_wrap(img_embeds, atts_img) # (B, N_prompt_tok, H), (B, N_prompt_tok)

            # Define EOS token IDs for stopping generation
            # Llama 3 uses <|eot_id|> and potentially others
            eos_token_id = self.llama_tokenizer.eos_token_id # Usually the primary one
            eot_id = self.llama_tokenizer.encode("<|eot_id|>", add_special_tokens=False) # Get ID for <|eot_id|>
            terminate_id = self.llama_tokenizer.encode("<|terminate|>", add_special_tokens=False) # Get ID for <|terminate|>

            eos_token_ids = [eos_token_id]
            if eot_id: eos_token_ids.append(eot_id[0]) # Add if found
            if terminate_id: eos_token_ids.append(terminate_id[0]) # Add if found
            eos_token_ids = list(set(eos_token_ids)) # Remove duplicates

            # print(f"Using EOS token IDs for generation: {eos_token_ids}")

            # Set tokenizer padding side for generation (usually left, but check model reqs)
            # self.llama_tokenizer.padding_side = "left" # Important for batch generation

            # Generate text using Hugging Face's generate method
            outputs = self.llama_model.generate(
                inputs_embeds=prompt_embeds,          # Use embeddings as input
                attention_mask=prompt_atts,           # Corresponding attention mask
                num_beams=self.num_beams,             # Beam search size
                do_sample=self.do_sample,             # Whether to use sampling
                max_new_tokens=self.max_new_tokens,   # Max tokens to generate *after* prompt
                temperature=self.temperature,         # Controls randomness (if do_sample=True)
                repetition_penalty=self.repetition_penalty, # Penalizes repeated tokens
                length_penalty=self.length_penalty,   # Penalizes longer/shorter sequences
                pad_token_id=self.llama_tokenizer.pad_token_id, # Ensure pad token ID is set
                eos_token_id=eos_token_ids            # Token(s) to indicate end of sequence
            )

            # Decode generated token IDs (excluding the prompt part)
            # The generate output usually includes input_ids, so slice them off
            # However, with inputs_embeds, the output might only contain new tokens. Check HF docs.
            # Assuming `outputs` contains only generated tokens *after* the prompt:
            generated_texts = [self.decode(output_ids) for output_ids in outputs]

            # If `outputs` includes prompt tokens, slice them:
            # input_len = prompt_embeds.shape[1]
            # generated_ids = outputs[:, input_len:]
            # generated_texts = [self.decode(output_ids) for output_ids in generated_ids]


         # Set model back to train mode if necessary (PL handles this usually)
         # self.train()

         return generated_texts


    def validation_step(self, batch, batch_idx):
        """Performs a validation step: generate text and store outputs."""
        generated_texts = self.generate_step(batch) # Generate predictions
        ids = batch['id'] # Sample identifiers
        refs = batch['ref'] # Ground truth text (report or label string)

        # Store results for epoch-end calculation
        batch_output = {"hypo": generated_texts, "ref": refs, "id": ids}

        # Optionally save images for qualitative analysis
        if hasattr(self.hparams, 'save_images') and self.hparams.save_images:
            # Move image tensor to CPU before storing to avoid GPU memory buildup
            batch_output["image"] = batch["image"].cpu()

        self.val_step_outputs.append(batch_output)
        return batch_output # Return value isn't strictly needed by PL unless used in callbacks

    def on_validation_epoch_end(self):
        """Calculates and logs validation metrics at the end of the epoch."""
        # Check if validation outputs exist and if we are on the main process (rank 0)
        if not self.val_step_outputs or not self.trainer.is_global_zero:
             if hasattr(self, 'val_step_outputs'): self.val_step_outputs.clear() # Clear list on non-main processes
             return # Only rank 0 performs evaluation and logging

        print(f"\nProcessing validation results for epoch {self.current_epoch}...")

        # Aggregate outputs from all validation steps
        all_hypo, all_ref, all_ids, all_images = [], [], [], []
        for output in self.val_step_outputs:
            all_hypo.extend(output.get('hypo', []))
            all_ref.extend(output.get('ref', []))
            all_ids.extend(output.get('id', []))
            if "image" in output: all_images.extend(output.get('image', []))

        # Format for scoring functions (dict of id: [text])
        hypo_dict = {k: [v] for k, v in zip(all_ids, all_hypo)}
        ref_dict = {k: [v] for k, v in zip(all_ids, all_ref)}

        # Calculate metrics
        eval_res = self.score(ref=ref_dict, hypo=hypo_dict)

        # Handle empty evaluation results
        if not eval_res:
            print("Warning: Validation scoring returned empty results.")
            self.val_step_outputs.clear() # Clear stored outputs
            return

        # Log metrics using self.log
        log_prefix = 'val/'
        # Define the primary metric for checkpointing and progress bar
        primary_metric_key = 'F1_mac' if self.task == 'classification' else 'CIDEr'
        if primary_metric_key not in eval_res:
             print(f"Warning: Primary validation metric '{primary_metric_key}' not found in results. Using 0.0.")
             primary_score = 0.0
        else: primary_score = eval_res[primary_metric_key]

        # Log the primary score with a distinct name for checkpointing
        self.log(f'{log_prefix}{primary_metric_key}_primary', primary_score, on_epoch=True, prog_bar=True, logger=True, sync_dist=False) # Rank 0 already

        # Log other relevant metrics
        metrics_to_log = ['F1_mic', 'P_mac', 'R_mac', 'Acc_subset'] if self.task == 'classification' else ['Bleu_4', 'ROUGE_L']
        for metric in metrics_to_log:
             if metric in eval_res:
                 self.log(f'{log_prefix}{metric}', eval_res[metric], on_epoch=True, prog_bar=False, logger=True, sync_dist=False)

        # Print results to console
        print(f"--- Validation Epoch {self.current_epoch} Results ---")
        for metric, score in eval_res.items():
            if metric != 'Per_Class': # Don't print the huge per-class dict here
                print(f"  {metric}: {score:.4f}")
            elif self.task == 'classification' and score: # Print summary of per-class if classification
                 print("  Per-Class F1 Scores (Macro):")
                 f1_scores = [cls_metrics.get('F1', 0) for cls_metrics in score.values()]
                 print(f"    Mean: {np.mean(f1_scores):.4f}, Std: {np.std(f1_scores):.4f}")
                 # Optional: print a few classes
                 # for disease, metrics in list(score.items())[:3]: print(f"    {disease}: {metrics.get('F1', 0):.4f}")

        print("-------------------------------------\n")

        # Save best model weights based on primary score (using .pth for adapters/projection)
        if primary_score > self.best_val_score:
             self.best_val_score = primary_score
             print(f"New best validation score ({primary_metric_key}): {primary_score:.4f}")
             # Define save directory and filename
             pth_save_dir = os.path.join(self.hparams.savedmodel_path, 'pth_weights')
             os.makedirs(pth_save_dir, exist_ok=True)
             # Use a consistent filename or include score/epoch
             pth_filename = getattr(self.hparams, 'pth_save_filename', f'best_model_{primary_metric_key}_{primary_score:.4f}_epoch{self.current_epoch}.pth')
             # Make filename safe
             pth_filename = pth_filename.replace("=","_").replace("/","_").replace("\\","_")
             pth_save_path = os.path.join(pth_save_dir, pth_filename)

             try:
                state_to_save = {}
                model_to_save = self # The LightningModule instance

                # Save LLM PEFT adapter weights if used
                if isinstance(model_to_save.llama_model, PeftModel):
                     print("Saving LLM PEFT adapter state...")
                     # Get state dict, keys might include 'base_model.model...' prefix
                     llm_state = model_to_save.llama_model.state_dict()
                     # Filter for adapter weights only if needed, or save the whole PEFT state
                     # state_to_save.update({f"llama_model.{k}": v for k, v in llm_state.items()}) # Save with prefix
                     state_to_save.update(llm_state) # Save directly, assumes loader handles prefixes

                # Save Vision PEFT adapter weights if used
                if isinstance(model_to_save.visual_encoder, PeftModel):
                     print("Saving Vision PEFT adapter state...")
                     vis_state = model_to_save.visual_encoder.state_dict()
                     # state_to_save.update({f"visual_encoder.{k}": v for k, v in vis_state.items()}) # Save with prefix
                     state_to_save.update(vis_state) # Save directly

                # Save projection layer weights
                if hasattr(self, 'llama_proj'):
                    state_to_save['llama_proj.weight'] = self.llama_proj.weight
                    if self.llama_proj.bias is not None: state_to_save['llama_proj.bias'] = self.llama_proj.bias
                if hasattr(self, 'projection_layer_norm'):
                    state_to_save['projection_layer_norm.weight'] = self.projection_layer_norm.weight
                    state_to_save['projection_layer_norm.bias'] = self.projection_layer_norm.bias

                # Check if anything was added to save
                if not state_to_save:
                     warnings.warn("No PEFT adapters or projection layers found to save in .pth file. Saving full model state_dict might be very large.")
                     # Optionally save the full state dict if no adapters are used
                     # state_to_save = self.state_dict()

                if state_to_save:
                    torch.save(state_to_save, pth_save_path)
                    print(f"Saved model components (adapters/projection) to: {pth_save_path}")
                else:
                    print("Nothing specific to save found (check LoRA/projection layers).")

             except Exception as e: print(f"Error saving .pth weights: {e}")

        # Save example images and text if enabled
        if hasattr(self.hparams, 'save_images') and self.hparams.save_images and all_images:
             self.save_batch_examples(
                 {"image": all_images, "ref": all_ref, "hypo": all_hypo},
                 "val", f'epoch_{self.current_epoch}'
             )

        # Clear stored outputs for the next epoch
        self.val_step_outputs.clear()


    def test_step(self, batch, batch_idx):
        """Performs a test step: generate text and store outputs."""
        generated_texts = self.generate_step(batch) # Generate predictions
        ids = batch['id'] # Sample identifiers
        refs = batch['ref'] # Ground truth text

        # Store results for epoch-end calculation
        batch_output = {"hypo": generated_texts, "ref": refs, "id": ids}
        if hasattr(self.hparams, 'save_images') and self.hparams.save_images:
            batch_output["image"] = batch["image"].cpu() # Move to CPU

        self.test_step_outputs.append(batch_output)
        # print(f"Test Step - ID: {ids[0]}, Hypo: {generated_texts[0][:50]}..., Ref: {refs[0][:50]}...") # Debug print one sample
        return batch_output

    def on_test_epoch_end(self):
        """Calculates, logs, and saves test metrics at the end of testing."""
        # Check if test outputs exist and if we are on the main process
        # *** CORRECTED LINE BELOW ***
        if not self.test_step_outputs or not self.trainer.is_global_zero:
            if hasattr(self, 'test_step_outputs'): self.test_step_outputs.clear()
            return # Only rank 0 performs evaluation

        print("\nProcessing test results...")
        # Aggregate outputs
        all_hypo, all_ref, all_ids, all_images = [], [], [], []
        for output in self.test_step_outputs:
            all_hypo.extend(output.get('hypo', []))
            all_ref.extend(output.get('ref', []))
            all_ids.extend(output.get('id', []))
            if "image" in output: all_images.extend(output.get('image', []))

        # Format for scoring
        hypo_dict = {k: [v] for k, v in zip(all_ids, all_hypo)}
        ref_dict = {k: [v] for k, v in zip(all_ids, all_ref)}

        # Calculate metrics
        eval_res = self.score(ref=ref_dict, hypo=hypo_dict)

        # Handle empty results
        if not eval_res:
            print("Warning: Test scoring returned empty results.")
            self.test_step_outputs.clear()
            return

        # Log test metrics using self.log (prefixed with 'test/')
        log_prefix = 'test/'
        print("\n--- Test Results ---")
        for metric, score in eval_res.items():
            if metric != 'Per_Class': # Avoid printing the large dict directly
                 print(f"  {metric}: {score:.4f}")
                 self.log(f'{log_prefix}{metric}', score, logger=True, sync_dist=False) # Log main metrics

        # Print and log per-class metrics if classification task
        if self.task == 'classification' and 'Per_Class' in eval_res and eval_res['Per_Class']:
             print("\n  Per-Class Metrics (Test):")
             per_class_results = eval_res['Per_Class']
             for disease, metrics in per_class_results.items():
                 f1 = metrics.get('F1', 0)
                 p = metrics.get('P', 0)
                 r = metrics.get('R', 0)
                 acc = metrics.get('Acc', 0)
                 print(f"    {disease:<28}: F1={f1:.3f} (P={p:.3f}, R={r:.3f}, Acc={acc:.3f}, Supp={metrics.get('Support',0)})")
                 # Log per-class F1 scores
                 self.log(f'{log_prefix}F1_{disease.replace(" ", "_")}', f1, logger=True, sync_dist=False)
        print("--------------------\n")

        # Save test results (hypotheses, references, metrics) to JSON files
        self._save_test_results(hypo_dict, ref_dict, eval_res)

        # Save example images and text if enabled
        if hasattr(self.hparams, 'save_images') and self.hparams.save_images and all_images:
            self.save_batch_examples(
                {"image": all_images, "ref": all_ref, "hypo": all_hypo},
                "test", "final_results" # Subfolder name for test examples
            )

        # Clear stored outputs
        self.test_step_outputs.clear()


    def save_batch_examples(self, batch_data, split_name, epoch_str):
         """Saves a few examples (image, ref text, hypo text) for qualitative analysis."""
         # Only save on rank 0
         if not self.trainer.is_global_zero: return

         # Define the directory to save images and texts
         img_folder = os.path.join(self.hparams.savedmodel_path, 'qualitative_examples', split_name, epoch_str)
         os.makedirs(img_folder, exist_ok=True)

         # Extract data, handle potential missing keys gracefully
         imgs = batch_data.get("image", [])
         refs = batch_data.get("ref", [])
         hypos = batch_data.get("hypo", [])
         ids = batch_data.get("id", [f"sample_{i}" for i in range(len(imgs))]) # Generate default IDs if missing

         # Determine number of samples to save (e.g., first 10)
         num_save = min(len(imgs), getattr(self.hparams, 'num_save_examples', 10))
         if num_save == 0: return # Don't save if num_save is 0

         print(f"Saving {num_save} qualitative examples to {img_folder}...")
         for idx in range(num_save):
            try:
                img_tensor = imgs[idx]
                ref_txt = refs[idx] if idx < len(refs) else "N/A"
                hypo_txt = hypos[idx] if idx < len(hypos) else "N/A"
                sample_id = ids[idx] if idx < len(ids) else f"sample_{idx}"
                # Sanitize ID for filename
                safe_id = "".join(c if c.isalnum() else "_" for c in str(sample_id))

                img_path = os.path.join(img_folder, f"{safe_id}_image.png")
                txt_path = os.path.join(img_folder, f"{safe_id}_text.txt")

                # Save image using torchvision.utils.save_image
                # Normalize image tensor to [0, 1] range for saving if needed
                # Assumes input tensor might be normalized differently
                img_save = img_tensor.float().cpu() # Ensure float and on CPU
                img_min, img_max = img_save.min(), img_save.max()
                if img_max > img_min: # Avoid division by zero
                    img_save = (img_save - img_min) / (img_max - img_min + 1e-8)
                else: img_save = torch.zeros_like(img_save) # Handle constant image case

                torchvision.utils.save_image(img_save, img_path)

                # Save reference and generated text
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"--- Sample ID: {sample_id} ---\n\n")
                    f.write("--- Reference Text ---\n")
                    f.write(f"{ref_txt}\n\n")
                    f.write("--- Generated Text ---\n")
                    f.write(f"{hypo_txt}\n")

            except Exception as e: print(f"Error saving qualitative example {idx} (ID: {sample_id}): {e}")


    def _save_test_results(self, hypo, ref, eval_res):
        """Saves test hypotheses, references, and overall metrics to JSON files."""
        # Only save on rank 0
        if not self.trainer.is_global_zero: return

        # Define the folder for saving results
        res_folder = os.path.join(self.hparams.savedmodel_path, 'test_results')
        os.makedirs(res_folder, exist_ok=True)

        try:
             # Determine a base filename from checkpoint or delta file used for testing
             ckpt_name = "final" # Default name
             if hasattr(self.hparams, 'delta_file') and self.hparams.delta_file and os.path.exists(self.hparams.delta_file):
                 ckpt_name = os.path.splitext(os.path.basename(self.hparams.delta_file))[0]
             elif hasattr(self.hparams, 'ckpt_file') and self.hparams.ckpt_file and os.path.exists(self.hparams.ckpt_file):
                 ckpt_name = os.path.splitext(os.path.basename(self.hparams.ckpt_file))[0]
             # Sanitize ckpt_name for filename
             safe_ckpt_name = "".join(c if c.isalnum() else "_" for c in ckpt_name)


             # Define file paths
             paths = {
                 "hypo": os.path.join(res_folder, f"test_hypotheses_{safe_ckpt_name}.json"),
                 "ref": os.path.join(res_folder, f"test_references_{safe_ckpt_name}.json"),
                 "metrics": os.path.join(res_folder, f"test_metrics_{safe_ckpt_name}.json")
             }

             # Save dictionaries to JSON files with indentation
             print(f"Saving test results to folder: {res_folder}")
             with open(paths["hypo"], 'w', encoding='utf-8') as f_hypo:
                 json.dump(hypo, f_hypo, indent=2, ensure_ascii=False)
             print(f"  Saved hypotheses to: {paths['hypo']}")

             with open(paths["ref"], 'w', encoding='utf-8') as f_ref:
                 json.dump(ref, f_ref, indent=2, ensure_ascii=False)
             print(f"  Saved references to: {paths['ref']}")

             # Save metrics dictionary
             # Need custom handler for numpy types if present in metrics
             class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer): return int(obj)
                    elif isinstance(obj, np.floating): return float(obj)
                    elif isinstance(obj, np.ndarray): return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)

             with open(paths["metrics"], 'w', encoding='utf-8') as f_metrics:
                 json.dump(eval_res, f_metrics, indent=2, ensure_ascii=False, cls=NumpyEncoder)
             print(f"  Saved metrics to: {paths['metrics']}")

        except Exception as e: print(f"Error saving test results: {e}")

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        # Filter parameters that require gradients
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        if not trainable_params:
            print("Warning: No trainable parameters found! Optimizer will not be configured.")
            return None # Return None if no parameters to optimize

        print(f"Configuring AdamW optimizer with lr={self.learning_rate} for {len(trainable_params)} trainable parameters.")
        # Use AdamW optimizer (common choice)
        optimizer = torch.optim.AdamW(trainable_params, lr=self.learning_rate)

        # Configure learning rate scheduler (e.g., Cosine Annealing)
        # T_max is often set to the total number of training steps or epochs
        max_epochs = getattr(self.trainer, 'max_epochs', getattr(self.hparams, 'max_epochs', 1))
        # Calculate total steps if scheduler interval is 'step'
        # total_steps = self.trainer.estimated_stepping_batches

        if max_epochs <= 0: max_epochs=1 # Avoid T_max=0

        print(f"Using CosineAnnealingLR scheduler with T_max={max_epochs} epochs.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=max_epochs, # Number of epochs or steps
            eta_min=getattr(self.hparams, 'lr_eta_min', 1e-6) # Minimum learning rate
        )

        # Return optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # 'step' or 'epoch'
                "frequency": 1,       # How often the scheduler updates
                "monitor": "val/loss" # Optional: Monitor a metric for ReduceLROnPlateau
            },
        }

    def get_progress_bar_dict(self):
        # Remove 'v_num' from the progress bar for cleaner output
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items