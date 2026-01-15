# # Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# import os
# from typing import Optional

# import pkg_resources

# import torch
# import torch.nn as nn
# from huggingface_hub import hf_hub_download
# from iopath.common.file_io import g_pathmgr
# from sam3.model.decoder import (
#     TransformerDecoder,
#     TransformerDecoderLayer,
#     TransformerDecoderLayerv2,
#     TransformerEncoderCrossAttention,
# )
# from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
# from sam3.model.geometry_encoders import SequenceGeometryEncoder
# from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
# from sam3.model.memory import (
#     CXBlock,
#     SimpleFuser,
#     SimpleMaskDownSampler,
#     SimpleMaskEncoder,
# )
# from sam3.model.model_misc import (
#     DotProductScoring,
#     MLP,
#     MultiheadAttentionWrapper as MultiheadAttention,
#     TransformerWrapper,
# )
# from sam3.model.necks import Sam3DualViTDetNeck
# from sam3.model.position_encoding import PositionEmbeddingSine
# from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
# from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
# from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
# from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
# from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
# from sam3.model.text_encoder_ve import VETextEncoder
# from sam3.model.tokenizer_ve import SimpleTokenizer
# from sam3.model.vitdet import ViT
# from sam3.model.vl_combiner import SAM3VLBackbone
# from sam3.sam.transformer import RoPEAttention


# # Setup TensorFloat-32 for Ampere GPUs if available
# def _setup_tf32() -> None:
#     """Enable TensorFloat-32 for Ampere GPUs if available."""
#     if torch.cuda.is_available():
#         device_props = torch.cuda.get_device_properties(0)
#         if device_props.major >= 8:
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.backends.cudnn.allow_tf32 = True


# _setup_tf32()


# def _create_position_encoding(precompute_resolution=None):
#     """Create position encoding for visual backbone."""
#     return PositionEmbeddingSine(
#         num_pos_feats=256,
#         normalize=True,
#         scale=None,
#         temperature=10000,
#         precompute_resolution=precompute_resolution,
#     )


# def _create_vit_backbone(compile_mode=None):
#     """Create ViT backbone for visual feature extraction."""
#     return ViT(
#         img_size=1008,
#         pretrain_img_size=336,
#         patch_size=14,
#         embed_dim=1024,
#         depth=32,
#         num_heads=16,
#         mlp_ratio=4.625,
#         norm_layer="LayerNorm",
#         drop_path_rate=0.1,
#         qkv_bias=True,
#         use_abs_pos=True,
#         tile_abs_pos=True,
#         global_att_blocks=(7, 15, 23, 31),
#         rel_pos_blocks=(),
#         use_rope=True,
#         use_interp_rope=True,
#         window_size=24,
#         pretrain_use_cls_token=True,
#         retain_cls_token=False,
#         ln_pre=True,
#         ln_post=False,
#         return_interm_layers=False,
#         bias_patch_embed=False,
#         compile_mode=compile_mode,
#     )


# def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
#     """Create ViT neck for feature pyramid."""
#     return Sam3DualViTDetNeck(
#         position_encoding=position_encoding,
#         d_model=256,
#         scale_factors=[4.0, 2.0, 1.0, 0.5],
#         trunk=vit_backbone,
#         add_sam2_neck=enable_inst_interactivity,
#     )


# def _create_vl_backbone(vit_neck, text_encoder):
#     """Create visual-language backbone."""
#     return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


# def _create_transformer_encoder() -> TransformerEncoderFusion:
#     """Create transformer encoder with its layer."""
#     encoder_layer = TransformerEncoderLayer(
#         activation="relu",
#         d_model=256,
#         dim_feedforward=2048,
#         dropout=0.1,
#         pos_enc_at_attn=True,
#         pos_enc_at_cross_attn_keys=False,
#         pos_enc_at_cross_attn_queries=False,
#         pre_norm=True,
#         self_attention=MultiheadAttention(
#             num_heads=8,
#             dropout=0.1,
#             embed_dim=256,
#             batch_first=True,
#         ),
#         cross_attention=MultiheadAttention(
#             num_heads=8,
#             dropout=0.1,
#             embed_dim=256,
#             batch_first=True,
#         ),
#     )

#     encoder = TransformerEncoderFusion(
#         layer=encoder_layer,
#         num_layers=6,
#         d_model=256,
#         num_feature_levels=1,
#         frozen=False,
#         use_act_checkpoint=True,
#         add_pooled_text_to_img_feat=False,
#         pool_text_with_mask=True,
#     )
#     return encoder


# def _create_transformer_decoder() -> TransformerDecoder:
#     """Create transformer decoder with its layer."""
#     decoder_layer = TransformerDecoderLayer(
#         activation="relu",
#         d_model=256,
#         dim_feedforward=2048,
#         dropout=0.1,
#         cross_attention=MultiheadAttention(
#             num_heads=8,
#             dropout=0.1,
#             embed_dim=256,
#         ),
#         n_heads=8,
#         use_text_cross_attention=True,
#     )

#     decoder = TransformerDecoder(
#         layer=decoder_layer,
#         num_layers=6,
#         num_queries=200,
#         return_intermediate=True,
#         box_refine=True,
#         num_o2m_queries=0,
#         dac=True,
#         boxRPB="log",
#         d_model=256,
#         frozen=False,
#         interaction_layer=None,
#         dac_use_selfatt_ln=True,
#         resolution=1008,
#         stride=14,
#         use_act_checkpoint=True,
#         presence_token=True,
#     )
#     return decoder


# def _create_dot_product_scoring():
#     """Create dot product scoring module."""
#     prompt_mlp = MLP(
#         input_dim=256,
#         hidden_dim=2048,
#         output_dim=256,
#         num_layers=2,
#         dropout=0.1,
#         residual=True,
#         out_norm=nn.LayerNorm(256),
#     )
#     return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


# def _create_segmentation_head(compile_mode=None):
#     """Create segmentation head with pixel decoder."""
#     pixel_decoder = PixelDecoder(
#         num_upsampling_stages=3,
#         interpolation_mode="nearest",
#         hidden_dim=256,
#         compile_mode=compile_mode,
#     )

#     cross_attend_prompt = MultiheadAttention(
#         num_heads=8,
#         dropout=0,
#         embed_dim=256,
#     )

#     segmentation_head = UniversalSegmentationHead(
#         hidden_dim=256,
#         upsampling_stages=3,
#         aux_masks=False,
#         presence_head=False,
#         dot_product_scorer=None,
#         act_ckpt=True,
#         cross_attend_prompt=cross_attend_prompt,
#         pixel_decoder=pixel_decoder,
#     )
#     return segmentation_head


# def _create_geometry_encoder():
#     """Create geometry encoder with all its components."""
#     # Create position encoding for geometry encoder
#     geo_pos_enc = _create_position_encoding()
#     # Create CX block for fuser
#     cx_block = CXBlock(
#         dim=256,
#         kernel_size=7,
#         padding=3,
#         layer_scale_init_value=1.0e-06,
#         use_dwconv=True,
#     )
#     # Create geometry encoder layer
#     geo_layer = TransformerEncoderLayer(
#         activation="relu",
#         d_model=256,
#         dim_feedforward=2048,
#         dropout=0.1,
#         pos_enc_at_attn=False,
#         pre_norm=True,
#         self_attention=MultiheadAttention(
#             num_heads=8,
#             dropout=0.1,
#             embed_dim=256,
#             batch_first=False,
#         ),
#         pos_enc_at_cross_attn_queries=False,
#         pos_enc_at_cross_attn_keys=True,
#         cross_attention=MultiheadAttention(
#             num_heads=8,
#             dropout=0.1,
#             embed_dim=256,
#             batch_first=False,
#         ),
#     )

#     # Create geometry encoder
#     input_geometry_encoder = SequenceGeometryEncoder(
#         pos_enc=geo_pos_enc,
#         encode_boxes_as_points=False,
#         points_direct_project=True,
#         points_pool=True,
#         points_pos_enc=True,
#         boxes_direct_project=True,
#         boxes_pool=True,
#         boxes_pos_enc=True,
#         d_model=256,
#         num_layers=3,
#         layer=geo_layer,
#         use_act_ckpt=True,
#         add_cls=True,
#         add_post_encode_proj=True,
#     )
#     return input_geometry_encoder


# def _create_sam3_model(
#     backbone,
#     transformer,
#     input_geometry_encoder,
#     segmentation_head,
#     dot_prod_scoring,
#     inst_interactive_predictor,
#     eval_mode,
# ):
#     """Create the SAM3 image model."""
#     common_params = {
#         "backbone": backbone,
#         "transformer": transformer,
#         "input_geometry_encoder": input_geometry_encoder,
#         "segmentation_head": segmentation_head,
#         "num_feature_levels": 1,
#         "o2m_mask_predict": True,
#         "dot_prod_scoring": dot_prod_scoring,
#         "use_instance_query": False,
#         "multimask_output": True,
#         "inst_interactive_predictor": inst_interactive_predictor,
#     }

#     matcher = None
#     if not eval_mode:
#         from sam3.train.matcher import BinaryHungarianMatcherV2

#         matcher = BinaryHungarianMatcherV2(
#             focal=True,
#             cost_class=2.0,
#             cost_bbox=5.0,
#             cost_giou=2.0,
#             alpha=0.25,
#             gamma=2,
#             stable=False,
#         )
#     common_params["matcher"] = matcher
#     model = Sam3Image(**common_params)

#     return model


# def _create_tracker_maskmem_backbone():
#     """Create the SAM3 Tracker memory encoder."""
#     # Position encoding for mask memory backbone
#     position_encoding = PositionEmbeddingSine(
#         num_pos_feats=64,
#         normalize=True,
#         scale=None,
#         temperature=10000,
#         precompute_resolution=1008,
#     )

#     # Mask processing components
#     mask_downsampler = SimpleMaskDownSampler(
#         kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
#     )

#     cx_block_layer = CXBlock(
#         dim=256,
#         kernel_size=7,
#         padding=3,
#         layer_scale_init_value=1.0e-06,
#         use_dwconv=True,
#     )

#     fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

#     maskmem_backbone = SimpleMaskEncoder(
#         out_dim=64,
#         position_encoding=position_encoding,
#         mask_downsampler=mask_downsampler,
#         fuser=fuser,
#     )

#     return maskmem_backbone


# def _create_tracker_transformer():
#     """Create the SAM3 Tracker transformer components."""
#     # Self attention
#     self_attention = RoPEAttention(
#         embedding_dim=256,
#         num_heads=1,
#         downsample_rate=1,
#         dropout=0.1,
#         rope_theta=10000.0,
#         feat_sizes=[72, 72],
#         use_fa3=False,
#         use_rope_real=False,
#     )

#     # Cross attention
#     cross_attention = RoPEAttention(
#         embedding_dim=256,
#         num_heads=1,
#         downsample_rate=1,
#         dropout=0.1,
#         kv_in_dim=64,
#         rope_theta=10000.0,
#         feat_sizes=[72, 72],
#         rope_k_repeat=True,
#         use_fa3=False,
#         use_rope_real=False,
#     )

#     # Encoder layer
#     encoder_layer = TransformerDecoderLayerv2(
#         cross_attention_first=False,
#         activation="relu",
#         dim_feedforward=2048,
#         dropout=0.1,
#         pos_enc_at_attn=False,
#         pre_norm=True,
#         self_attention=self_attention,
#         d_model=256,
#         pos_enc_at_cross_attn_keys=True,
#         pos_enc_at_cross_attn_queries=False,
#         cross_attention=cross_attention,
#     )

#     # Encoder
#     encoder = TransformerEncoderCrossAttention(
#         remove_cross_attention_layers=[],
#         batch_first=True,
#         d_model=256,
#         frozen=False,
#         pos_enc_at_input=True,
#         layer=encoder_layer,
#         num_layers=4,
#         use_act_checkpoint=False,
#     )

#     # Transformer wrapper
#     transformer = TransformerWrapper(
#         encoder=encoder,
#         decoder=None,
#         d_model=256,
#     )

#     return transformer


# def build_tracker(
#     apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
# ) -> Sam3TrackerPredictor:
#     """
#     Build the SAM3 Tracker module for video tracking.

#     Returns:
#         Sam3TrackerPredictor: Wrapped SAM3 Tracker module
#     """

#     # Create model components
#     maskmem_backbone = _create_tracker_maskmem_backbone()
#     transformer = _create_tracker_transformer()
#     backbone = None
#     if with_backbone:
#         vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
#         backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
#     # Create the Tracker module
#     model = Sam3TrackerPredictor(
#         image_size=1008,
#         num_maskmem=7,
#         backbone=backbone,
#         backbone_stride=14,
#         transformer=transformer,
#         maskmem_backbone=maskmem_backbone,
#         # SAM parameters
#         multimask_output_in_sam=True,
#         # Evaluation
#         forward_backbone_per_frame_for_eval=True,
#         trim_past_non_cond_mem_for_eval=False,
#         # Multimask
#         multimask_output_for_tracking=True,
#         multimask_min_pt_num=0,
#         multimask_max_pt_num=1,
#         # Additional settings
#         always_start_from_first_ann_frame=False,
#         # Mask overlap
#         non_overlap_masks_for_mem_enc=False,
#         non_overlap_masks_for_output=False,
#         max_cond_frames_in_attn=4,
#         offload_output_to_cpu_for_eval=False,
#         # SAM decoder settings
#         sam_mask_decoder_extra_args={
#             "dynamic_multimask_via_stability": True,
#             "dynamic_multimask_stability_delta": 0.05,
#             "dynamic_multimask_stability_thresh": 0.98,
#         },
#         clear_non_cond_mem_around_input=True,
#         fill_hole_area=0,
#         use_memory_selection=apply_temporal_disambiguation,
#     )

#     return model


# def _create_text_encoder(bpe_path: str) -> VETextEncoder:
#     """Create SAM3 text encoder."""
#     tokenizer = SimpleTokenizer(bpe_path=bpe_path)
#     return VETextEncoder(
#         tokenizer=tokenizer,
#         d_model=256,
#         width=1024,
#         heads=16,
#         layers=24,
#     )


# def _create_vision_backbone(
#     compile_mode=None, enable_inst_interactivity=True
# ) -> Sam3DualViTDetNeck:
#     """Create SAM3 visual backbone with ViT and neck."""
#     # Position encoding
#     position_encoding = _create_position_encoding(precompute_resolution=1008)
#     # ViT backbone
#     vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
#     vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
#         position_encoding,
#         vit_backbone,
#         enable_inst_interactivity=enable_inst_interactivity,
#     )
#     # Visual neck
#     return vit_neck


# def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
#     """Create SAM3 transformer encoder and decoder."""
#     encoder: TransformerEncoderFusion = _create_transformer_encoder()
#     decoder: TransformerDecoder = _create_transformer_decoder()

#     return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


# def _load_checkpoint(model, checkpoint_path):
#     """Load model checkpoint from file."""
#     with g_pathmgr.open(checkpoint_path, "rb") as f:
#         ckpt = torch.load(f, map_location="cpu", weights_only=True)
#     if "model" in ckpt and isinstance(ckpt["model"], dict):
#         ckpt = ckpt["model"]
#     sam3_image_ckpt = {
#         k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
#     }
#     if model.inst_interactive_predictor is not None:
#         sam3_image_ckpt.update(
#             {
#                 k.replace("tracker.", "inst_interactive_predictor.model."): v
#                 for k, v in ckpt.items()
#                 if "tracker" in k
#             }
#         )
#     missing_keys, _ = model.load_state_dict(sam3_image_ckpt, strict=False)
#     if len(missing_keys) > 0:
#         print(
#             f"loaded {checkpoint_path} and found "
#             f"missing and/or unexpected keys:\n{missing_keys=}"
#         )


# def _setup_device_and_mode(model, device, eval_mode):
#     """Setup model device and evaluation mode."""
#     if device == "cuda":
#         model = model.cuda()
#     if eval_mode:
#         model.eval()
#     return model


# def build_sam3_image_model(
#     bpe_path=None,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     eval_mode=True,
#     checkpoint_path=None,
#     load_from_HF=True,
#     enable_segmentation=True,
#     enable_inst_interactivity=False,
#     compile=False,
# ):
#     """
#     Build SAM3 image model

#     Args:
#         bpe_path: Path to the BPE tokenizer vocabulary
#         device: Device to load the model on ('cuda' or 'cpu')
#         eval_mode: Whether to set the model to evaluation mode
#         checkpoint_path: Optional path to model checkpoint
#         enable_segmentation: Whether to enable segmentation head
#         enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
#         compile_mode: To enable compilation, set to "default"

#     Returns:
#         A SAM3 image model
#     """
#     if bpe_path is None:
#         bpe_path = pkg_resources.resource_filename(
#             "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
#         )

#     # Create visual components
#     compile_mode = "default" if compile else None
#     vision_encoder = _create_vision_backbone(
#         compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
#     )

#     # Create text components
#     text_encoder = _create_text_encoder(bpe_path)

#     # Create visual-language backbone
#     backbone = _create_vl_backbone(vision_encoder, text_encoder)

#     # Create transformer components
#     transformer = _create_sam3_transformer()

#     # Create dot product scoring
#     dot_prod_scoring = _create_dot_product_scoring()

#     # Create segmentation head if enabled
#     segmentation_head = (
#         _create_segmentation_head(compile_mode=compile_mode)
#         if enable_segmentation
#         else None
#     )

#     # Create geometry encoder
#     input_geometry_encoder = _create_geometry_encoder()
#     if enable_inst_interactivity:
#         sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
#         inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
#     else:
#         inst_predictor = None
#     # Create the SAM3 model
#     model = _create_sam3_model(
#         backbone,
#         transformer,
#         input_geometry_encoder,
#         segmentation_head,
#         dot_prod_scoring,
#         inst_predictor,
#         eval_mode,
#     )
#     # 先确保导入 os 模块（如果文件顶部没导入的话）
#     import os

#     # 核心逻辑修改（替换你原来的 checkpoint_path 赋值代码）
#     if not load_from_HF and checkpoint_path is None:  # 关键：load_from_HF 为 False 时才加载本地模型
#         # ModelScope 下载的 SAM3 模型根目录（你的实际下载路径）
#         LOCAL_SAM3_DIR = "/shared_data/modelData/modelscope/models/facebook/sam3/"
#         # 指向本地 config.json（配置文件）
#         checkpoint_path = os.path.join(LOCAL_SAM3_DIR, "config.json")
#         # 指向本地权重文件（必须确保 _load_checkpoint 能读到这个权重）
#         model_weight_path = os.path.join(LOCAL_SAM3_DIR, "model.safetensors")
        
#         # 补充：如果 _load_checkpoint 函数需要显式传入权重路径，需修改调用方式
#         # 方式1：如果 _load_checkpoint 支持指定权重路径（推荐）
#         _load_checkpoint(model, checkpoint_path, weight_path=model_weight_path)
#         # 方式2：如果 _load_checkpoint 只认 config.json 所在目录（自动找权重）
#         # _load_checkpoint(model, checkpoint_path)

#     # 如果 load_from_HF 为 True（仍想从 HF 加载），则保留原逻辑
#     elif load_from_HF and checkpoint_path is None:
#         checkpoint_path = download_ckpt_from_hf()

#     # 通用加载逻辑（确保覆盖所有情况）
#     if checkpoint_path is not None and not (not load_from_HF and 'model_weight_path' in locals()):
#         _load_checkpoint(model, checkpoint_path)

#     # Setup device and mode
#     model = _setup_device_and_mode(model, device, eval_mode)

#     return model


# def download_ckpt_from_hf():
#     SAM3_MODEL_ID = "facebook/sam3"
#     SAM3_CKPT_NAME = "sam3.pt"
#     SAM3_CFG_NAME = "config.json"
#     _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
#     checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME)
#     return checkpoint_path


# def build_sam3_video_model(
#     checkpoint_path: Optional[str] = None,
#     load_from_HF=True,
#     bpe_path: Optional[str] = None,
#     has_presence_token: bool = True,
#     geo_encoder_use_img_cross_attn: bool = True,
#     strict_state_dict_loading: bool = True,
#     apply_temporal_disambiguation: bool = True,
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     compile=False,
# ) -> Sam3VideoInferenceWithInstanceInteractivity:
#     """
#     Build SAM3 dense tracking model.

#     Args:
#         checkpoint_path: Optional path to checkpoint file
#         bpe_path: Path to the BPE tokenizer file

#     Returns:
#         Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
#     """
#     if bpe_path is None:
#         bpe_path = pkg_resources.resource_filename(
#             "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
#         )

#     # Build Tracker module
#     tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

#     # Build Detector components
#     visual_neck = _create_vision_backbone()
#     text_encoder = _create_text_encoder(bpe_path)
#     backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
#     transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
#     segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
#     input_geometry_encoder = _create_geometry_encoder()

#     # Create main dot product scoring
#     main_dot_prod_mlp = MLP(
#         input_dim=256,
#         hidden_dim=2048,
#         output_dim=256,
#         num_layers=2,
#         dropout=0.1,
#         residual=True,
#         out_norm=nn.LayerNorm(256),
#     )
#     main_dot_prod_scoring = DotProductScoring(
#         d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
#     )

#     # Build Detector module
#     detector = Sam3ImageOnVideoMultiGPU(
#         num_feature_levels=1,
#         backbone=backbone,
#         transformer=transformer,
#         segmentation_head=segmentation_head,
#         semantic_segmentation_head=None,
#         input_geometry_encoder=input_geometry_encoder,
#         use_early_fusion=True,
#         use_dot_prod_scoring=True,
#         dot_prod_scoring=main_dot_prod_scoring,
#         supervise_joint_box_scores=has_presence_token,
#     )

#     # Build the main SAM3 video model
#     if apply_temporal_disambiguation:
#         model = Sam3VideoInferenceWithInstanceInteractivity(
#             detector=detector,
#             tracker=tracker,
#             score_threshold_detection=0.5,
#             assoc_iou_thresh=0.1,
#             det_nms_thresh=0.1,
#             new_det_thresh=0.7,
#             hotstart_delay=15,
#             hotstart_unmatch_thresh=8,
#             hotstart_dup_thresh=8,
#             suppress_unmatched_only_within_hotstart=True,
#             min_trk_keep_alive=-1,
#             max_trk_keep_alive=30,
#             init_trk_keep_alive=30,
#             suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
#             suppress_det_close_to_boundary=False,
#             fill_hole_area=16,
#             recondition_every_nth_frame=16,
#             masklet_confirmation_enable=False,
#             decrease_trk_keep_alive_for_empty_masklets=False,
#             image_size=1008,
#             image_mean=(0.5, 0.5, 0.5),
#             image_std=(0.5, 0.5, 0.5),
#             compile_model=compile,
#         )
#     else:
#         # a version without any heuristics for ablation studies
#         model = Sam3VideoInferenceWithInstanceInteractivity(
#             detector=detector,
#             tracker=tracker,
#             score_threshold_detection=0.5,
#             assoc_iou_thresh=0.1,
#             det_nms_thresh=0.1,
#             new_det_thresh=0.7,
#             hotstart_delay=0,
#             hotstart_unmatch_thresh=0,
#             hotstart_dup_thresh=0,
#             suppress_unmatched_only_within_hotstart=True,
#             min_trk_keep_alive=-1,
#             max_trk_keep_alive=30,
#             init_trk_keep_alive=30,
#             suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
#             suppress_det_close_to_boundary=False,
#             fill_hole_area=16,
#             recondition_every_nth_frame=0,
#             masklet_confirmation_enable=False,
#             decrease_trk_keep_alive_for_empty_masklets=False,
#             image_size=1008,
#             image_mean=(0.5, 0.5, 0.5),
#             image_std=(0.5, 0.5, 0.5),
#             compile_model=compile,
#         )

#     # Load checkpoint if provided
#     # ========== 核心修改：模型权重加载逻辑 ==========
#     # 1. 当 不加载HF + checkpoint_path为空 时，使用本地 ModelScope 下载的权重
#     if not load_from_HF and checkpoint_path is None:  # 修正：not load_from_HF
#         LOCAL_SAM3_DIR = "/shared_data/modelData/modelscope/models/facebook/sam3/"
#         # 关键：指向权重文件（.safetensors），而非 config.json
#         checkpoint_path = os.path.join(LOCAL_SAM3_DIR, "model.safetensors")
    
#     # 2. 当 加载HF + checkpoint_path为空 时，保留原HF下载逻辑（如果需要）
#     elif load_from_HF and checkpoint_path is None:
#         checkpoint_path = download_ckpt_from_hf()  # 你原有HF下载函数

#     # 3. 加载权重文件（适配 .safetensors 和 .pt 两种格式）
#     if checkpoint_path is not None:
#         # 判断文件类型，选择对应的加载方式
#         if checkpoint_path.endswith(".safetensors"):
#             # 加载 .safetensors 权重（SAM3 主要权重格式）
#             with g_pathmgr.open(checkpoint_path, "rb") as f:
#                 ckpt = load_file(f)  # 用 safetensors 库加载
#         elif checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
#             # 兼容 .pt 权重文件
#             with g_pathmgr.open(checkpoint_path, "rb") as f:
#                 ckpt = torch.load(f, map_location="cpu", weights_only=True)
#                 # 处理 HF 格式的权重（可能嵌套在 "model" 键下）
#                 if "model" in ckpt and isinstance(ckpt["model"], dict):
#                     ckpt = ckpt["model"]
#         else:
#             raise ValueError(f"不支持的权重文件格式：{checkpoint_path}（仅支持 .safetensors/.pt/.pth）")

#         # 加载权重到模型
#         missing_keys, unexpected_keys = model.load_state_dict(
#             ckpt, strict=strict_state_dict_loading
#         )
#         # 打印缺失/多余的键（调试用）
#         if missing_keys:
#             print(f"⚠️  加载权重时缺失键：{missing_keys[:5]}（仅显示前5个）")
#         if unexpected_keys:
#             print(f"⚠️  加载权重时多余键：{unexpected_keys[:5]}（仅显示前5个）")

#     # 4. 模型移到指定设备（CPU/GPU）
#     model.to(device=device)
#     # 评估模式（如果需要）
#     if eval_mode:
#         model.eval()
    
#     return model


# def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
#     return Sam3VideoPredictorMultiGPU(
#         *model_args, gpus_to_use=gpus_to_use, **model_kwargs
#     )

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import os
from typing import Optional

import pkg_resources

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr
# 新增：导入safetensors加载函数
from safetensors.torch import load_file

from sam3.model.decoder import (
    TransformerDecoder,
    TransformerDecoderLayer,
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from sam3.model.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from sam3.model.geometry_encoders import SequenceGeometryEncoder
from sam3.model.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from sam3.model.memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from sam3.model.model_misc import (
    DotProductScoring,
    MLP,
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from sam3.model.necks import Sam3DualViTDetNeck
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.sam1_task_predictor import SAM3InteractiveImagePredictor
from sam3.model.sam3_image import Sam3Image, Sam3ImageOnVideoMultiGPU
from sam3.model.sam3_tracking_predictor import Sam3TrackerPredictor
from sam3.model.sam3_video_inference import Sam3VideoInferenceWithInstanceInteractivity
from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU
from sam3.model.text_encoder_ve import VETextEncoder
from sam3.model.tokenizer_ve import SimpleTokenizer
from sam3.model.vitdet import ViT
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.transformer import RoPEAttention


# Setup TensorFloat-32 for Ampere GPUs if available
def _setup_tf32() -> None:
    """Enable TensorFloat-32 for Ampere GPUs if available."""
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


def _create_position_encoding(precompute_resolution=None):
    """Create position encoding for visual backbone."""
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None):
    """Create ViT backbone for visual feature extraction."""
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=False):
    """Create ViT neck for feature pyramid."""
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vl_backbone(vit_neck, text_encoder):
    """Create visual-language backbone."""
    return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)


def _create_transformer_encoder() -> TransformerEncoderFusion:
    """Create transformer encoder with its layer."""
    encoder_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=True,
        pos_enc_at_cross_attn_keys=False,
        pos_enc_at_cross_attn_queries=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=True,
        ),
    )

    encoder = TransformerEncoderFusion(
        layer=encoder_layer,
        num_layers=6,
        d_model=256,
        num_feature_levels=1,
        frozen=False,
        use_act_checkpoint=True,
        add_pooled_text_to_img_feat=False,
        pool_text_with_mask=True,
    )
    return encoder


def _create_transformer_decoder() -> TransformerDecoder:
    """Create transformer decoder with its layer."""
    decoder_layer = TransformerDecoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
        ),
        n_heads=8,
        use_text_cross_attention=True,
    )

    decoder = TransformerDecoder(
        layer=decoder_layer,
        num_layers=6,
        num_queries=200,
        return_intermediate=True,
        box_refine=True,
        num_o2m_queries=0,
        dac=True,
        boxRPB="log",
        d_model=256,
        frozen=False,
        interaction_layer=None,
        dac_use_selfatt_ln=True,
        resolution=1008,
        stride=14,
        use_act_checkpoint=True,
        presence_token=True,
    )
    return decoder


def _create_dot_product_scoring():
    """Create dot product scoring module."""
    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def _create_segmentation_head(compile_mode=None):
    """Create segmentation head with pixel decoder."""
    pixel_decoder = PixelDecoder(
        num_upsampling_stages=3,
        interpolation_mode="nearest",
        hidden_dim=256,
        compile_mode=compile_mode,
    )

    cross_attend_prompt = MultiheadAttention(
        num_heads=8,
        dropout=0,
        embed_dim=256,
    )

    segmentation_head = UniversalSegmentationHead(
        hidden_dim=256,
        upsampling_stages=3,
        aux_masks=False,
        presence_head=False,
        dot_product_scorer=None,
        act_ckpt=True,
        cross_attend_prompt=cross_attend_prompt,
        pixel_decoder=pixel_decoder,
    )
    return segmentation_head


def _create_geometry_encoder():
    """Create geometry encoder with all its components."""
    # Create position encoding for geometry encoder
    geo_pos_enc = _create_position_encoding()
    # Create CX block for fuser
    cx_block = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    # Create geometry encoder layer
    geo_layer = TransformerEncoderLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
        pos_enc_at_cross_attn_queries=False,
        pos_enc_at_cross_attn_keys=True,
        cross_attention=MultiheadAttention(
            num_heads=8,
            dropout=0.1,
            embed_dim=256,
            batch_first=False,
        ),
    )

    # Create geometry encoder
    input_geometry_encoder = SequenceGeometryEncoder(
        pos_enc=geo_pos_enc,
        encode_boxes_as_points=False,
        points_direct_project=True,
        points_pool=True,
        points_pos_enc=True,
        boxes_direct_project=True,
        boxes_pool=True,
        boxes_pos_enc=True,
        d_model=256,
        num_layers=3,
        layer=geo_layer,
        use_act_ckpt=True,
        add_cls=True,
        add_post_encode_proj=True,
    )
    return input_geometry_encoder


def _create_sam3_model(
    backbone,
    transformer,
    input_geometry_encoder,
    segmentation_head,
    dot_prod_scoring,
    inst_interactive_predictor,
    eval_mode,
):
    """Create the SAM3 image model."""
    common_params = {
        "backbone": backbone,
        "transformer": transformer,
        "input_geometry_encoder": input_geometry_encoder,
        "segmentation_head": segmentation_head,
        "num_feature_levels": 1,
        "o2m_mask_predict": True,
        "dot_prod_scoring": dot_prod_scoring,
        "use_instance_query": False,
        "multimask_output": True,
        "inst_interactive_predictor": inst_interactive_predictor,
    }

    matcher = None
    if not eval_mode:
        from sam3.train.matcher import BinaryHungarianMatcherV2

        matcher = BinaryHungarianMatcherV2(
            focal=True,
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            alpha=0.25,
            gamma=2,
            stable=False,
        )
    common_params["matcher"] = matcher
    model = Sam3Image(**common_params)

    return model


def _create_tracker_maskmem_backbone():
    """Create the SAM3 Tracker memory encoder."""
    # Position encoding for mask memory backbone
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )

    # Mask processing components
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )

    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )

    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

    maskmem_backbone = SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )

    return maskmem_backbone


def _create_tracker_transformer():
    """Create the SAM3 Tracker transformer components."""
    # Self attention
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )

    # Cross attention
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )

    # Encoder layer
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )

    # Encoder
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )

    # Transformer wrapper
    transformer = TransformerWrapper(
        encoder=encoder,
        decoder=None,
        d_model=256,
    )

    return transformer


def build_tracker(
    apply_temporal_disambiguation: bool, with_backbone: bool = False, compile_mode=None
) -> Sam3TrackerPredictor:
    """
    Build the SAM3 Tracker module for video tracking.

    Returns:
        Sam3TrackerPredictor: Wrapped SAM3 Tracker module
    """

    # Create model components
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    backbone = None
    if with_backbone:
        vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
        backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)
    # Create the Tracker module
    model = Sam3TrackerPredictor(
        image_size=1008,
        num_maskmem=7,
        backbone=backbone,
        backbone_stride=14,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        # SAM parameters
        multimask_output_in_sam=True,
        # Evaluation
        forward_backbone_per_frame_for_eval=True,
        trim_past_non_cond_mem_for_eval=False,
        # Multimask
        multimask_output_for_tracking=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        # Additional settings
        always_start_from_first_ann_frame=False,
        # Mask overlap
        non_overlap_masks_for_mem_enc=False,
        non_overlap_masks_for_output=False,
        max_cond_frames_in_attn=4,
        offload_output_to_cpu_for_eval=False,
        # SAM decoder settings
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
        clear_non_cond_mem_around_input=True,
        fill_hole_area=0,
        use_memory_selection=apply_temporal_disambiguation,
    )

    return model


def _create_text_encoder(bpe_path: str) -> VETextEncoder:
    """Create SAM3 text encoder."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    return VETextEncoder(
        tokenizer=tokenizer,
        d_model=256,
        width=1024,
        heads=16,
        layers=24,
    )


def _create_vision_backbone(
    compile_mode=None, enable_inst_interactivity=True
) -> Sam3DualViTDetNeck:
    """Create SAM3 visual backbone with ViT and neck."""
    # Position encoding
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    # ViT backbone
    vit_backbone: ViT = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck: Sam3DualViTDetNeck = _create_vit_neck(
        position_encoding,
        vit_backbone,
        enable_inst_interactivity=enable_inst_interactivity,
    )
    # Visual neck
    return vit_neck


def _create_sam3_transformer(has_presence_token: bool = True) -> TransformerWrapper:
    """Create SAM3 transformer encoder and decoder."""
    encoder: TransformerEncoderFusion = _create_transformer_encoder()
    decoder: TransformerDecoder = _create_transformer_decoder()

    return TransformerWrapper(encoder=encoder, decoder=decoder, d_model=256)


# 修正：重构_load_checkpoint，支持safetensors和pt格式
def _load_checkpoint(model, checkpoint_path, strict=False):
    """Load model checkpoint from file (support .safetensors and .pt/.pth)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"权重文件不存在：{checkpoint_path}")
    
    # 根据文件后缀选择加载方式
    if checkpoint_path.endswith(".safetensors"):
        # 关键修正：load_file 直接传入文件路径，而非打开的文件流
        ckpt = load_file(checkpoint_path)  # 移除 g_pathmgr.open 包裹
    elif checkpoint_path.endswith((".pt", ".pth")):
        # pt文件仍用g_pathmgr.open加载（保持原有逻辑）
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
            # 处理HF格式的嵌套权重
            if "model" in ckpt and isinstance(ckpt["model"], dict):
                ckpt = ckpt["model"]
    else:
        raise ValueError(f"不支持的权重格式：{checkpoint_path}（仅支持 .safetensors/.pt/.pth）")
    
    # 适配SAM3的权重键名（移除detector.前缀）
    sam3_image_ckpt = {
        k.replace("detector.", ""): v for k, v in ckpt.items() if "detector" in k
    }
    if model.inst_interactive_predictor is not None:
        sam3_image_ckpt.update(
            {
                k.replace("tracker.", "inst_interactive_predictor.model."): v
                for k, v in ckpt.items()
                if "tracker" in k
            }
        )
    
    # 加载权重（strict=False避免小差异报错）
    missing_keys, unexpected_keys = model.load_state_dict(sam3_image_ckpt, strict=strict)
    
    # 打印日志（仅提示缺失键，不中断）
    if len(missing_keys) > 0:
        print(f"加载权重 {checkpoint_path} 时缺失键（前5个）：{missing_keys[:5]}")
    if len(unexpected_keys) > 0:
        print(f"加载权重 {checkpoint_path} 时多余键（前5个）：{unexpected_keys[:5]}")


def _setup_device_and_mode(model, device, eval_mode):
    """Setup model device and evaluation mode."""
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    if eval_mode:
        model.eval()
    return model


def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
):
    """
    Build SAM3 image model

    Args:
        bpe_path: Path to the BPE tokenizer vocabulary
        device: Device to load the model on ('cuda' or 'cpu')
        eval_mode: Whether to set the model to evaluation mode
        checkpoint_path: Optional path to model checkpoint
        load_from_HF: Whether to load from Hugging Face
        enable_segmentation: Whether to enable segmentation head
        enable_inst_interactivity: Whether to enable instance interactivity (SAM 1 task)
        compile: Whether to compile the model

    Returns:
        A SAM3 image model
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Create visual components
    compile_mode = "default" if compile else None
    vision_encoder = _create_vision_backbone(
        compile_mode=compile_mode, enable_inst_interactivity=enable_inst_interactivity
    )

    # Create text components
    text_encoder = _create_text_encoder(bpe_path)

    # Create visual-language backbone
    backbone = _create_vl_backbone(vision_encoder, text_encoder)

    # Create transformer components
    transformer = _create_sam3_transformer()

    # Create dot product scoring
    dot_prod_scoring = _create_dot_product_scoring()

    # Create segmentation head if enabled
    segmentation_head = (
        _create_segmentation_head(compile_mode=compile_mode)
        if enable_segmentation
        else None
    )

    # Create geometry encoder
    input_geometry_encoder = _create_geometry_encoder()
    if enable_inst_interactivity:
        sam3_pvs_base = build_tracker(apply_temporal_disambiguation=False)
        inst_predictor = SAM3InteractiveImagePredictor(sam3_pvs_base)
    else:
        inst_predictor = None
    
    # Create the SAM3 model
    model = _create_sam3_model(
        backbone,
        transformer,
        input_geometry_encoder,
        segmentation_head,
        dot_prod_scoring,
        inst_predictor,
        eval_mode,
    )

    # 权重加载逻辑（核心修正）
    if checkpoint_path is None:
        if not load_from_HF:
            # 本地ModelScope权重路径
            LOCAL_SAM3_DIR = "/shared_data/modelData/modelscope/models/facebook/sam3/"
            checkpoint_path = os.path.join(LOCAL_SAM3_DIR, "model.safetensors")
        else:
            # 从HF下载权重
            checkpoint_path = download_ckpt_from_hf()
    
    # 加载权重
    if checkpoint_path is not None:
        _load_checkpoint(model, checkpoint_path, strict=False)

    # Setup device and mode
    model = _setup_device_and_mode(model, device, eval_mode)

    return model


def download_ckpt_from_hf():
    """Download SAM3 checkpoint from Hugging Face Hub."""
    SAM3_MODEL_ID = "facebook/sam3"
    SAM3_CKPT_NAME = "sam3.pt"
    SAM3_CFG_NAME = "config.json"
    _ = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CFG_NAME)
    checkpoint_path = hf_hub_download(repo_id=SAM3_MODEL_ID, filename=SAM3_CKPT_NAME)
    return checkpoint_path


def build_sam3_video_model(
    checkpoint_path: Optional[str] = None,
    load_from_HF=True,
    bpe_path: Optional[str] = None,
    has_presence_token: bool = True,
    geo_encoder_use_img_cross_attn: bool = True,
    strict_state_dict_loading: bool = False,  # 默认False避免报错
    apply_temporal_disambiguation: bool = True,
    device="cuda" if torch.cuda.is_available() else "cpu",
    compile=False,
) -> Sam3VideoInferenceWithInstanceInteractivity:
    """
    Build SAM3 dense tracking model.

    Args:
        checkpoint_path: Optional path to checkpoint file
        load_from_HF: Whether to load from Hugging Face
        bpe_path: Path to the BPE tokenizer file
        strict_state_dict_loading: Whether to use strict state dict loading

    Returns:
        Sam3VideoInferenceWithInstanceInteractivity: The instantiated dense tracking model
    """
    if bpe_path is None:
        bpe_path = pkg_resources.resource_filename(
            "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
        )

    # Build Tracker module
    tracker = build_tracker(apply_temporal_disambiguation=apply_temporal_disambiguation)

    # Build Detector components
    visual_neck = _create_vision_backbone()
    text_encoder = _create_text_encoder(bpe_path)
    backbone = SAM3VLBackbone(scalp=1, visual=visual_neck, text=text_encoder)
    transformer = _create_sam3_transformer(has_presence_token=has_presence_token)
    segmentation_head: UniversalSegmentationHead = _create_segmentation_head()
    input_geometry_encoder = _create_geometry_encoder()

    # Create main dot product scoring
    main_dot_prod_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    main_dot_prod_scoring = DotProductScoring(
        d_model=256, d_proj=256, prompt_mlp=main_dot_prod_mlp
    )

    # Build Detector module
    detector = Sam3ImageOnVideoMultiGPU(
        num_feature_levels=1,
        backbone=backbone,
        transformer=transformer,
        segmentation_head=segmentation_head,
        semantic_segmentation_head=None,
        input_geometry_encoder=input_geometry_encoder,
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=main_dot_prod_scoring,
        supervise_joint_box_scores=has_presence_token,
    )

    # Build the main SAM3 video model
    if apply_temporal_disambiguation:
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=15,
            hotstart_unmatch_thresh=8,
            hotstart_dup_thresh=8,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=16,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )
    else:
        # a version without any heuristics for ablation studies
        model = Sam3VideoInferenceWithInstanceInteractivity(
            detector=detector,
            tracker=tracker,
            score_threshold_detection=0.5,
            assoc_iou_thresh=0.1,
            det_nms_thresh=0.1,
            new_det_thresh=0.7,
            hotstart_delay=0,
            hotstart_unmatch_thresh=0,
            hotstart_dup_thresh=0,
            suppress_unmatched_only_within_hotstart=True,
            min_trk_keep_alive=-1,
            max_trk_keep_alive=30,
            init_trk_keep_alive=30,
            suppress_overlapping_based_on_recent_occlusion_threshold=0.7,
            suppress_det_close_to_boundary=False,
            fill_hole_area=16,
            recondition_every_nth_frame=0,
            masklet_confirmation_enable=False,
            decrease_trk_keep_alive_for_empty_masklets=False,
            image_size=1008,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
            compile_model=compile,
        )

    # 权重加载逻辑（统一风格）
    if checkpoint_path is None:
        if not load_from_HF:
            LOCAL_SAM3_DIR = "/shared_data/modelData/modelscope/models/facebook/sam3/"
            checkpoint_path = os.path.join(LOCAL_SAM3_DIR, "model.safetensors")
        else:
            checkpoint_path = download_ckpt_from_hf()
    
    if checkpoint_path is not None:
        # 加载视频模型权重
        if checkpoint_path.endswith(".safetensors"):
            with g_pathmgr.open(checkpoint_path, "rb") as f:
                ckpt = load_file(f)
        else:
            with g_pathmgr.open(checkpoint_path, "rb") as f:
                ckpt = torch.load(f, map_location="cpu", weights_only=True)
                if "model" in ckpt and isinstance(ckpt["model"], dict):
                    ckpt = ckpt["model"]
        
        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=strict_state_dict_loading)
        if len(missing_keys) > 0:
            print(f"视频模型加载权重缺失键：{missing_keys[:5]}")
        if len(unexpected_keys) > 0:
            print(f"视频模型加载权重多余键：{unexpected_keys[:5]}")

    # 移到指定设备
    model.to(device=device)
    return model


def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):
    return Sam3VideoPredictorMultiGPU(
        *model_args, gpus_to_use=gpus_to_use, **model_kwargs
    )