_base_ = [
    '../../../../_base_/default_runtime.py',
    '../../../../_base_/datasets/h36m.py'
]
evaluation = dict(interval=10, metric=['mpjpe', 'p-mpjpe'], save_best='MPJPE')

# optimizer settings
optimizer = dict(
    type='Adam',
    lr=1e-3,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[170, 200])

total_epochs = 160

log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

channel_cfg = dict(
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='TopDown',
    pretrained='https://download.openmmlab.com/mmpose/'
    'pretrain_models/hrnet_w32-36af842e.pth',
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
    ),
    keypoint_head=dict(
        type='TokenPoseHead',
        in_channels=32,
        num_joints=channel_cfg['num_output_channels'],
        loss_keypoint=dict(type='JointsMSELoss', use_target_weight=True),
        tokenpose_cfg=dict(
            feature_size=[64, 64],
            patch_size=[4,4],
            dim=256,
            depth=12,
            heads=8,
            mlp_ratio=3,
            heatmap_size=[64, 64],
            pos_embedding_type='sine-full',
            apply_init=True
        )),
    train_cfg=dict(),
    test_cfg=dict(
        flip_test=True,
        post_process='unbiased',
        shift_heatmap=True,
        modulate_kernel=11))

# data settings
data_root = '/HOME/scz3186/run/fanao/dataset/human3.6m'
data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    seq_len=27,
    seq_frame_interval=1,
    causal=False,
    temporal_padding=True,
    joint_2d_src='gt',
    need_camera_param=True,
    camera_param_file=f'{data_root}/annotation_body3d/cameras.pkl',
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=2),
    
    dict(
        type='GetRootCenteredPose',
        item='3d_target',
        visible_item='3d_target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(
        type='RelativeJointRandomFlip',
        item=['input_2d', '3d_target'],
        flip_cfg=[
            dict(center_mode='static', center_x=0.),
            dict(center_mode='root', center_index=0)
        ],
        visible_item=['input_2d_visible', '3d_target_visible'],
        flip_prob=0.5),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=['img', 'target', 'target_weight',('input_2d', 'input'), '3d_target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position','image_file', 
                   'center', 'scale','rotation', 'bbox_score'])
]

val_pipeline = [
     dict(type='LoadImageFromFile'),
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    
    dict(
        type='GetRootCenteredPose',
        item='target',
        visible_item='target_visible',
        root_index=0,
        root_name='root_position',
        remove_root=False),
    dict(type='ImageCoordinateNormalization', item='input_2d'),
    dict(type='PoseSequenceToTensor', item='input_2d'),
    dict(
        type='Collect',
        keys=['img',('input_2d', 'input'), 'target'],
        meta_name='metas',
        meta_keys=['target_image_path', 'flip_pairs', 'root_position','image_file', 
                   'center', 'scale', 'rotation', 'bbox_score'])
]

test_pipeline = val_pipeline

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=3),
    test_dataloader=dict(samples_per_gpu=3),
    train=dict(
        type='SelfBody3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_train.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='SelfBody3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='SelfBody3DH36MDataset',
        ann_file=f'{data_root}/annotation_body3d/fps10/h36m_test.npz',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
