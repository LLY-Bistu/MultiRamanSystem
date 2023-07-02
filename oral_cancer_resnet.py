# ResNet50 configuration
model = dict(
    type='RamanClassifier',  # Type of classifier
    backbone=dict(
        type='ResNet50',
        # type='VGG',
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiTaskLinearClsHead',
        num_classes=5,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1,),
    ),
)
