# Copyright (c) OpenMMLab. All rights reserved.
from .body3d_h36m_dataset import Body3DH36MDataset
from .body3d_mpi_inf_3dhp_dataset import Body3DMpiInf3dhpDataset
from .body3d_mview_direct_campus_dataset import Body3DMviewDirectCampusDataset
from .body3d_mview_direct_panoptic_dataset import \
    Body3DMviewDirectPanopticDataset
from .body3d_mview_direct_shelf_dataset import Body3DMviewDirectShelfDataset
from .body3d_semi_supervision_dataset import Body3DSemiSupervisionDataset
from .body3d_h36m_modified_dataset import Body3DH36MModifiedDataset
from .body3d_aist_dataset import Body3DAISTDataset
from .body3d_panoptic_dataset import Body3DPanopticDataset
from .body3d_combined_dataset import Body3DCombinedDataset
from .body3d_aist_coco_dataset import Body3DAISTCOCODataset
from .body3d_h36m_coco_dataset import Body3DH36MCOCODataset

__all__ = [
    'Body3DH36MDataset', 'Body3DSemiSupervisionDataset',
    'Body3DMpiInf3dhpDataset', 'Body3DMviewDirectPanopticDataset',
    'Body3DMviewDirectShelfDataset', 'Body3DMviewDirectCampusDataset',
    'Body3DH36MModifiedDataset',
    'Body3DAISTDataset', "Body3DPanopticDataset",
    "Body3DCombinedDataset", "Body3DAISTCOCODataset",
    "Body3DH36MCOCODataset"
]
