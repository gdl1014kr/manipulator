## onnx -> tensorrt(image encoder, FP16)

trtexec \
    --onnx=data/image_encoder.onnx \
    --saveEngine=data/image_encoder_fp16_native.engine \
    --fp16 \
    --layerDeviceTypes=*:GPU
    
## onnx -> tensorrt(mask decoder, FP16)

trtexec \
    --onnx=data/mask_decoder.onnx \
    --saveEngine=data/mask_decoder_fp16_native.engine \
    --fp16 \
    --layerDeviceTypes=*:GPU \
    --minShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
    --optShapes=image_embeddings:1x256x64x64,point_coords:1x1x2,point_labels:1x1,mask_input:1x1x256x256,has_mask_input:1 \
    --maxShapes=image_embeddings:1x256x64x64,point_coords:1x10x2,point_labels:1x10,mask_input:1x1x256x256,has_mask_input:1
