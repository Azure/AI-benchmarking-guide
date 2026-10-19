# NVIDIA MLPerf Inference Benchmarks

## List of Benchmarks

Please refer to the `README.md` in each benchmark directory for implementation details.
- [3d-unet](3d-unet/tensorrt/README.md)
- [bert](bert/tensorrt/README.md)
- [dlrm-v2](dlrm/tensorrt/README.md)
- [resnet50](resnet50/tensorrt/README.md)
- [gptj](gptj/tensorrt/README.md)
- [retinanet](retinanet/tensorrt/README.md)

## Other Directories

- [common](common) - holds shared scripts to generate TensorRT optimized plan files and to run the harnesses.
- [harness](harness) - holds source codes of the harness interfacing with LoadGen.
- [plugin](plugin) - holds source codes of TensorRT plugins used by the benchmarks.
