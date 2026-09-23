# TensorRT-LLM 0.12 for JetPack 6 (L4T r36.4). Used only to build engines and to
# copy the C++ runtime out (install-runtime.sh); local-copilot-codebuddy itself runs natively.
FROM dustynv/tensorrt_llm:0.12-r36.4.0

# The base image asks for every driver capability, including graphics, whose
# Vulkan files clash with newer NVIDIA container runtimes. CUDA is all we need.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
