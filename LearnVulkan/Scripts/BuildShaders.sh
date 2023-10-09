CURRENT_DIR=$(cd $(dirname $0); pwd)
echo "Current Dir: "${CURRENT_DIR}
${CURRENT_DIR}/glslc ./Shaders/ShaderDrawTriangle.vert -o ./Shaders/ShaderDrawTriangle_Vert.spv
${CURRENT_DIR}/glslc ./Shaders/ShaderDrawTriangle.frag -o ./Shaders/ShaderDrawTriangle_Frag.spv
