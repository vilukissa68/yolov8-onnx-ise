#include "tvm/runtime/c_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>

// Point to the header inside the codegen folder
#include "tvmgen_default.h"

int main() {
    // 1. Setup Input/Output data
    // (Dimensions must match your model, e.g., 1x3x640x640)
    size_t input_size = 1 * 3 * 640 * 640;
    size_t output_size =
        1 * 84 * 8400; // Adjust based on your specific YOLO output

    float *input_data = (float *)malloc(input_size * sizeof(float));
    float *output_data = (float *)malloc(output_size * sizeof(float));

    // Fill input with dummy data
    for (size_t i = 0; i < input_size; i++)
        input_data[i] = 0.5f;

    // 2. Prepare the struct expected by the AOT executor
    struct tvmgen_default_inputs inputs = {.images = input_data};
    struct tvmgen_default_outputs outputs = {.output = output_data};

    // 3. Run Inference
    printf("Running inference...\n");
    int ret = tvmgen_default_run(&inputs, &outputs);

    if (ret == 0) {
        printf("Success! Output[0] = %f\n", output_data[0]);
    } else {
        printf("Error: %d\n", ret);
    }

    free(input_data);
    free(output_data);
    return 0;
}
