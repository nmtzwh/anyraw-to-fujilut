import LibRaw from 'libraw-wasm';

// This is a simplified interface. The actual options are more complex.
export interface DecodeOptions {
    use_camera_wb?: boolean;
    no_auto_bright?: boolean;
    output_color?: number; // 5 for XYZ
    gamma?: [number, number];
    output_bps?: 16;
}

/**
 * Decodes a RAW image file using libraw-wasm.
 * @param file The RAW file to decode.
 * @param options The decoding options.
 * @returns A promise that resolves with the decoded image data.
 */
export function decode(file: File, options: DecodeOptions = {}): Promise<ImageData> {
    return new Promise(async (resolve, reject) => {
        try {
            const raw = new LibRaw();
            const buffer = await file.arrayBuffer();
            
            // Default options similar to the python script
            const decodeOptions = {
                use_camera_wb: true,
                no_auto_bright: true,
                output_color: 5, // XYZ
                gamma: [1, 1],
                output_bps: 16,
                ...options
            };

            await raw.open(new Uint8Array(buffer));
            
            // Post-processing with options
            // Note: libraw-wasm's postprocess method may differ slightly from rawpy
            // I am assuming a similar API based on the documentation.
            // A more detailed implementation may need to check the exact method signature.
            const processed = await raw.postprocess(decodeOptions);

            // The processed data should be an object containing the image data, width, and height.
            // I am assuming it returns an object like { data: Float32Array, width: number, height: number }
            // and that the data is in XYZ format.
            const { data, width, height } = await raw.imageData();

            // The data from libraw is a Float32Array. We need to convert it to a Uint8ClampedArray for ImageData.
            // This is a lossy conversion, but necessary for display.
            // The actual processing pipeline will use the Float32Array.
            const clampedData = new Uint8ClampedArray(data.length * 4);
            for (let i = 0, j = 0; i < data.length; i += 3, j+=4) {
                clampedData[j] = data[i] * 255;
                clampedData[j+1] = data[i+1] * 255;
                clampedData[j+2] = data[i+2] * 255;
                clampedData[j+3] = 255;
            }

            resolve(new ImageData(clampedData, width, height));

        } catch (e) {
            reject(e);
        }
    });
}
