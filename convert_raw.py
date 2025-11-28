import rawpy
import numpy as np
import imageio.v3 as iio
import colour

import os
import argparse

def apply_flog2_curve(linear_data):
    """
    Applies the Fujifilm F-Log2 OETF (Opto-Electronic Transfer Function).
    Based on F-Log2 Data Sheet Ver.1.1
    """
    # F-Log2 Constants
    a = 5.555556
    b = 0.064829
    c = 0.245281
    d = 0.384316
    e = 8.799461
    f = 0.092864
    cut1 = 0.000889

    # Prepare output array
    out = np.zeros_like(linear_data)

    # Mask for the piecewise function
    mask_log = linear_data >= cut1
    mask_linear = linear_data < cut1

    # Apply Logarithmic Curve (Shoulder)
    # V = c * log10(a * x + b) + d
    out[mask_log] = c * np.log10(a * linear_data[mask_log] + b) + d

    # Apply Linear Segment (Toe)
    # V = e * x + f
    out[mask_linear] = e * linear_data[mask_linear] + f

    # Clip to valid 0.0 - 1.0 range just in case
    return np.clip(out, 0.0, 1.0)

def get_exposure_gain(xyz_image, target_grey=0.18, ev_offset=0.0):
    """
    Calculates the gain required to normalize the image exposure based on
    the Geometric Mean of the Luminance (Y) channel.
    """
    # 1. Extract Luminance (Y is index 1 in XYZ)
    # We use a stride (slice) to downsample the image for faster calculation.
    # Calculating mean on 24MP takes time; calculating on a 1% subsample is instant and accurate enough.
    Y_channel = xyz_image[::10, ::10, 1]

    # 2. Calculate Geometric Mean
    # Geometric Mean = exp(mean(log(x)))
    # We add a tiny epsilon to avoid log(0) errors on pure black pixels
    epsilon = 1e-6
    log_Y = np.log(np.maximum(Y_channel, epsilon))
    geom_mean = np.exp(np.mean(log_Y))

    # 3. Calculate Gain
    # If the image is pitch black (geom_mean near 0), gain would be infinite. 
    # We clip the denominator to be safe.
    safe_mean = max(geom_mean, epsilon)
    
    # Base gain to reach 18% grey
    gain = target_grey / safe_mean
    
    # 4. Apply User EV Offset (2^EV)
    # e.g., +1 EV = 2x brightness
    final_gain = gain * (2.0 ** ev_offset)
    
    return final_gain, safe_mean

def crop_raw_with_flips(xyz_img, imagesize):
    flip = imagesize.flip

    # https://www.libraw.org/docs/API-datastruct-eng.html#libraw_image_sizes_t
    match flip:
        case 0: # normal
            left = imagesize.crop_left_margin
            top  = imagesize.crop_top_margin
            right = left + imagesize.crop_width
            bottom = top + imagesize.crop_height
            return xyz_img[top:bottom, left:right]
        case 3: # 180-deg 
            left = imagesize.raw_width - imagesize.crop_left_margin - imagesize.crop_width
            top  = imagesize.raw_height - imagesize.crop_top_margin - imagesize.crop_height
            right = left + imagesize.crop_width
            bottom = top + imagesize.crop_height
            return xyz_img[top:bottom, left:right]
        case 5: # 90-deg counterclockwise 
            left = imagesize.crop_top_margin
            top = imagesize.raw_width - imagesize.crop_left_margin - imagesize.crop_width
            right = left + imagesize.crop_height
            bottom = top + imagesize.crop_width
            return xyz_img[top:bottom, left:right]
        case 6: # 90-deg clockwise 
            left = imagesize.raw_height - imagesize.crop_top_margin - imagesize.crop_height
            top = imagesize.crop_left_margin
            right = left + imagesize.crop_height
            bottom = top + imagesize.crop_width
            return xyz_img[top:bottom, left:right]
        case _:  # Default case (wildcard)
            raise ValueError(f"Unknown flip: {flip}")

def convert_raw_to_flog2(input_path):
    print(f"Processing: {input_path}")
    
    with rawpy.imread(input_path) as raw:
        # 1. Demosaic and convert to CIE XYZ (Linear)
        # use_camera_wb=True applies the "As Shot" white balance
        # no_auto_bright=True prevents automatic histogram stretching
        # output_color=rawpy.ColorSpace.XYZ directs libraw to output in XYZ space
        xyz_image = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            bright=1.0,
            user_sat=None,
            output_color=rawpy.ColorSpace.XYZ,
            gamma=(1, 1),  # Linear gamma (essential!)
            output_bps=16  # High bit depth for precision
        )
        # 1.5 effective image size         
        xyz_cropped = crop_raw_with_flips(xyz_image, raw.sizes)
        
        print(f"Original Size: {xyz_image.shape}")
        print(f"Cropped Size:  {xyz_cropped.shape}")

    # Convert from Integer 16-bit to Float 0-1
    xyz_float = xyz_cropped.astype(np.float32) / 65535.0
    
    # 2. apply exposure correction to 18% gray
    # important if you notice under-exposed image output
    gain, original_mean = get_exposure_gain(xyz_float)
    print(f"  > Original Geometric Mean: {original_mean:.4f}")
    print(f"  > Applied Gain: {gain:.4f} (Approx {np.log2(gain):.2f} stops)")
    # Apply gain
    xyz_exposed = xyz_float * gain

    # 3. Convert CIE XYZ to Rec.2020 (Linear)
    # Matrix: XYZ -> Rec.2020
    M_XYZ_to_Rec2020 = np.array([
        [ 1.716651, -0.355671, -0.252965],
        [-0.666684,  1.616481,  0.015768],
        [ 0.017640, -0.042771,  0.942103]
    ])

    # Reshape for matrix multiplication: (H*W, 3)
    h, w, ch = xyz_exposed.shape
    xyz_flat = xyz_exposed.reshape(-1, 3)
    
    # Perform the color space transform
    # Transpose matrix for (N,3) dot (3,3)
    rec2020_flat = np.dot(xyz_flat, M_XYZ_to_Rec2020.T)
    
    # Reshape back to image dimensions
    rec2020_linear = rec2020_flat.reshape(h, w, ch)

    # Handle out-of-gamut colors (negative values) caused by matrix transform
    rec2020_linear = np.maximum(rec2020_linear, 0.0)

    # 4. Apply F-Log2 Gamma Curve
    flog2_data = apply_flog2_curve(rec2020_linear)

    # 5. Quantize to 16-bit Integer
    # F-Log2 is often 10-bit, but 16-bit gives more grading room.
    return (flog2_data * 65535).astype(np.uint16)


def load_lut(lut_path):
    """
    Loads a .cube file using the colour-science library.
    """
    print(f"Loading LUT: {lut_path}")
    try:
        LUT = colour.read_LUT(lut_path)
        return LUT
    except ValueError as e:
        print(f"Error loading LUT: {e}")
        return None

def apply_lut_to_image(img, lut_obj, output_path, bit_depth_in=16):
    """
    Applies a 3D LUT to an image.
    
    Args:
        img (numpy array): F-Log2 image in uint16.
        lut_obj (LUT3D): The loaded LUT object.
        output_path (str): Where to save the result.
    """
    # 1. Normalize to 0.0 - 1.0 range
    # CRITICAL STEP: .cube LUTs expect inputs from 0.0 to 1.0.
    # Our previous script saved 12-bit data (max 4095).
    # If we just divide by 65535 (standard 16-bit), the image will be too dark for the LUT.
    max_val = (2**bit_depth_in) - 1
    img_float = img.astype(np.float32) / max_val
    
    # Clip input to ensure it stays within the LUT's domain
    img_float = np.clip(img_float, 0.0, 1.0)

    # 2. Apply the LUT (Trilinear Interpolation)
    print("Applying LUT (this may take a moment)...")
    img_graded = lut_obj.apply(img_float)

    # 4. Prepare for Output
    # Most LUTs transform Log -> Rec.709 (Display Standard).
    # We usually save graded images as 8-bit or 16-bit standard integers.
    
    # Scale to 16-bit integer range (0-65535) for high quality save
    # img_out = (img_graded * 65535).astype(np.uint16)
    img_out = (img_graded * 255).astype(np.uint8) # use 8bit for smaller files

    # 5. Save
    iio.imwrite(output_path, img_out, quality=90)
    print(f"Graded image saved to: {output_path}")

def process_batch(image_path, lut_folder_path):
    """
    Applies every LUT in a folder to a single image.
    """
    # Load the image once implies we will process it multiple times
    if not os.path.exists(image_path):
        print("Image not found.")
        return
    
    image_flog2 = convert_raw_to_flog2(image_path)
    print(f"Image loaded and converted to F-Log2.")

    lut_files = [f for f in os.listdir(lut_folder_path) if f.endswith('.cube')]
    
    if not lut_files:
        print("No .cube files found in the folder.")
        return

    print(f"Found {len(lut_files)} LUTs.")

    for lut_file in lut_files:
        lut_path = os.path.join(lut_folder_path, lut_file)
        lut_name = os.path.splitext(lut_file)[0]
        
        # Create a descriptive output filename
        # e.g., "photo_VintageLook.tiff"
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = f"{base_name}_{lut_name}.jpeg"
        
        lut_obj = load_lut(lut_path)
        if lut_obj:
            apply_lut_to_image(image_flog2, lut_obj, out_path)

def main():
    """
    1. Define your target raw image
    2. Define folder containing your .cube files
    """
    parser = argparse.ArgumentParser(description="A Python program that applies LUTs on RAW image")
    
    # Add arguments
    parser.add_argument("-i", "--image", type=str, help="Path to target raw image.")
    parser.add_argument("-l", "--lut", type=str, help="Folder that contains LUTs.")
    args = parser.parse_args()

    process_batch(args.image, args.lut)

if __name__ == "__main__":
    main()
