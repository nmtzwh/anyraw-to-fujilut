import torch
import torch.nn.functional as F
import rawpy
import numpy as np
import colour
import imageio.v3 as iio

import os
import argparse


# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

# --- 1. Define the exposure alignment Module ---

class AutoExposure(torch.nn.Module):
    def __init__(self, target_grey=0.18):
        super().__init__()
        self.target = target_grey
        # Epsilon to prevent log(0) errors
        self.eps = 1e-6

    def forward(self, xyz_tensor, manual_ev=0.0, spot_coords=None):
        """
        xyz_tensor: (Batch, H, W, 3) or (H, W, 3)
        manual_ev: User override (like the exposure dial on a camera)
        spot_coords: Tuple (x, y, radius) to meter off a specific grey card.
        """
        # Extract Luminance (Y is channel 1 in XYZ)
        # We assume channel last format
        Y = xyz_tensor[..., 1]

        if spot_coords:
            # --- Spot Metering Mode ---
            cx, cy, r = spot_coords
            # Slice out the region of interest
            # specific implementation depends on image dims, simplified here:
            y_roi = Y[..., max(0, cy-r):cy+r, max(0, cx-r):cx+r]
            
            # Use arithmetic mean for a flat grey card patch
            avg_luminance = torch.mean(y_roi)
            
        else:
            # --- Matrix Metering Mode (Geometric Mean) ---
            # 1. Downsample for speed (we don't need 24MP to calculate exposure)
            # We use average pooling to shrink image to approx 256x256
            stride = max(1, Y.shape[-2] // 256)
            y_small = Y[..., ::stride, ::stride]
            
            # 2. geometric_mean = exp(mean(log(x)))
            # We clamp low values to epsilon to avoid -inf
            log_y = torch.log(torch.clamp(y_small, min=self.eps))
            avg_luminance = torch.exp(torch.mean(log_y))

        # Calculate Gain
        # If avg is 0.09 (underexposed), we need gain of 2.0 to reach 0.18
        gain = self.target / torch.clamp(avg_luminance, min=self.eps)
        
        # Apply Manual EV Offset: Gain * 2^EV
        # If user wants +1 stop, we double the gain.
        final_gain = gain * (2.0 ** manual_ev)
        
        # Apply gain to all channels (X, Y, Z)
        return xyz_tensor * final_gain, final_gain.item()

# --- 2. Define the CIE XYZ to FLOG2 Module ---

class FLog2Pipeline(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Matrices and Constants
        M_np = np.array([
            [ 1.716651, -0.355671, -0.252965],
            [-0.666684,  1.616481,  0.015768],
            [ 0.017640, -0.042771,  0.942103]
        ], dtype=np.float32)
        self.register_buffer('M_XYZ_Rec2020', torch.from_numpy(M_np.T))
        
        self.a = 5.555556
        self.b = 0.064829
        self.c = 0.245281
        self.d = 0.384316
        self.e = 8.799461
        self.f = 0.092864
        self.cut1 = 0.000889
        
        # Initialize Exposure Module
        self.auto_exp = AutoExposure(target_grey=0.18)

    def forward(self, xyz_data, ev_offset=0.0, spot_crop=None):
        # 1. Auto Exposure (Linear Domain)
        exposed_xyz, calculated_gain = self.auto_exp(xyz_data, ev_offset, spot_crop)
        
        # 2. Color Space: XYZ -> Rec.2020
        linear_rec2020 = exposed_xyz @ self.M_XYZ_Rec2020
        x = torch.clamp(linear_rec2020, min=0.0)

        # 3. F-Log2 Curve
        log_segment = self.c * torch.log10(self.a * x + self.b) + self.d
        lin_segment = self.e * x + self.f
        out = torch.where(x >= self.cut1, log_segment, lin_segment)
        
        return torch.clamp(out, 0.0, 1.0), calculated_gain

# --- 3. Define the 3D-LUT applier Module --- 
class LUT3DApplier(torch.nn.Module):
    def __init__(self, lut_numpy):
        super().__init__()
        # LUT needs to be reshaped for grid_sample: (Batch, Channels, Depth, Height, Width)
        # Standard numpy LUT is usually (Size, Size, Size, 3)
        
        # Transpose from (B, G, R, C) -> (C, R, G, B)
        # TODO: check fujifilm's 3d-lut specification
        lut_tensor = torch.from_numpy(lut_numpy).permute(3, 2, 1, 0).unsqueeze(0).float()
        self.register_buffer('lut', lut_tensor)

    def forward(self, img_tensor):
        """
        img_tensor: (1, H, W, 3) float32 in range [0, 1]
        """
        # 1. Convert [0, 1] image data to [-1, 1] coordinate system
        # grid_sample expects coordinates (x, y, z) in range [-1, 1]
        # We flip the channel order if necessary, but usually LUTs expect RGB input.
        grid = img_tensor * 2.0 - 1.0
        
        # 2. Expand dimension to match grid_sample requirement
        # grid needs to be (N, D, H, W, 3). 
        # For us, we are processing a 2D image, so we treat 'Depth' as 1,
        # or technically, we treat the image spatial dims as the "grid" dimensions.
        # Actually, for grid_sample 3D, the grid input shape is (N, Output_Depth, Output_H, Output_W, 3).
        # Since we want a 2D image out, we can set Output_Depth = 1.
        # However, simple way: (N, H, W, 1, 3) works best for 2D images mapped through 3D LUTs.
        grid = grid.unsqueeze(3) # Shape: (1, H, W, 1, 3)

        # 3. Sample
        # align_corners=True matches the standard LUT logic (corner to corner mapping)
        sampled = F.grid_sample(self.lut, grid, mode='bilinear', padding_mode='border', align_corners=True)
        
        # Output is (N, C, H, W, 1). Remove unneeded dims -> (H, W, C)
        # Permute back to (H, W, C) for saving
        sampled = sampled.squeeze(4).permute(0, 2, 3, 1)
        
        return sampled
    
# crop out hidden pixels (for correct exposure calculation)
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
        
def process_pipeline(image_path, lut_folder_path, output_path):
    # --- Step A: CPU Bound (LibRaw) ---
    # There is no escaping CPU here easily without custom C++ CUDA decoders
    # Load the image once implies we will process it multiple times
    if not os.path.exists(image_path):
        print("Image not found.")
        return
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    with rawpy.imread(image_path) as raw:
        xyz_image = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,
            bright=1.0,
            output_color=rawpy.ColorSpace.XYZ,
            gamma=(1, 1), 
            output_bps=16
        )
        xyz_cropped = crop_raw_with_flips(xyz_image, raw.sizes)       
        print(f"Original Size: {xyz_image.shape}")
        print(f"Cropped Size:  {xyz_cropped.shape}")
    
    # find all lut files in the path
    lut_files = [f for f in os.listdir(lut_folder_path) if f.endswith('.cube')]
    
    if not lut_files:
        print("No .cube files found in the folder.")
        return

    print(f"Found {len(lut_files)} LUTs.")
    
    # To Tensor (Normalize 16bit -> float)
    # We send to GPU immediately
    raw_tensor = torch.from_numpy(xyz_cropped.astype(np.float32)).to(device) / 65535.0

    # Compile
    processor = torch.compile(FLog2Pipeline(), mode="reduce-overhead").to(device)
    
    # --- Step B: GPU Accelerated (Raw -> F-Log2) ---
    with torch.inference_mode():
        flog2_img, gain_applied = processor(raw_tensor)
        
        # --- Step C: GPU Accelerated (LUT Application) ---
        # Note: We usually compile the LUT applier per LUT, or just use functional grid_sample 
        # if swapping LUTs frequently to avoid re-compiling every time.
        for lut_file in lut_files:
            lut_path = os.path.join(lut_folder_path, lut_file)
            lut_name = os.path.splitext(lut_file)[0]

            # Load LUT (CPU)
            lut_obj = colour.read_LUT(lut_path)
            # Assume .cube is valid, extract table
            lut_np = lut_obj.table.astype(np.float32)
            
            # Create applier
            lut_applier = LUT3DApplier(lut_np).to(device)
            # JIT Compile the LUT applier? 
            # If LUT size changes, this triggers recompilation. 
            # If LUT size is constant (e.g. 33x33x33), this is very fast.
            lut_applier = torch.compile(lut_applier)
            
            # Reshape image for grid_sample: (H, W, C) -> (1, H, W, C)
            batch_img = flog2_img.unsqueeze(0)
            
            final_img = lut_applier(batch_img)
            
            # Squeeze back
            final_img = final_img.squeeze(0)

            # --- Step D: Save (CPU) ---
            # Convert back to uint8 cpu numpy (jpeg)
            final_np = (final_img * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

            out_name = f"{base_name}_{lut_name}.jpeg"     
            iio.imwrite(output_path + "/" + out_name, final_np, quality=90)

            # --- Step E: Report
            print(f"  -> Auto-Exposure Applied Gain: {gain_applied:.4f}")
            print(f"  -> Equivalent ISO Push: {np.log2(gain_applied):.2f} stops")
            print(f"  -> Saved to {output_path}")


def main():
    """
    1. Define your target raw image
    2. Define folder containing your .cube files
    """
    parser = argparse.ArgumentParser(description="A PyTorch program that applies LUTs on RAW image")
    
    # Add arguments
    parser.add_argument("-i", "--image", type=str, help="Path to target raw image.")
    parser.add_argument("-l", "--lut", type=str, help="Folder that contains LUTs.")
    parser.add_argument("-o", "--output", type=str, default="./",  help="Folder that stores converted images.")
    args = parser.parse_args()

    process_pipeline(args.image, args.lut, args.output)

if __name__ == "__main__":
    main()