
import os
import numpy as np
import glob
import torch
import torchvision.models as models
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import matplotlib
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from sklearn.linear_model import LinearRegression
from shapely.geometry import Polygon, MultiPolygon
import alphashape
import matplotlib.patches as patches
from skan import Skeleton, summarize




# Functions for loading the trained model
def set_parameter_requires_grad(model, feature_extracting):
    # From: https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def to_long(mask, **kwargs):
    """ Convert mask to long datatype (np.int64)"""
    return mask.astype(np.int64)

def load_model(model_path, num_classes=5):
    """ Load trained model from specified path
    Args:
        model_path (str): Path to the trained model file."""
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
    set_parameter_requires_grad(model, feature_extracting=False)
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))
    model.aux_classifier[4]  = torch.nn.Conv2d(256, num_classes, kernel_size=(1,1), stride=(1,1))

    model.eval()
    load_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_device.type == 'cpu':
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(model_path))
    model.to(load_device)
    return model, load_device

# Function for applying the model to segment images
def predict_and_visualize(image_path, model, device, num_classes = 5, float32=True, norm_to_one=True, save = False, visualize = True):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    transforms = A.Compose([
    A.geometric.resize.LongestMaxSize(max_size = 960, interpolation = cv2.INTER_NEAREST),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1),
    A.Lambda(mask=to_long),
    ToTensorV2()])
        
    try:
        # Load and preprocess the image
        input_image = image_path
        if isinstance(image_path, str):
            input_image = cv2.imread(image_path)
        if input_image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        if norm_to_one:
            input_image = input_image.astype(np.float32) / 255.0
        if float32:
            input_image = input_image.astype(np.float32)
        transformed = (image=input_image)
        batch = torch.unsqueeze(transformed['image'], 0).to(device)

        im = batch[0].cpu().numpy()
        im = np.transpose(im, (1, 2, 0))

        with torch.no_grad():
            output = model(batch)['out'][0]

        #Calculate softmax probabilities
        softmax_output = torch.nn.functional.softmax(output, dim=0)

        # Convert output to predicted class
        original_height, original_width = input_image.shape[:2]
        aspect_ratio = original_height/original_width
        predicted_mask = torch.argmax(output, dim=0).cpu().numpy()

        # Visualize
        if visualize:
            fig, axes = plt.subplots(1, 5, figsize=(30, 10))

            axes[0].imshow(input_image, interpolation = 'nearest')
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            cmp = matplotlib.colors.ListedColormap(["#373533",  "#c2763e", "#3e8ac2", "#763ec2", "#8ac23e"])
            axes[1].imshow(predicted_mask, cmap=cmp, vmin=0, vmax=num_classes -1, interpolation = 'nearest')
            axes[1].set_title("Predicted Mask")
            axes[1].axis('off')

            # Plot softmax probabilities class 2
            probability_map = softmax_output[2].cpu().numpy()
            top = probability_map.max()
            im = axes[2].imshow(probability_map, cmap='inferno', vmin = 0, vmax = top, interpolation = 'nearest')
            axes[2].set_title("Probability Heatmap (Primary Venation)")
            axes[2].axis('off')
        
            #Plot softmax probabilities class 3
            probability_map = softmax_output[3].cpu().numpy()
            top = probability_map.max()
            im = axes[3].imshow(probability_map, cmap='inferno', vmin = 0, vmax = top, interpolation = 'nearest')
            axes[3].set_title("Probability Heatmap (Secondary Venation)")
            axes[3].axis('off')

            #Plot softmax probabilities class 1
            probability_map = softmax_output[1].cpu().numpy()
            top = probability_map.max()
            im = axes[4].imshow(probability_map, cmap='inferno', vmin = 0, vmax = top, interpolation = 'nearest')
            axes[4].set_title("Probability Heatmap (Margin)")
            axes[4].axis('off')

            #Plot scalebar for heatmaps
            cbar_ax = fig.add_axes([0.91, 0.25, 0.01, 0.5])
            fig.colorbar(im, cax = cbar_ax)

            plt.subplots_adjust(wspace = 0.02, hspace = 0)

        if save:
            fig, axes = plt.subplots(1, 5, figsize=(30, 10))

            axes[0].imshow(input_image, interpolation = 'nearest')
            axes[0].set_title("Original Image")
            axes[0].axis('off')

            cmp = matplotlib.colors.ListedColormap(["#373533",  "#c2763e", "#3e8ac2", "#763ec2", "#8ac23e"])
            axes[1].imshow(predicted_mask, cmap=cmp, vmin=0, vmax=num_classes -1, interpolation = 'nearest')
            axes[1].set_title("Predicted Mask")
            axes[1].axis('off')

            # Plot softmax probabilities class 2
            probability_map = softmax_output[2].cpu().numpy()
            top = probability_map.max()
            im = axes[2].imshow(probability_map, cmap='inferno', vmin = 0, vmax = top, interpolation = 'nearest')
            axes[2].set_title("Probability Heatmap (Primary Venation)")
            axes[2].axis('off')
        
            #Plot softmax probabilities class 3
            probability_map = softmax_output[3].cpu().numpy()
            top = probability_map.max()
            im = axes[3].imshow(probability_map, cmap='inferno', vmin = 0, vmax = top, interpolation = 'nearest')
            axes[3].set_title("Probability Heatmap (Secondary Venation)")
            axes[3].axis('off')

            #Plot softmax probabilities class 1
            probability_map = softmax_output[1].cpu().numpy()
            top = probability_map.max()
            im = axes[4].imshow(probability_map, cmap='inferno', vmin = 0, vmax = top, interpolation = 'nearest')
            axes[4].set_title("Probability Heatmap (Margin)")
            axes[4].axis('off')

            #Plot scalebar for heatmaps
            cbar_ax = fig.add_axes([0.91, 0.25, 0.01, 0.5])
            fig.colorbar(im, cax = cbar_ax)

            plt.subplots_adjust(wspace = 0.02, hspace = 0)
            
            # Save the figure as a prediction
            image_name = os.path.split(image_path)[1]
            image_name = os.path.splitext(image_name)[0]
            super_path = os.path.split(image_path)[0]
            new_path = os.path.join(super_path, "predictions")
            os.makedirs(new_path, exist_ok=True)
            final_path = os.path.join(new_path, image_name)
            final_path = final_path + ".png"
            print(final_path)
            plt.savefig(final_path, bbox_inches='tight')
            plt.close()
        return output

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

# Functions for automatic area reconstruction
def find_midvein_angle(image):
    """ Find the angle of the midvein in a leaf image using vertical histogram analysis."""
    best_angle = 0
    best_histogram_max = 0
    # Iterate over angles from -90 to 90 degrees
    for angle in range(-90, 91):
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))

        # Calculate the vertical histogram
        vertical_histogram = np.sum(rotated, axis=0)
        histogram_max = np.max(vertical_histogram)

        # Update the best angle if the current histogram sum is greater than the best one found so far
        if histogram_max > best_histogram_max:
            best_histogram_max = histogram_max
            best_angle = angle

    M_best = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rotated_best = cv2.warpAffine(image, M_best, (w, h))
    vertical_histogram_best = np.sum(rotated_best, axis=0)
    histogram_peak = np.argmax(vertical_histogram_best)

    return best_angle, histogram_peak

def measure_vein_length(region):
    """Compute skeletonized vein length inside a region."""
    return np.count_nonzero(region)

def process_veins(vein_map):
    """Process the vein map to extract the skeleton of the veins."""
    kernel = np.ones((5, 5), np.uint8)
    closed_vein_map = cv2.morphologyEx(vein_map, cv2.MORPH_OPEN, kernel)
    closed_vein_map = cv2.morphologyEx(closed_vein_map, cv2.MORPH_OPEN, kernel)

    skeleton = skeletonize(closed_vein_map, method='lee')
    skeleton = skeleton.astype(np.uint8)

    return skeleton

def project_veins_toward_midvein(region_skeleton, midvein_x):
    """Estimate vein fragment trajectories toward the midvein."""
    labeled_veins = label(region_skeleton)
    intersection_points = []

    for region in regionprops(labeled_veins):
        coords = region.coords

        # Filter out small vein fragments
        if len(coords) < 5:
            continue

        # Fit linear regression to vein trajectory
        x_vals = np.array([c[1] for c in coords]).reshape(-1, 1)
        y_vals = np.array([c[0] for c in coords])
        model = LinearRegression().fit(x_vals, y_vals)

        # Predict the midvein intersection point
        predicted_midvein_y = model.predict(np.array([[midvein_x]])).item()

        intersection_points.append(predicted_midvein_y)

    return intersection_points

def is_strong_secondary(region_skeleton, rect_width, rect_height, min_length_ratio=0.9):
    """Check if a strong, continuous secondary vein crosses the rectangle."""
    num_components, labels = cv2.connectedComponents(region_skeleton.astype(np.uint8))

    # Compute the size of each connected component (vein fragment)
    component_sizes = [np.count_nonzero(labels == i) for i in range(1, num_components)]

    # Check for regions with no veins
    if not component_sizes:
        return False

    smallest_component_size = min(component_sizes)
    total_vein_length = sum(component_sizes)

    # Filter by minimum length requirement
    if smallest_component_size < rect_width * min_length_ratio:
        return False 

    # Ensure total vein length spans significant part of rectangle
    if total_vein_length < rect_width * 0.5:
        return False

    return True

def reconstruct_area(image_path, conversion, plot = True, save = False):
    """Reconstruct leaf area from vein density and skeleton analysis.
    
    Args:
        image_path (str): Path to the leaf image.
        conversion (float): Conversion factor for pixel to cm².
        plot (bool): Whether to plot the results.
        save (bool): Whether to save the plots."""
    
    # Load in image to segment
    kernel = np.ones((5, 5), np.uint8)
    input_image = cv2.imread(image_path)
    conversion_factor = 1/conversion
    if input_image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # First round of segmentation
    testData = predict_and_visualize(input_image, save = False, visualize = False)
    predicted_mask = torch.argmax(testData, dim=0).cpu().numpy()
    vein_mask = np.where(predicted_mask == 2, 255, 0).astype(np.uint8)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)
    skeleton = process_veins(vein_mask)
    angle, midvein_coord = find_midvein_angle(skeleton)
    height, width, _ = input_image.shape
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(input_image, M, (width, height), flags = cv2.INTER_NEAREST)
    new_size = rotated.shape[:2]
    scale_ratio = np.sqrt((new_size[0] * new_size[1]) / (input_image.shape[0] * input_image.shape[1]))
    adjusted_conversion_factor = conversion_factor * scale_ratio

    #New predicted mask segmentation
    testData2 = predict_and_visualize(rotated, save = False, visualize = False)
    predicted_mask = torch.argmax(testData2, dim=0).cpu().numpy()

    original_height, original_width = input_image.shape[:2]
    new_height, new_width = predicted_mask.shape[:2]

    # Test for bad conversion
    if ((original_height/original_width) / (new_height/new_width)) > 1.1 or ((original_height/original_width) / (new_height/new_width)) < 0.9 :
        print("non aspect ratio preserving conversion")
        print(original_height/original_width)
        print(new_height/new_width)
    size_conversion = original_width/new_width

    # Begin manipulating the final segmentation
    grey_mask = np.where((predicted_mask == 1) | (predicted_mask == 3) | (predicted_mask == 2), 255, 0).astype(np.uint8)
    erode_mask = cv2.dilate(grey_mask, kernel, iterations=1)
    erode_mask = cv2.erode(erode_mask, kernel, iterations=1)
    erode_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
    erode_mask = cv2.morphologyEx(grey_mask, cv2.MORPH_OPEN, kernel)
    grey_mask = grey_mask * erode_mask

    # Find contours of the grey mask, and test for empty contours
    contours, hierarchy = cv2.findContours(grey_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0,0,0

    all_points = np.concatenate(contours).reshape(-1, 2)

    alpha_shape = alphashape.alphashape(all_points, 0.02)

    #Select the largest alpha shape
    largest_hull = None
    largest_area = 0
    cm_area = 0

    if isinstance(alpha_shape, Polygon):
        largest_hull = alpha_shape
        largest_area = alpha_shape.area
    elif isinstance(alpha_shape, MultiPolygon):
        for polygon in alpha_shape.geoms:
            area = polygon.area
            if area > largest_area:
                largest_area = area
                largest_hull = polygon

    #Make a mask of the largest alpha shape
    hull_mask = np.zeros_like(grey_mask)
    if largest_hull is not None:
        exterior_coords = np.array(largest_hull.exterior.coords, dtype=np.int32)
        cv2.fillPoly(hull_mask, [exterior_coords], 255)
        cm_area = (largest_area * size_conversion * size_conversion)*(adjusted_conversion_factor*adjusted_conversion_factor)
        hull_width = largest_hull.bounds[2] - largest_hull.bounds[0]
        hull_height = largest_hull.bounds[3] - largest_hull.bounds[1]
    if cm_area != 0:
        cm_area = cm_area
    else:
        cm_area = 0

    drawing = np.zeros((grey_mask.shape[0], grey_mask.shape[1], 3), np.uint8)

    vein_mask = np.where(predicted_mask == 3, 255, 0).astype(np.uint8)
    vein_mask = cv2.bitwise_and(vein_mask, hull_mask)
    vein_mask = cv2.morphologyEx(vein_mask, cv2.MORPH_CLOSE, kernel)

    skeleton = process_veins(vein_mask)
    figure_skeleton = cv2.dilate(skeleton, kernel, iterations=1)

    if np.any(skeleton):
        branch_data = summarize(Skeleton(skeleton), separator='_')
        total_length = sum(branch_data.branch_distance)
    else:
        print("Warning: Empty skeleton detected. Setting total length to 0.")
        total_length = 0

    # Define rectangle search parameters
    width = int(hull_width/4)
    height = int(hull_height/4)
    start = int(hull_width/8)

    if width < 20:
        width = 120
    if height < 40:
        height = 210
    if start < 5:
        start = 40

    # Draw rectangles across the skeleton
    rect_width_range = range(20, width, int(width/3))
    rect_height_range = range(40, height, int(height/3))
    rect_start_range = range(5, start, int(start/2))

    rectangles = []
    valid = 0

    for rect_width in rect_width_range:
        for rect_height in rect_height_range:
            for side in ["left", "right"]:
                for start in rect_start_range:
                    rect_x = midvein_coord - rect_width - start if side == "left" else midvein_coord + start
                for rect_y in range(50, hull_mask.shape[0]-50, 10):
                    if rect_x < 0 or rect_y < 0 or rect_x + rect_width >= hull_mask.shape[1] or rect_y + rect_height >= hull_mask.shape[0]:
                        continue
                    region_skeleton = skeleton[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
                    region_hull = hull_mask[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]
                    
                    if np.count_nonzero(region_hull) < rect_width * rect_height * 0.95:
                        continue

                    # Identify secondary vein intersection points with midvein
                    projected_midvein_intersections = project_veins_toward_midvein(region_skeleton, midvein_coord)

                    # Count unique secondary veins by clustering intersection points
                    unique_vein_count = len(set(np.round(projected_midvein_intersections)))  # Round to group close points

                    if unique_vein_count < 2:
                        valid+=1
                        continue

                    # Compute vein length & rectangle area
                    vein_length = measure_vein_length(region_skeleton) * size_conversion * conversion_factor
                    rectangle_area = (rect_width * size_conversion * rect_height * size_conversion) * conversion_factor ** 2  # Convert pixels to cm²
                    vein_density = np.log10(vein_length / rectangle_area) if rectangle_area > 0 else 0
                    reconstructed_area = 10**(1.63 - 1.69 * vein_density)

                    # Store the rectangle data
                    rectangles.append((reconstructed_area, rect_x, rect_y, rect_width, rect_height, vein_density))

    # Apply strong vein filtering during rectangle selection
    filtered_rectangles = []
    for reconstructed_area, rect_x, rect_y, rect_width, rect_height, vein_density in rectangles:
        region_skeleton = skeleton[rect_y:rect_y+rect_height, rect_x:rect_x+rect_width]

        if is_strong_secondary(region_skeleton, rect_width, rect_height):
            filtered_rectangles.append((reconstructed_area, rect_x, rect_y, rect_width, rect_height, vein_density))

    #Compute median density across the top rectangles
    average_density = np.median([r[5] for r in filtered_rectangles])
    area_calc = 10**(1.63 - 1.69 * average_density) if average_density > 0 else 0

    # Step 7: Plot skeleton & overlay top rectangles
    if plot:
        fig, axes = plt.subplots(1, 5, figsize=(30, 10))
        labels = ['A', 'B', 'C', 'D', 'E']

        for ax, label in zip(axes, labels):
            ax.text(0.05, 0.98, label, transform=ax.transAxes,
            fontsize=64, fontweight='bold', va='top', color = 'white')

        axes[0].imshow(rotated, interpolation = 'nearest')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        cmp = matplotlib.colors.ListedColormap(["#373533",  "#c2763e", "#3e8ac2", "#763ec2", "#8ac23e"])
        axes[1].imshow(predicted_mask, cmap = cmp)
        axes[1].set_title('Original Vein Map')
        axes[1].axis('off')
        axes[2].imshow(grey_mask, cmap=matplotlib.colors.ListedColormap(["#373533",  "#FFFFFF"]))
        axes[2].set_title('Fragment Hull')
        axes[2].axis('off')
        if largest_hull is not None:
            exterior_coords = np.array(largest_hull.exterior.coords)
            axes[2].plot(exterior_coords[:, 0], exterior_coords[:, 1], 'r-', linewidth=2)
        axes[3].imshow(figure_skeleton, cmap=matplotlib.colors.ListedColormap(["#373533",  "#FFFFFF"]))
        axes[3].set_title('Secondary Vein Skeleton')
        axes[3].axis('off')
        axes[4].imshow(figure_skeleton, cmap=matplotlib.colors.ListedColormap(["#373533",  "#FFFFFF"]))
        for _, rect_x, rect_y, w, h, _ in filtered_rectangles:
            rect = patches.Rectangle((rect_x, rect_y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
            axes[4].add_patch(rect)
        axes[4].set_title("Secondary Density Segments")
        axes[4].axis('off')
        plt.subplots_adjust(wspace = 0.02, hspace = 0)
        fig.tight_layout()
        plt.show()
    
    if save:
        fig, axes = plt.subplots(1, 5, figsize=(30, 10))
        labels = ['A', 'B', 'C', 'D', 'E']

        for ax, label in zip(axes, labels):
            ax.text(0.05, 0.98, label, transform=ax.transAxes,
            fontsize=64, fontweight='bold', va='top', color = 'white')

        axes[0].imshow(rotated, interpolation = 'nearest')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        cmp = matplotlib.colors.ListedColormap(["#373533",  "#c2763e", "#3e8ac2", "#763ec2", "#8ac23e"])
        axes[1].imshow(predicted_mask, cmap = cmp)
        axes[1].set_title('Original Vein Map')
        axes[1].axis('off')
        axes[2].imshow(grey_mask, cmap=matplotlib.colors.ListedColormap(["#373533",  "#FFFFFF"]))
        axes[2].set_title('Fragment Hull')
        axes[2].axis('off')
        if largest_hull is not None:
            exterior_coords = np.array(largest_hull.exterior.coords)
            axes[2].plot(exterior_coords[:, 0], exterior_coords[:, 1], 'r-', linewidth=2)
        axes[3].imshow(figure_skeleton, cmap=matplotlib.colors.ListedColormap(["#373533",  "#FFFFFF"]))
        axes[3].set_title('Secondary Vein Skeleton')
        axes[3].axis('off')
        axes[4].imshow(figure_skeleton, cmap=matplotlib.colors.ListedColormap(["#373533",  "#FFFFFF"]))
        for _, rect_x, rect_y, w, h, _ in filtered_rectangles:
            rect = patches.Rectangle((rect_x, rect_y), w, h, linewidth=2, edgecolor='yellow', facecolor='none')
            axes[4].add_patch(rect)
        axes[4].set_title("Secondary Density Segments")
        axes[4].axis('off')
        plt.subplots_adjust(wspace = 0.02, hspace = 0)
        fig.tight_layout()
    
        image_name = os.path.split(image_path)[1]
        image_name = os.path.splitext(image_name)[0]
        super_path = os.path.split(image_path)[0]
        new_path = os.path.join(super_path, "predictions")
        os.makedirs(new_path, exist_ok=True)
        final_path = os.path.join(new_path, image_name)
        final_path = final_path + "area_reconstruction.png"
        print(final_path)
        plt.savefig(final_path, bbox_inches='tight')
        plt.close()
        plt.show()

    return cm_area, area_calc, average_density, [r[-1] for r in filtered_rectangles] 
