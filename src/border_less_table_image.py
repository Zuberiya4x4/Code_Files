import cv2
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from io import BytesIO
import re
from collections import defaultdict
import pandas as pd
from sklearn.cluster import DBSCAN
from datetime import datetime
import pytesseract

def clean_cell_text(cell):
    """Clean cell content by removing newlines and trimming whitespace."""
    if isinstance(cell, str):
        return ' '.join(cell.split())
    return cell

def clean_header_name(header):
    """Clean header name for use as key"""
    if not header:
        return ""

    # Remove special characters and normalize spaces
    cleaned = re.sub(r'[^\w\s]', '', header)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())

    # Convert to title case and remove leading/trailing underscores
    cleaned = cleaned.title().strip('_')

    return cleaned if cleaned else header.replace(' ', '_')

def extract_text_blocks_with_positions_image(image, bbox=None):
    """Extract text blocks with their positions for borderless table detection from image"""
    # If bbox is provided, crop the image to that region
    if bbox:
        x1, y1, x2, y2 = bbox
        image_region = image[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1
    else:
        image_region = image
        offset_x, offset_y = 0, 0

    # Use pytesseract to get detailed text information
    try:
        data = pytesseract.image_to_data(image_region, output_type=pytesseract.Output.DICT)
    except Exception as e:
        print(f"OCR failed: {e}")
        return []

    text_blocks = []

    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30 and data['text'][i].strip():  # Filter low confidence
            text = data['text'][i].strip()
            left = data['left'][i] + offset_x
            top = data['top'][i] + offset_y
            width = data['width'][i]
            height = data['height'][i]

            text_blocks.append({
                "text": text,
                "bbox": [left, top, left + width, top + height],
                "x": left,
                "y": top,
                "width": width,
                "height": height,
                "center_x": left + width / 2,
                "center_y": top + height / 2
            })

    return text_blocks

def cluster_coordinates(coords, eps=10):
    """Cluster coordinates using DBSCAN to find groups of similar positions"""
    if not coords:
        return []

    coords = np.array(coords).reshape(-1, 1)
    clustering = DBSCAN(eps=eps, min_samples=1).fit(coords)
    labels = clustering.labels_

    # Calculate cluster centers
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(coords[i][0])

    # Return sorted cluster centers
    return sorted([np.mean(cluster) for cluster in clusters.values()])

def create_virtual_grid_image(text_blocks, tolerance=10):
    """Create virtual horizontal and vertical grid lines from text blocks"""
    if not text_blocks:
        return [], []

    # Extract all x and y coordinates
    x_coords = [block["x"] for block in text_blocks] + [block["x"] + block["width"] for block in text_blocks]
    y_coords = [block["y"] for block in text_blocks] + [block["y"] + block["height"] for block in text_blocks]

    # Cluster coordinates to form virtual lines
    x_lines = cluster_coordinates(x_coords, eps=tolerance)
    y_lines = cluster_coordinates(y_coords, eps=tolerance)

    return x_lines, y_lines

def extract_table_with_virtual_grid_image(image, table_bbox=None, tolerance=10):
    """Extract table using virtual grid lines inferred from text positions in image"""
    # Get text blocks within the image or specified bbox
    text_blocks = extract_text_blocks_with_positions_image(image, table_bbox)

    if not text_blocks:
        return []

    # Create virtual grid lines
    x_lines, y_lines = create_virtual_grid_image(text_blocks, tolerance)

    if len(x_lines) < 2 or len(y_lines) < 2:
        return []

    # Create table data from virtual grid
    table_data = []

    # Track which text blocks have been assigned to avoid duplication
    assigned_blocks = set()

    # Process each row (between horizontal lines)
    for i in range(len(y_lines) - 1):
        row_data = []
        y_top = y_lines[i]
        y_bottom = y_lines[i+1]

        # Process each column (between vertical lines)
        for j in range(len(x_lines) - 1):
            x_left = x_lines[j]
            x_right = x_lines[j+1]

            # Extract text in this cell
            cell_text = ""

            for block_idx, block in enumerate(text_blocks):
                if block_idx in assigned_blocks:
                    continue

                # Check if the center of the text block is within this cell
                if (x_left <= block["center_x"] <= x_right and
                    y_top <= block["center_y"] <= y_bottom):
                    cell_text += block["text"] + " "
                    assigned_blocks.add(block_idx)

            row_data.append(clean_cell_text(cell_text))

        # Only add row if it has non-empty cells
        if any(cell.strip() for cell in row_data):
            table_data.append(row_data)

    return table_data

def detect_rows_and_columns_image(text_blocks, y_tolerance=15, x_tolerance=25):
    """Detect rows and columns from text blocks in image"""
    if not text_blocks:
        return [], []

    # Sort blocks by Y position first, then by X position
    sorted_blocks = sorted(text_blocks, key=lambda b: (b["y"], b["x"]))

    # Group blocks into rows based on Y position
    rows = []
    current_row = [sorted_blocks[0]]

    for block in sorted_blocks[1:]:
        # Check if this block is on the same row (similar Y position)
        if abs(block["y"] - current_row[0]["y"]) <= y_tolerance:
            current_row.append(block)
        else:
            # Sort current row by X position and add to rows
            current_row.sort(key=lambda b: b["x"])
            rows.append(current_row)
            current_row = [block]

    # Don't forget the last row
    if current_row:
        current_row.sort(key=lambda b: b["x"])
        rows.append(current_row)

    # Detect column positions by analyzing X coordinates across all rows
    all_x_positions = []
    for row in rows:
        for block in row:
            all_x_positions.append(block["x"])

    # Sort and group X positions to find column boundaries
    all_x_positions.sort()
    column_positions = []

    if all_x_positions:
        current_group = [all_x_positions[0]]

        for x in all_x_positions[1:]:
            if x - current_group[-1] <= x_tolerance:
                current_group.append(x)
            else:
                # Take the median of the current group as the column position
                column_positions.append(sum(current_group) / len(current_group))
                current_group = [x]

        # Don't forget the last group
        if current_group:
            column_positions.append(sum(current_group) / len(current_group))

    return rows, column_positions

def create_table_from_rows_and_columns_image(rows, column_positions, x_tolerance=25):
    """Create a table matrix from detected rows and columns in image"""
    if not rows or not column_positions:
        return []

    table_matrix = []

    for row in rows:
        table_row = [""] * len(column_positions)

        for block in row:
            # Find which column this block belongs to
            best_col = 0
            min_distance = float('inf')

            for col_idx, col_x in enumerate(column_positions):
                distance = abs(block["x"] - col_x)
                if distance < min_distance and distance <= x_tolerance:
                    min_distance = distance
                    best_col = col_idx

            # Add text to the appropriate column
            if table_row[best_col]:
                table_row[best_col] += " " + block["text"]
            else:
                table_row[best_col] = block["text"]

        table_matrix.append(table_row)

    return table_matrix

def extract_table_with_text_analysis_image(image, table_bbox=None, y_tolerance=15, x_tolerance=25):
    """Extract table by analyzing text layout and patterns in image"""
    # Get text blocks within the image or specified bbox
    text_blocks = extract_text_blocks_with_positions_image(image, table_bbox)

    if not text_blocks:
        return []

    # Detect rows and columns
    rows, column_positions = detect_rows_and_columns_image(text_blocks, y_tolerance, x_tolerance)

    # Create table matrix
    table_matrix = create_table_from_rows_and_columns_image(rows, column_positions, x_tolerance)

    return table_matrix

def detect_borderless_table_regions_image(image):
    """Detect potential borderless table regions in image based on text density"""
    # Get all text blocks
    text_blocks = extract_text_blocks_with_positions_image(image)

    if not text_blocks:
        return []

    # If we have text blocks, consider the entire image as a potential table region
    # In a more sophisticated version, you could cluster text blocks to find dense regions
    height, width = image.shape[:2]

    # For now, return the entire image as a potential table region
    # You can enhance this to detect multiple regions based on text clustering
    return [(0, 0, width, height)]

def matrix_to_column_dict(matrix, table_name="Table"):
    """Convert a matrix to column-oriented dictionary"""
    if not matrix:
        return {}

    # Clean the matrix first - remove completely empty rows
    cleaned_matrix = []
    for row in matrix:
        if any(cell and str(cell).strip() for cell in row):
            cleaned_matrix.append(row)

    if not cleaned_matrix:
        return {}

    # Ensure all rows have the same length
    max_cols = max(len(row) for row in cleaned_matrix) if cleaned_matrix else 0
    for row in cleaned_matrix:
        while len(row) < max_cols:
            row.append("")

    # Determine if we have headers
    has_headers = False
    headers = []

    if len(cleaned_matrix) > 1:
        # Check if first row looks like headers
        first_row = cleaned_matrix[0]
        text_cells = sum(1 for cell in first_row
                        if cell and isinstance(cell, str) and not re.match(r'^\d+([.,]\d+)*$', cell.strip()))

        if text_cells >= len(first_row) * 0.6:  # At least 60% text cells
            has_headers = True
            headers = [clean_header_name(str(cell)) if cell else f"Column_{i+1}"
                      for i, cell in enumerate(first_row)]

    # If no headers, generate generic column names
    if not has_headers:
        headers = [f"Column_{i+1}" for i in range(max_cols)]

    # Initialize column dictionary
    column_dict = {header: [] for header in headers}

    # Process data rows
    data_rows = cleaned_matrix[1:] if has_headers else cleaned_matrix

    for row in data_rows:
        for col_idx, header in enumerate(headers):
            if col_idx < len(row):
                value = clean_cell_text(row[col_idx])
            else:
                value = ""
            column_dict[header].append(value)

    return column_dict

def convert_to_dataframe(result):
    """Convert extraction result to pandas DataFrame"""
    all_data = []

    tables = result.get("tables", [])
    for table in tables:
        table_id = table.get("table_id", "")
        column_data = table.get("column_data", {})

        # Convert column-oriented data to rows
        if column_data:
            # Get the maximum number of rows in any column
            max_rows = max(len(values) for values in column_data.values()) if column_data else 0

            # Create rows from column data
            for row_idx in range(max_rows):
                row_data = {
                    "table_id": table_id,
                    "row_index": row_idx + 1
                }

                # Add each column's value for this row
                for col_name, values in column_data.items():
                    if row_idx < len(values):
                        row_data[col_name] = values[row_idx]
                    else:
                        row_data[col_name] = ""

                all_data.append(row_data)

    return pd.DataFrame(all_data)

def save_json_result(result, output_dir="json_output"):
    """Save extraction result to JSON file"""
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"borderless_table_extraction_result_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)

    # Save JSON to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"JSON result saved to: {output_path}")
    return output_path

def enhanced_borderless_extraction_image_kv(image_path):
    """Enhanced extraction for borderless tables from images using multiple approaches"""
    if not os.path.exists(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Could not load image: {image_path}"}

    print(f"Processing image: {image_path}")

    # Detect potential table regions
    table_regions = detect_borderless_table_regions_image(image)
    extracted_tables = []

    for region_idx, bbox in enumerate(table_regions):
        print(f"Region {region_idx + 1}: Processing with multiple methods")

        # Try different extraction methods
        methods = [
            ("Virtual Grid", lambda img, bb: extract_table_with_virtual_grid_image(img, bb)),
            ("Text Analysis", lambda img, bb: extract_table_with_text_analysis_image(img, bb))
        ]

        best_table = None
        best_score = 0
        best_method = ""

        for method_name, method_func in methods:
            print(f"  Trying {method_name} method...")
            raw_matrix = method_func(image, bbox)

            # Clean and process the matrix
            cleaned_matrix = []
            for row in raw_matrix:
                cleaned_row = []
                for cell in row:
                    if cell is None:
                        cleaned_row.append("")
                    else:
                        cleaned_row.append(clean_cell_text(str(cell)))
                cleaned_matrix.append(cleaned_row)

            # Remove empty rows and columns
            cleaned_matrix = [row for row in cleaned_matrix if any(cell.strip() for cell in row)]

            # Remove duplicate rows
            unique_rows = []
            seen_rows = set()
            for row in cleaned_matrix:
                row_tuple = tuple(row)
                if row_tuple not in seen_rows:
                    seen_rows.add(row_tuple)
                    unique_rows.append(row)

            cleaned_matrix = unique_rows

            # Score the table based on structure quality
            if cleaned_matrix and len(cleaned_matrix) >= 2:
                # Calculate a score based on row consistency and non-empty cells
                row_lengths = [len(row) for row in cleaned_matrix]
                consistent_rows = len(set(row_lengths)) == 1
                non_empty_cells = sum(1 for row in cleaned_matrix for cell in row if cell.strip())
                total_cells = len(cleaned_matrix) * max(row_lengths) if row_lengths else 0
                fill_ratio = non_empty_cells / total_cells if total_cells > 0 else 0

                # Score is higher for tables with consistent row lengths and good fill ratio
                score = (10 if consistent_rows else 5) + fill_ratio * 10

                if score > best_score:
                    best_score = score
                    best_table = cleaned_matrix
                    best_method = method_name
                    print(f"    New best score: {score:.2f} using {method_name} (consistent: {consistent_rows}, fill: {fill_ratio:.2f})")
            else:
                print(f"    No valid table structure found")

        # If we have a valid table, add it to the results
        if best_table:
            table_tag = f"EnhancedBorderlessTable_{region_idx + 1}"
            column_data = matrix_to_column_dict(best_table, table_tag)

            extracted_tables.append({
                "table_id": table_tag,
                "table_type": "borderless",
                "extraction_method": best_method,
                "bbox": list(bbox),
                "score": best_score,
                "column_data": column_data
            })
            print(f"  Successfully extracted table with {len(column_data)} columns using {best_method} (score: {best_score:.2f})")
        else:
            print(f"  No valid table structure found in region")

    return {
        "file_name": os.path.basename(image_path),
        "tables_extracted": len(extracted_tables),
        "extraction_type": "borderless_tables_image_enhanced_with_multiple_methods",
        "tables": extracted_tables,
        "success": bool(extracted_tables)
    }

def primary_extraction_borderless_image_kv(image_path):
    """Primary extraction for borderless tables from images using basic OCR strategy"""
    if not os.path.exists(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Could not load image: {image_path}"}

    print(f"Processing image: {image_path}")

    extracted_tables = []

    # Extract text using basic OCR approach
    try:
        # Get text with bounding boxes
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

        # Group text by lines (rows)
        lines = defaultdict(list)
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Filter low confidence text
                text = data['text'][i].strip()
                if text:
                    top = data['top'][i]
                    # Group by similar y-coordinates (rows)
                    line_key = round(top / 20) * 20  # Group within 20 pixels
                    lines[line_key].append((data['left'][i], text))

        # Sort lines by y-coordinate and create matrix
        raw_matrix = []
        for line_y in sorted(lines.keys()):
            # Sort words in each line by x-coordinate
            line_words = sorted(lines[line_y], key=lambda x: x[0])
            row = [word[1] for word in line_words]
            raw_matrix.append(row)

        # Clean the matrix
        cleaned_matrix = []
        for row in raw_matrix:
            cleaned_row = [clean_cell_text(str(cell)) for cell in row]
            cleaned_matrix.append(cleaned_row)

        # Remove empty rows
        cleaned_matrix = [row for row in cleaned_matrix if any(cell.strip() for cell in row)]

        if cleaned_matrix:
            table_tag = "BorderlessTable_1"
            column_data = matrix_to_column_dict(cleaned_matrix, table_tag)

            extracted_tables.append({
                "table_id": table_tag,
                "table_type": "borderless",
                "extraction_method": "primary_borderless_image",
                "bbox": [0, 0, image.shape[1], image.shape[0]],
                "column_data": column_data
            })

    except Exception as e:
        print(f"OCR extraction failed: {e}")

    return {
        "file_name": os.path.basename(image_path),
        "tables_extracted": len(extracted_tables),
        "extraction_type": "borderless_tables_image_primary",
        "tables": extracted_tables,
        "success": bool(extracted_tables)
    }

def visualize_borderless_tables_image(image_path, extraction_function, output_dir="borderless_plots"):
    """Visualize borderless table extraction results for images"""
    if not os.path.exists(image_path):
        print(f"Error: File not found: {image_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Get extraction results
    result = extraction_function(image_path)

    if not result["success"]:
        print(f"Extraction failed: {result.get('error', 'Unknown error')}")
        return

    print(f"Total tables extracted: {result['tables_extracted']}")

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create matplotlib figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 12))

    # Left side - Image with virtual grid
    ax1.imshow(image_rgb)
    ax1.set_title(f'Image - Borderless Tables Detection', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Draw table regions and virtual grids
    for table_idx, table in enumerate(result["tables"]):
        bbox = table.get("bbox", [0, 0, image.shape[1], image.shape[0]])

        # Draw table boundary
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=2, edgecolor='green', facecolor='none', alpha=0.7
        )
        ax1.add_patch(rect)
        ax1.text(bbox[0], bbox[1]-10, f'{table["table_id"]} ({table["extraction_method"]}, Score: {table.get("score", 0):.1f})',
                color='green', fontweight='bold', fontsize=10)

        # Draw virtual grid lines for this table
        text_blocks = extract_text_blocks_with_positions_image(image, bbox)
        x_lines, y_lines = create_virtual_grid_image(text_blocks)

        # Draw virtual horizontal lines
        for y in y_lines:
            ax1.axhline(y=y, xmin=bbox[0]/image.shape[1], xmax=bbox[2]/image.shape[1],
                       color='blue', linestyle='--', alpha=0.5)

        # Draw virtual vertical lines
        for x in x_lines:
            ax1.axvline(x=x, ymin=bbox[1]/image.shape[0], ymax=bbox[3]/image.shape[0],
                       color='red', linestyle='--', alpha=0.5)

    # Right side - Column data
    ax2.axis('off')
    ax2.set_title(f'Extracted Column Data (Borderless Tables)', fontsize=12, fontweight='bold')

    tables_data = result["tables"]
    print(f"Displaying {len(tables_data)} tables")

    if tables_data:
        y_pos = 0.95
        for table_idx, table in enumerate(tables_data):
            # Table header
            ax2.text(0.02, y_pos, f"{table['table_id']}:",
                    fontsize=12, fontweight='bold', color='darkred', transform=ax2.transAxes)
            y_pos -= 0.04

            column_data = table["column_data"]
            print(f"  Table {table_idx+1} has {len(column_data)} columns")

            for col_name, values in column_data.items():
                # Column header
                ax2.text(0.04, y_pos, f"{col_name}:", fontsize=10, fontweight='bold',
                        color='darkblue', transform=ax2.transAxes)
                y_pos -= 0.03

                # Column values (show first 5)
                for i, value in enumerate(values[:5]):
                    display_text = f"  [{i+1}] {value}"
                    if len(display_text) > 80:
                        display_text = display_text[:77] + "..."
                    ax2.text(0.06, y_pos, display_text, fontsize=9,
                            color='black', transform=ax2.transAxes)
                    y_pos -= 0.025

                # Show if there are more values
                if len(values) > 5:
                    ax2.text(0.06, y_pos, f"  ... and {len(values) - 5} more values",
                            fontsize=9, style='italic', color='gray', transform=ax2.transAxes)
                    y_pos -= 0.025

                if y_pos < 0.05:  # Prevent text from going off screen
                    ax2.text(0.02, y_pos, "... (more data truncated for display)",
                            fontsize=9, style='italic', color='gray', transform=ax2.transAxes)
                    break

            y_pos -= 0.02  # Space between tables
            if y_pos < 0.05:
                break
    else:
        ax2.text(0.02, 0.5, "No borderless tables found in this image",
                fontsize=12, color='red', transform=ax2.transAxes)

    # Save the plot
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_borderless_tables.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved borderless table visualization to: {output_path}")

    plt.show()
    plt.close()

def interactive_borderless_analysis_image():
    """Interactive function for borderless table extraction analysis on images"""
    print("=== BORDERLESS TABLES IMAGE EXTRACTION ANALYSIS ===")

    # Get image path from user
    image_path = input("Enter the path to your image file: ").strip()
    image_path = image_path.strip('"\'')

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return

    print(f"\nProcessing: {image_path}")
    print("-" * 50)

    # Show extraction options
    print("=== BORDERLESS TABLE EXTRACTION METHODS ===")
    print("1. Primary borderless extraction → Column-oriented")
    print("2. Enhanced borderless extraction with multiple methods → Column-oriented")
    print("3. Both methods comparison")

    choice = input("Choose method (1-3, default=2): ").strip()
    if not choice:
        choice = "2"  # Default to enhanced method

    results = {}

    if choice in ["1", "3"]:
        print("\n=== PRIMARY BORDERLESS EXTRACTION COLUMN-ORIENTED OUTPUT ===")
        result1 = primary_extraction_borderless_image_kv(image_path)
        print(json.dumps(result1, indent=2, ensure_ascii=False))

        # Save JSON result to file
        if result1["success"]:
            save_json_result(result1)

        # Convert to DataFrame and display
        df1 = convert_to_dataframe(result1)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df1)

        results["primary"] = result1

    if choice in ["2", "3"]:
        print("\n=== ENHANCED BORDERLESS EXTRACTION WITH MULTIPLE METHODS COLUMN-ORIENTED OUTPUT ===")
        result2 = enhanced_borderless_extraction_image_kv(image_path)
        print(json.dumps(result2, indent=2, ensure_ascii=False))

        # Save JSON result to file
        if result2["success"]:
            save_json_result(result2)

        # Convert to DataFrame and display
        df2 = convert_to_dataframe(result2)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df2)

        results["enhanced"] = result2

    # Visualization option
    print("\n" + "=" * 50)
    print("=== VISUALIZATION OPTIONS ===")
    vis_choice = input("Show visualization? (y/n, default=y): ").strip().lower()

    if vis_choice != 'n':
        if choice == "1":
            visualize_borderless_tables_image(image_path, primary_extraction_borderless_image_kv)
        elif choice == "2":
            visualize_borderless_tables_image(image_path, enhanced_borderless_extraction_image_kv)
        else:
            # Show enhanced by default for comparison
            visualize_borderless_tables_image(image_path, enhanced_borderless_extraction_image_kv)

    return results

if __name__ == "__main__":
    # Check if running in interactive mode
    use_interactive = input("Use interactive mode for Borderless Tables extraction from images? (y/n, default=y): ").strip().lower()

    if use_interactive != 'n':
        # Interactive mode
        interactive_borderless_analysis_image()
    else:
        # Default image path - replace with your actual path
        image_path = "/path/to/your/borderless_table_image.jpg"

        # Check file exists
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            print("Please provide a valid image path")
            exit()

        # Run enhanced borderless table extraction (default)
        print("=== ENHANCED BORDERLESS EXTRACTION WITH MULTIPLE METHODS COLUMN-ORIENTED OUTPUT ===")
        result = enhanced_borderless_extraction_image_kv(image_path)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        # Save JSON result to file
        if result["success"]:
            save_json_result(result)

        # Convert to DataFrame and display
        df = convert_to_dataframe(result)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df)

        # Show visualization
        print("\nGenerating visualization...")
        visualize_borderless_tables_image(image_path, enhanced_borderless_extraction_image_kv)