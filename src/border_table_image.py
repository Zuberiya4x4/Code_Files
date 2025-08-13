# IMAGE TABLE EXTRACTION CODE
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

def detect_lines_in_image(image):
    """Detect horizontal and vertical lines in the image using OpenCV"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

    # Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    return horizontal_lines, vertical_lines

def find_line_coordinates(line_image, orientation='horizontal'):
    """Find coordinates of lines from binary line image"""
    contours, _ = cv2.findContours(line_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if orientation == 'horizontal' and w > 20:  # Filter short lines
            lines.append((x, y + h//2, x + w, y + h//2))  # (x0, y0, x1, y1)
        elif orientation == 'vertical' and h > 20:  # Filter short lines
            lines.append((x + w//2, y, x + w//2, y + h))  # (x0, y0, x1, y1)

    return lines

def extract_table_with_row_grouping_image(image, table_bbox=None):
    """Extract table using row grouping with horizontal and vertical lines from image"""
    # Detect lines in the image
    horizontal_line_img, vertical_line_img = detect_lines_in_image(image)

    # Get line coordinates
    horizontal_lines = find_line_coordinates(horizontal_line_img, 'horizontal')
    vertical_lines = find_line_coordinates(vertical_line_img, 'vertical')

    # If a table bbox is provided, filter lines within that bbox
    if table_bbox:
        x_min, y_min, x_max, y_max = table_bbox
        horizontal_lines = [line for line in horizontal_lines
                          if line[0] >= x_min and line[2] <= x_max and
                             line[1] >= y_min and line[1] <= y_max]
        vertical_lines = [line for line in vertical_lines
                        if line[0] >= x_min and line[0] <= x_max and
                           line[1] >= y_min and line[3] <= y_max]

    # Sort lines by position
    horizontal_lines.sort(key=lambda line: line[1])  # Sort by y-coordinate
    vertical_lines.sort(key=lambda line: line[0])    # Sort by x-coordinate

    # Group lines by proximity (row grouping for horizontal lines)
    tolerance = 10  # Tolerance for grouping lines

    # Group horizontal lines (rows)
    if not horizontal_lines:
        return []

    row_groups = []
    current_group = [horizontal_lines[0]]

    for i in range(1, len(horizontal_lines)):
        if abs(horizontal_lines[i][1] - current_group[-1][1]) <= tolerance:
            current_group.append(horizontal_lines[i])
        else:
            # Calculate average y-coordinate for the group
            avg_y = sum(line[1] for line in current_group) / len(current_group)
            row_groups.append(avg_y)
            current_group = [horizontal_lines[i]]

    if current_group:
        avg_y = sum(line[1] for line in current_group) / len(current_group)
        row_groups.append(avg_y)

    # Group vertical lines (columns)
    if not vertical_lines:
        return []

    col_groups = []
    current_group = [vertical_lines[0]]

    for i in range(1, len(vertical_lines)):
        if abs(vertical_lines[i][0] - current_group[-1][0]) <= tolerance:
            current_group.append(vertical_lines[i])
        else:
            # Calculate average x-coordinate for the group
            avg_x = sum(line[0] for line in current_group) / len(current_group)
            col_groups.append(avg_x)
            current_group = [vertical_lines[i]]

    if current_group:
        avg_x = sum(line[0] for line in current_group) / len(current_group)
        col_groups.append(avg_x)

    # Create table data from row and column groups
    table_data = []

    # Process each row (between horizontal line groups)
    for i in range(len(row_groups) - 1):
        row_data = []
        y_top = int(row_groups[i])
        y_bottom = int(row_groups[i+1])

        # Process each column (between vertical line groups)
        for j in range(len(col_groups) - 1):
            x_left = int(col_groups[j])
            x_right = int(col_groups[j+1])

            # Extract cell image
            cell_image = image[y_top:y_bottom, x_left:x_right]

            # Extract text using OCR
            try:
                cell_text = pytesseract.image_to_string(cell_image, config='--psm 6').strip()
            except:
                cell_text = ""

            row_data.append(clean_cell_text(cell_text))

        # Only add row if it has non-empty cells
        if any(cell.strip() for cell in row_data):
            table_data.append(row_data)

    return table_data

def detect_table_regions(image):
    """Detect potential table regions in the image"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply threshold
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Detect horizontal and vertical lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    # Combine horizontal and vertical lines
    table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)

    # Find contours to identify table regions
    contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter based on size (tables should be reasonably large)
        if w > 100 and h > 50:
            table_regions.append((x, y, x + w, y + h))

    return table_regions

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

    # Determine if we have headers
    has_headers = False
    headers = []

    if len(cleaned_matrix) > 1:
        # Check if first row looks like headers
        first_row = cleaned_matrix[0]
        if all(isinstance(cell, str) and cell.strip() for cell in first_row if cell):
            has_headers = True
            headers = [clean_header_name(str(cell)) for cell in first_row]

    # If no headers, generate generic column names
    if not has_headers:
        num_cols = max(len(row) for row in cleaned_matrix) if cleaned_matrix else 0
        headers = [f"Column_{i+1}" for i in range(num_cols)]

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

def primary_extraction_bordered_image_kv(image_path):
    """Primary extraction for bordered tables in images with key-value pair output"""
    if not os.path.exists(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Could not load image: {image_path}"}

    # Detect table regions
    table_regions = detect_table_regions(image)
    print(f"Found {len(table_regions)} potential table regions")

    extracted_tables = []

    for table_idx, bbox in enumerate(table_regions):
        print(f"Processing Table {table_idx+1}")

        # Extract table region
        x1, y1, x2, y2 = bbox
        table_image = image[y1:y2, x1:x2]

        # Use OCR to extract text from the entire table region
        try:
            # Get text with bounding boxes
            data = pytesseract.image_to_data(table_image, output_type=pytesseract.Output.DICT)

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
            matrix = []
            for line_y in sorted(lines.keys()):
                # Sort words in each line by x-coordinate
                line_words = sorted(lines[line_y], key=lambda x: x[0])
                row = [word[1] for word in line_words]
                matrix.append(row)

        except Exception as e:
            print(f"OCR failed for table {table_idx+1}: {e}")
            matrix = []

        if matrix:
            table_tag = f"BorderedTable_{table_idx+1}"
            column_data = matrix_to_column_dict(matrix, table_tag)

            extracted_tables.append({
                "table_id": table_tag,
                "table_type": "bordered",
                "extraction_method": "primary_bordered_image",
                "bbox": bbox,
                "column_data": column_data
            })
        else:
            print(f"Table {table_idx+1} is empty after processing")

    return {
        "file_name": os.path.basename(image_path),
        "tables_extracted": len(extracted_tables),
        "extraction_type": "bordered_tables_image",
        "tables": extracted_tables,
        "success": bool(extracted_tables)
    }

def enhanced_bordered_extraction_image_kv(image_path):
    """Enhanced extraction for bordered tables in images using line detection and row grouping"""
    if not os.path.exists(image_path):
        return {"success": False, "error": f"File not found: {image_path}"}

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return {"success": False, "error": f"Could not load image: {image_path}"}

    # Detect table regions
    table_regions = detect_table_regions(image)
    print(f"Found {len(table_regions)} potential table regions")

    extracted_tables = []

    for table_idx, bbox in enumerate(table_regions):
        print(f"Processing Table {table_idx+1}")

        # Extract table region
        x1, y1, x2, y2 = bbox
        table_image = image[y1:y2, x1:x2]

        # Try to extract using row grouping with line detection
        raw_matrix = extract_table_with_row_grouping_image(table_image, None)

        # If row grouping didn't work, fall back to standard OCR
        if not raw_matrix or all(not any(cell.strip() for cell in row) for row in raw_matrix):
            print(f"Row grouping failed for Table {table_idx+1}, falling back to standard OCR")
            try:
                # Get text with bounding boxes
                data = pytesseract.image_to_data(table_image, output_type=pytesseract.Output.DICT)

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

            except Exception as e:
                print(f"Fallback OCR failed for table {table_idx+1}: {e}")
                raw_matrix = []

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

        # Remove empty rows
        cleaned_matrix = [row for row in cleaned_matrix if any(cell.strip() for cell in row)]

        if cleaned_matrix:
            table_tag = f"BorderedTable_{table_idx+1}"
            column_data = matrix_to_column_dict(cleaned_matrix, table_tag)

            extracted_tables.append({
                "table_id": table_tag,
                "table_type": "bordered",
                "extraction_method": "enhanced_bordered_image_with_row_grouping",
                "bbox": bbox,
                "column_data": column_data
            })
        else:
            print(f"Table {table_idx+1} is empty after processing")

    return {
        "file_name": os.path.basename(image_path),
        "tables_extracted": len(extracted_tables),
        "extraction_type": "bordered_tables_image_enhanced_with_row_grouping",
        "tables": extracted_tables,
        "success": bool(extracted_tables)
    }

def visualize_bordered_tables_image(image_path, extraction_function, output_dir="bordered_plots"):
    """Visualize bordered table extraction results for images"""
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

    # Left side - Image with table boundaries and detected lines
    ax1.imshow(image_rgb)
    ax1.set_title(f'Image - Bordered Tables Detection', fontsize=12, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Detect and draw lines
    horizontal_line_img, vertical_line_img = detect_lines_in_image(image)
    horizontal_lines = find_line_coordinates(horizontal_line_img, 'horizontal')
    vertical_lines = find_line_coordinates(vertical_line_img, 'vertical')

    # Draw detected lines
    for line in horizontal_lines:
        ax1.plot([line[0], line[2]], [line[1], line[3]], 'b-', linewidth=1, alpha=0.5)
    for line in vertical_lines:
        ax1.plot([line[0], line[2]], [line[1], line[3]], 'r-', linewidth=1, alpha=0.5)

    # Draw table boundaries
    for i, table in enumerate(result["tables"]):
        bbox = table["bbox"]
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1],
            linewidth=2, edgecolor='green', facecolor='none', alpha=0.7
        )
        ax1.add_patch(rect)
        ax1.text(bbox[0], bbox[1]-10, f'Table_{i+1}',
                color='green', fontweight='bold', fontsize=10)

    # Right side - Column data
    ax2.axis('off')
    ax2.set_title(f'Extracted Column Data (Bordered Tables)', fontsize=12, fontweight='bold')

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
        ax2.text(0.02, 0.5, "No bordered tables found in this image",
                fontsize=12, color='red', transform=ax2.transAxes)

    # Save the plot
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}_bordered_tables.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved bordered table visualization to: {output_path}")

    plt.show()
    plt.close()

def interactive_bordered_analysis_image():
    """Interactive function for bordered table extraction analysis on images"""
    print("=== BORDERED TABLES IMAGE EXTRACTION ANALYSIS ===")

    # Get image path from user
    image_path = input("Enter the path to your image file: ").strip()
    image_path = image_path.strip('"\'')

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found!")
        return

    print(f"\nProcessing: {image_path}")
    print("-" * 50)

    # Show extraction options
    print("=== BORDERED TABLE EXTRACTION METHODS ===")
    print("1. Primary bordered extraction → Column-oriented")
    print("2. Enhanced bordered extraction with row grouping → Column-oriented")
    print("3. Both methods comparison")

    choice = input("Choose method (1-3, default=3): ").strip()
    if not choice:
        choice = "3"

    if choice in ["1", "3"]:
        print("\n=== PRIMARY BORDERED EXTRACTION COLUMN-ORIENTED OUTPUT ===")
        result1 = primary_extraction_bordered_image_kv(image_path)
        print(json.dumps(result1, indent=2, ensure_ascii=False))

        # Convert to DataFrame and display
        df1 = convert_to_dataframe(result1)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df1)

    if choice in ["2", "3"]:
        print("\n=== ENHANCED BORDERED EXTRACTION WITH ROW GROUPING COLUMN-ORIENTED OUTPUT ===")
        result2 = enhanced_bordered_extraction_image_kv(image_path)
        print(json.dumps(result2, indent=2, ensure_ascii=False))

        # Convert to DataFrame and display
        df2 = convert_to_dataframe(result2)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df2)

    # Visualization option
    print("\n" + "=" * 50)
    print("=== VISUALIZATION OPTIONS ===")
    vis_choice = input("Show visualization? (y/n, default=y): ").strip().lower()

    if vis_choice != 'n':
        if choice == "1":
            visualize_bordered_tables_image(image_path, primary_extraction_bordered_image_kv)
        elif choice == "2":
            visualize_bordered_tables_image(image_path, enhanced_bordered_extraction_image_kv)
        else:
            # Show enhanced by default for comparison
            visualize_bordered_tables_image(image_path, enhanced_bordered_extraction_image_kv)

if __name__ == "__main__":
    # Check if running in interactive mode
    use_interactive = input("Use interactive mode for Bordered Tables extraction from images? (y/n, default=y): ").strip().lower()

    if use_interactive != 'n':
        # Interactive mode
        interactive_bordered_analysis_image()
    else:
        # Default image path
        image_path = "/path/to/your/table_image.jpg"

        # Check file exists
        if not os.path.exists(image_path):
            print(f"Error: File not found: {image_path}")
            print("Please provide a valid image path")
            exit()

        # Run all bordered table extractions
        print("=== PRIMARY BORDERED EXTRACTION COLUMN-ORIENTED OUTPUT ===")
        result1 = primary_extraction_bordered_image_kv(image_path)
        print(json.dumps(result1, indent=2, ensure_ascii=False))

        # Convert to DataFrame and display
        df1 = convert_to_dataframe(result1)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df1)

        print("\n=== ENHANCED BORDERED EXTRACTION WITH ROW GROUPING COLUMN-ORIENTED OUTPUT ===")
        result2 = enhanced_bordered_extraction_image_kv(image_path)
        print(json.dumps(result2, indent=2, ensure_ascii=False))

        # Convert to DataFrame and display
        df2 = convert_to_dataframe(result2)
        print("\n=== DATAFRAME OUTPUT ===")
        print(df2)

        # Show visualization
        print("\nGenerating visualization...")
        visualize_bordered_tables_image(image_path, enhanced_bordered_extraction_image_kv)