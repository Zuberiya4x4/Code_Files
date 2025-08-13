import fitz  # PyMuPDF
import pandas as pd
import re
import os

# Load coordinate template
def parse_template_file(template_path):
    with open(template_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract field data
    fields = re.findall(
        r"FIELD: (.*?)\n[-]+\nValue: (.*?)\nConfidence: (.*?)%\n\nHeader Bounding Box:\n  \(x0, y0, x1, y1\): \((.*?)\)\n\nValue Bounding Box(?:\(es\))?:\n(.*?)\n\nCombined Bounding Box.*?\[(.*?)\]",
        content, re.DOTALL)

    field_data = []
    for field in fields:
        name, value, conf, header_box, val_boxes_raw, combined_box = field
        val_boxes = re.findall(r'\(x0, y0, x1, y1\): \((.*?)\)', val_boxes_raw)
        field_data.append({
            'field': name.strip(),
            'value': value.strip(),
            'confidence': float(conf.strip()),
            'value_boxes': [tuple(map(float, v.split(', '))) for v in val_boxes],
            'combined_box': tuple(map(float, combined_box.split(', ')))
        })

    # Extract table data
    tables = re.findall(
        r"--- Table_(.*?) ---\nTable bbox: \((.*?)\)\n.*?Grid: (\d+) rows x (\d+) columns.*?CELL COORDINATES:\n(.*?)\n\n",
        content, re.DOTALL)

    table_data = []
    for table_id, bbox, rows, cols, cells in tables:
        cell_info = re.findall(r"R(\d+)C(\d+): \((.*?)\) = '(.*?)'", cells)
        table_cells = {}
        for r, c, coords, val in cell_info:
            table_cells[(int(r), int(c))] = {
                'bbox': tuple(map(float, coords.split(', '))),
                'text': val
            }
        table_data.append({
            'table_id': table_id.strip(),
            'bbox': tuple(map(float, bbox.split(', '))),
            'rows': int(rows),
            'cols': int(cols),
            'cells': table_cells
        })

    return field_data, table_data


# Extract text from PDF within a bounding box
def extract_text_in_bbox(page, bbox):
    rect = fitz.Rect(bbox)
    words = page.get_text("words")  # Each word = (x0, y0, x1, y1, "text", ...)
    return ' '.join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect)).strip()


# Extract fields from PDF using template
def extract_fields_from_pdf(pdf_path, field_templates):
    doc = fitz.open(pdf_path)
    page = doc[0]  # Assuming template matches only 1st page

    extracted_fields = {}
    for field in field_templates:
        field_value = ''
        for box in field['value_boxes']:
            text = extract_text_in_bbox(page, box)
            if text:
                field_value += text + ' '
        extracted_fields[field['field']] = field_value.strip()
    return extracted_fields


# Extract tables from PDF using template coordinates
def extract_table_from_pdf(pdf_path, table_templates):
    doc = fitz.open(pdf_path)
    page = doc[0]
    all_tables = []

    for tbl in table_templates:
        rows, cols = tbl['rows'], tbl['cols']
        data = [[''] * cols for _ in range(rows)]

        for (r, c), cell in tbl['cells'].items():
            bbox = cell['bbox']
            rect = fitz.Rect(bbox)
            words = page.get_text("words")
            cell_text = ' '.join(w[4] for w in words if fitz.Rect(w[:4]).intersects(rect)).strip()
            data[r][c] = cell_text

        df = pd.DataFrame(data)
        all_tables.append((tbl['table_id'], df))
    return all_tables


# Main function
def extract_data_from_pdf_using_template(template_txt_path, target_pdf_path, output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    print("Parsing template...")
    field_templates, table_templates = parse_template_file(template_txt_path)

    print("Extracting fields...")
    fields = extract_fields_from_pdf(target_pdf_path, field_templates)
    for k, v in fields.items():
        print(f"{k}: {v}")
    
    print("\nExtracting tables...")
    tables = extract_table_from_pdf(target_pdf_path, table_templates)
    for table_id, df in tables:
        print(f"\nTable {table_id}")
        print(df)
        df.to_csv(os.path.join(output_dir, f"Table_{table_id}.csv"), index=False)

    # Save extracted fields
    pd.DataFrame([fields]).to_csv(os.path.join(output_dir, "extracted_fields.csv"), index=False)
    print(f"\nExtraction completed. Files saved in: {output_dir}")


# ===== Example usage =====
if __name__ == "__main__":
    template_txt_file = r"C:\Users\SyedaZuberiya\Downloads\combined_output (1).txt"
    new_pdf_file = r"C:\Users\SyedaZuberiya\Desktop\PyMuPDF Testing\Payment_Advice_F6.pdf"

    extract_data_from_pdf_using_template(template_txt_file, new_pdf_file)
