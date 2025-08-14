#!/usr/bin/env python3
"""
Main script for Template-Agnostic PDF and Image Processing
Configures logging and runs the processing workflow
"""
import logging
import traceback
import json  # Add this import
from document_processor import TemplateAgnosticProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_image_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the Template-Agnostic PDF and Image Processing
    """
    # Configuration - Update these paths as needed
    FIELD_INFO_PATH = r"C:\Users\SyedaZuberiya\Downloads\full_pdf_extracted_data_auto_tag.json"
    FILE_PATH = r"C:\Users\SyedaZuberiya\Desktop\PyMuPDF Testing\Payment_Advice_F4.pdf"  # Can be PDF or image
    OUTPUT_DIRECTORY = "template_extraction_results"
    
    try:
        logger.info("Starting Template-Agnostic PDF and Image Processing")
        
        # Initialize processor
        processor = TemplateAgnosticProcessor(FIELD_INFO_PATH)
        logger.info(f"Processor initialized for template: {processor.template_name}")
        
        # Process file(s)
        results = processor.process_file_batch(FILE_PATH, OUTPUT_DIRECTORY)
        
        # Display extracted data
        processor.display_extracted_data(results)
        
        # Print final summary
        print("\n" + "="*80)
        print("PROCESSING COMPLETED")
        print("="*80)
        print(f"Template: {results['template_name']}")
        print(f"Total files processed: {results['total_files']}")
        print(f"Successful: {results['successful_files']}")
        print(f"Failed: {results['failed_files']}")
        if results['total_files'] > 0:
            success_percentage = (results['successful_files'] / results['total_files']) * 100
            print(f"Overall Success Rate: {success_percentage:.1f}%")
        print(f"Results saved to: {OUTPUT_DIRECTORY}")
        print("="*80)
        
        # Output final results as JSON
        print("\nFinal JSON Output:")
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        logger.error(traceback.format_exc())
        print(f"\nProcessing failed: {e}")
        
        # Output error as JSON
        error_output = {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }
        print("\nFinal JSON Output:")
        print(json.dumps(error_output, indent=2))

if __name__ == "__main__":
    main()